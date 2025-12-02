import os
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
import sys
import time
from einops import rearrange

# 프로젝트 루트 디렉토리를 절대 경로로 추가
project_root = osp.dirname(osp.dirname(osp.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from Retinotopy.dataset.HCP_3sets_ROI import Retinotopy
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SplineConv

# Physics Attention 모듈 (Transolver에서 가져옴)
class Physics_Attention_Irregular_Mesh(nn.Module):
    """Physics Attention for irregular meshes - edge 정보 사용 안 함"""
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., slice_num=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)

        self.in_project_x = nn.Linear(dim, inner_dim)
        self.in_project_fx = nn.Linear(dim, inner_dim)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        for l in [self.in_project_slice]:
            torch.nn.init.orthogonal_(l.weight)
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x: (B, N, C) 형태만 받음, edge 정보는 사용 안 함
        B, N, C = x.shape

        # (1) Slice: 물리적 상태를 슬라이스 토큰으로 변환
        fx_mid = self.in_project_fx(x).reshape(B, N, self.heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()  # B H N C
        x_mid = self.in_project_x(x).reshape(B, N, self.heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()  # B H N C
        slice_weights = self.softmax(self.in_project_slice(x_mid) / self.temperature)  # B H N G
        slice_norm = slice_weights.sum(2)  # B H G
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
        slice_token = slice_token / ((slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head))

        # (2) Attention among slice tokens
        q_slice_token = self.to_q(slice_token)
        k_slice_token = self.to_k(slice_token)
        v_slice_token = self.to_v(slice_token)
        dots = torch.matmul(q_slice_token, k_slice_token.transpose(-1, -2)) * self.scale
        attn = self.softmax(dots)
        attn = self.dropout(attn)
        out_slice_token = torch.matmul(attn, v_slice_token)  # B H G D

        # (3) Deslice: 다시 노드로 변환
        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights)
        out_x = rearrange(out_x, 'b h n d -> b n (h d)')
        return self.to_out(out_x)


class deepRetinotopy_OptionA(torch.nn.Module):
    """하이브리드 모델: SplineConv + Physics Attention (edge 정보 미사용)"""
    def __init__(self, num_features):
        super(deepRetinotopy_OptionA, self).__init__()
        # 초기 SplineConv 레이어들
        self.conv1 = SplineConv(num_features, 8, dim=3, kernel_size=25)
        self.bn1 = torch.nn.BatchNorm1d(8)
        self.conv2 = SplineConv(8, 16, dim=3, kernel_size=25)
        self.bn2 = torch.nn.BatchNorm1d(16)
        self.conv3 = SplineConv(16, 32, dim=3, kernel_size=25)
        self.bn3 = torch.nn.BatchNorm1d(32)

        # Physics Attention 블록 1 (conv3 이후)
        self.phys_attn1 = Physics_Attention_Irregular_Mesh(dim=32, heads=8, dim_head=32//8, 
                                                           dropout=0.1, slice_num=32)
        self.ln1 = nn.LayerNorm(32)
        self.mlp1 = nn.Sequential(
            nn.Linear(32, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 32),
            nn.Dropout(0.1)
        )

        # 중간 SplineConv 레이어들
        self.conv4 = SplineConv(32, 32, dim=3, kernel_size=25)
        self.bn4 = torch.nn.BatchNorm1d(32)
        self.conv5 = SplineConv(32, 32, dim=3, kernel_size=25)
        self.bn5 = torch.nn.BatchNorm1d(32)
        self.conv6 = SplineConv(32, 32, dim=3, kernel_size=25)
        self.bn6 = torch.nn.BatchNorm1d(32)

        # Physics Attention 블록 2 (conv6 이후)
        self.phys_attn2 = Physics_Attention_Irregular_Mesh(dim=32, heads=8, dim_head=32//8,
                                                           dropout=0.1, slice_num=32)
        self.ln2 = nn.LayerNorm(32)
        self.mlp2 = nn.Sequential(
            nn.Linear(32, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 32),
            nn.Dropout(0.1)
        )

        # 후반 SplineConv 레이어들
        self.conv7 = SplineConv(32, 32, dim=3, kernel_size=25)
        self.bn7 = torch.nn.BatchNorm1d(32)
        self.conv8 = SplineConv(32, 32, dim=3, kernel_size=25)
        self.bn8 = torch.nn.BatchNorm1d(32)
        self.conv9 = SplineConv(32, 32, dim=3, kernel_size=25)
        self.bn9 = torch.nn.BatchNorm1d(32)
        self.conv10 = SplineConv(32, 16, dim=3, kernel_size=25)
        self.bn10 = torch.nn.BatchNorm1d(16)
        self.conv11 = SplineConv(16, 8, dim=3, kernel_size=25)
        self.bn11 = torch.nn.BatchNorm1d(8)
        self.conv12 = SplineConv(8, 1, dim=3, kernel_size=25)

    def forward(self, data):
        x, edge_index, pseudo = data.x, data.edge_index, data.edge_attr
        
        # 초기 SplineConv 레이어들 (edge 정보 사용)
        x = F.elu(self.conv1(x, edge_index, pseudo))
        x = self.bn1(x)
        x = F.dropout(x, p=.10, training=self.training)
        x = F.elu(self.conv2(x, edge_index, pseudo))
        x = self.bn2(x)
        x = F.dropout(x, p=.10, training=self.training)
        x = F.elu(self.conv3(x, edge_index, pseudo))
        x = self.bn3(x)
        x = F.dropout(x, p=.10, training=self.training)

        # Physics Attention 블록 1 (edge 정보 미사용)
        # (N, C) -> (1, N, C)로 변환
        x_batch = x.unsqueeze(0)  # (1, N, 32)
        x_residual = x_batch
        x_batch = self.phys_attn1(self.ln1(x_batch)) + x_residual  # Residual connection
        x_batch = self.mlp1(x_batch) + x_batch  # MLP with residual
        x = x_batch.squeeze(0)  # (N, 32)로 복원

        # 중간 SplineConv 레이어들 (edge 정보 사용)
        x = F.elu(self.conv4(x, edge_index, pseudo))
        x = self.bn4(x)
        x = F.dropout(x, p=.10, training=self.training)
        x = F.elu(self.conv5(x, edge_index, pseudo))
        x = self.bn5(x)
        x = F.dropout(x, p=.10, training=self.training)
        x = F.elu(self.conv6(x, edge_index, pseudo))
        x = self.bn6(x)
        x = F.dropout(x, p=.10, training=self.training)

        # Physics Attention 블록 2 (edge 정보 미사용)
        x_batch = x.unsqueeze(0)  # (1, N, 32)
        x_residual = x_batch
        x_batch = self.phys_attn2(self.ln2(x_batch)) + x_residual
        x_batch = self.mlp2(x_batch) + x_batch
        x = x_batch.squeeze(0)  # (N, 32)로 복원

        # 후반 SplineConv 레이어들 (edge 정보 사용)
        x = F.elu(self.conv7(x, edge_index, pseudo))
        x = self.bn7(x)
        x = F.dropout(x, p=.10, training=self.training)
        x = F.elu(self.conv8(x, edge_index, pseudo))
        x = self.bn8(x)
        x = F.dropout(x, p=.10, training=self.training)
        x = F.elu(self.conv9(x, edge_index, pseudo))
        x = self.bn9(x)
        x = F.dropout(x, p=.10, training=self.training)
        x = F.elu(self.conv10(x, edge_index, pseudo))
        x = self.bn10(x)
        x = F.dropout(x, p=.10, training=self.training)
        x = F.elu(self.conv11(x, edge_index, pseudo))
        x = self.bn11(x)
        x = F.dropout(x, p=.10, training=self.training)
        x = F.elu(self.conv12(x, edge_index, pseudo)).view(-1)
        return x


path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'Retinotopy', 'data')

pre_transform = T.Compose([T.FaceToEdge()])
train_dataset = Retinotopy(path, 'Train', transform=T.Cartesian(),
                           pre_transform=pre_transform, n_examples=181,
                           prediction='eccentricity', myelination=True,
                           hemisphere='Left')
dev_dataset = Retinotopy(path, 'Development', transform=T.Cartesian(),
                         pre_transform=pre_transform, n_examples=181,
                         prediction='eccentricity', myelination=True,
                         hemisphere='Left')
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=1, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = deepRetinotopy_OptionA(num_features=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train(epoch):
    model.train()
    if epoch == 100:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.005

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()

        R2 = data.R2.view(-1)
        threshold = R2.view(-1) > 2.2

        loss = torch.nn.SmoothL1Loss()
        output_loss = loss(R2 * model(data), R2 * data.y.view(-1))
        output_loss.backward()

        MAE = torch.mean(abs(
            data.y.view(-1)[threshold == 1] - model(data)[threshold == 1])).item()

        optimizer.step()
    return output_loss.detach(), MAE


def test():
    model.eval()
    MeanAbsError = 0
    MeanAbsError_thr = 0
    y = []
    y_hat = []
    R2_plot = []

    for data in dev_loader:
        pred = model(data.to(device)).detach()
        y_hat.append(pred)
        y.append(data.to(device).y.view(-1))

        R2 = data.R2.view(-1)
        R2_plot.append(R2)
        threshold = R2.view(-1) > 2.2
        threshold2 = R2.view(-1) > 17

        MAE = torch.mean(abs(data.to(device).y.view(-1)[threshold == 1] - pred[
            threshold == 1])).item()
        MAE_thr = torch.mean(abs(
            data.to(device).y.view(-1)[threshold2 == 1] - pred[
                threshold2 == 1])).item()
        MeanAbsError_thr += MAE_thr
        MeanAbsError += MAE

    test_MAE = MeanAbsError / len(dev_loader)
    test_MAE_thr = MeanAbsError_thr / len(dev_loader)
    output = {'Predicted_values': y_hat, 'Measured_values': y, 'R2': R2_plot,
              'MAE': test_MAE, 'MAE_thr': test_MAE_thr}
    return output


# Create an output folder if it doesn't already exist
directory = './output'
if not osp.exists(directory):
    os.makedirs(directory)

# Model training
for epoch in range(1, 201):
    loss, MAE = train(epoch)
    test_output = test()
    print(
        'Epoch: {:02d}, Train_loss: {:.4f}, Train_MAE: {:.4f}, Test_MAE: {'
        ':.4f}, Test_MAE_thr: {:.4f}'.format(
            epoch, loss, MAE, test_output['MAE'], test_output['MAE_thr']))
    if epoch % 25 == 0:
        torch.save({'Epoch': epoch,
                    'Predicted_values': test_output['Predicted_values'],
                    'Measured_values': test_output['Measured_values'],
                    'R2': test_output['R2'], 'Loss': loss,
                    'Dev_MAE': test_output['MAE']},
                   osp.join(osp.dirname(osp.realpath(__file__)),
                            'output',
                            'deepRetinotopy_ecc_LH_optionA_output_epoch' + str(
                                epoch) + '.pt'))

torch.save(model.state_dict(),
           osp.join(osp.dirname(osp.realpath(__file__)), 'output',
                    'deepRetinotopy_ecc_LH_optionA_model.pt'))