import os
import os.path as osp
import sys
import time
import argparse
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T

# Neptune import and initialization
import neptune

# =============================
# Argument Parser Setup
# =============================
parser = argparse.ArgumentParser(
    description="Train deepRetinotopy (eccentricity) model with Neptune logging."
)
parser.add_argument('--project', type=str, default='common/project',
                    help='Neptune project name (e.g., user/project)')
parser.add_argument('--api_token', type=str, default='YOUR_API_KEY',
                    help='Neptune API token or use NEPTUNE_API_TOKEN env variable')
parser.add_argument('--hemisphere', type=str, default='Left', choices=['Left', 'Right'],
                    help='Hemisphere to use for prediction')
parser.add_argument('--batch_size', type=int, default=1,
                    help='Batch size for loader')
parser.add_argument('--n_epochs', type=int, default=200,
                    help='Total number of epochs to train')
parser.add_argument('--lr_init', type=float, default=0.01,
                    help='Initial learning rate')
parser.add_argument('--lr_decay_epoch', type=int, default=100,
                    help='Epoch at which learning rate decays')
parser.add_argument('--lr_decay', type=float, default=0.005,
                    help='Learning rate after decay')
parser.add_argument('--interm_save_every', type=int, default=25,
                    help='Interval (epoch) for saving intermediate predictions')
parser.add_argument('--output_dir', type=str, default='./output',
                    help='Directory to save outputs')
parser.add_argument('--n_examples', type=int, default=181,
                    help='Number of examples per split')
parser.add_argument('--prediction', type=str, default='eccentricity',
                    help='Prediction target (default: eccentricity)')
parser.add_argument('--myelination', type=bool, default=True,
                    help='Use myelination as feature (default: True)')

args = parser.parse_args()

# =============================
# Path, Import, Dataset Preparation
# =============================

# Add project root directory to sys.path as absolute path
project_root = osp.dirname(osp.dirname(osp.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from Retinotopy.dataset.HCP_3sets_ROI import Retinotopy
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SplineConv

# Create Neptune run (API token and project name from arguments)
run = neptune.init_run(
    project=args.project,
    api_token=args.api_token
)

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'Retinotopy', 'data')

pre_transform = T.Compose([T.FaceToEdge()])
train_dataset = Retinotopy(
    path, 'Train',
    transform=T.Cartesian(),
    pre_transform=pre_transform,
    n_examples=args.n_examples,
    prediction=args.prediction,
    myelination=args.myelination,
    hemisphere=args.hemisphere
)
dev_dataset = Retinotopy(
    path, 'Development',
    transform=T.Cartesian(),
    pre_transform=pre_transform,
    n_examples=args.n_examples,
    prediction=args.prediction,
    myelination=args.myelination,
    hemisphere=args.hemisphere
)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False)

# Log main config (all argument values to Neptune)
run["config"] = vars(args)
run["config/architecture"] = "deepRetinotopy"
run["config/optimizer"] = "Adam"
run["config/loss_fn"] = "SmoothL1Loss"
run["config/dataset"] = "HCP_3sets_ROI"
run["config/train_size"] = len(train_dataset)
run["config/dev_size"] = len(dev_dataset)

class deepRetinotopy(torch.nn.Module):
    def __init__(self, num_features):
        super(deepRetinotopy, self).__init__()
        self.conv1 = SplineConv(num_features, 8, dim=3, kernel_size=25)
        self.bn1 = torch.nn.BatchNorm1d(8)

        self.conv2 = SplineConv(8, 16, dim=3, kernel_size=25)
        self.bn2 = torch.nn.BatchNorm1d(16)

        self.conv3 = SplineConv(16, 32, dim=3, kernel_size=25)
        self.bn3 = torch.nn.BatchNorm1d(32)

        self.conv4 = SplineConv(32, 32, dim=3, kernel_size=25)
        self.bn4 = torch.nn.BatchNorm1d(32)

        self.conv5 = SplineConv(32, 32, dim=3, kernel_size=25)
        self.bn5 = torch.nn.BatchNorm1d(32)

        self.conv6 = SplineConv(32, 32, dim=3, kernel_size=25)
        self.bn6 = torch.nn.BatchNorm1d(32)

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
        x = F.elu(self.conv1(x, edge_index, pseudo))
        x = self.bn1(x)
        x = F.dropout(x, p=.10, training=self.training)

        x = F.elu(self.conv2(x, edge_index, pseudo))
        x = self.bn2(x)
        x = F.dropout(x, p=.10, training=self.training)

        x = F.elu(self.conv3(x, edge_index, pseudo))
        x = self.bn3(x)
        x = F.dropout(x, p=.10, training=self.training)

        x = F.elu(self.conv4(x, edge_index, pseudo))
        x = self.bn4(x)
        x = F.dropout(x, p=.10, training=self.training)

        x = F.elu(self.conv5(x, edge_index, pseudo))
        x = self.bn5(x)
        x = F.dropout(x, p=.10, training=self.training)

        x = F.elu(self.conv6(x, edge_index, pseudo))
        x = self.bn6(x)
        x = F.dropout(x, p=.10, training=self.training)

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = deepRetinotopy(num_features=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_init)

def train(epoch):
    model.train()

    if epoch == args.lr_decay_epoch:
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr_decay
            run["lr_current"] = args.lr_decay  # Log lr schedule to Neptune

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()

        R2 = data.R2.view(-1)
        threshold = R2.view(-1) > 2.2

        loss = torch.nn.SmoothL1Loss()
        output_loss = loss(R2 * model(data), R2 * data.y.view(-1))
        output_loss.backward()

        MAE = torch.mean(abs(
            data.y.view(-1)[threshold == 1] - model(data)[
                threshold == 1])).item()  # To check the performance of the model while training

        optimizer.step()
    # Log epoch-wise train loss/MAE to Neptune
    run["train/loss"].append(output_loss.detach().cpu().item())
    run["train/mae"].append(MAE)
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
            threshold == 1])).item()  # To check the performance of the
        # model while training
        MAE_thr = torch.mean(abs(
            data.to(device).y.view(-1)[threshold2 == 1] - pred[
                threshold2 == 1])).item()  # To check the performance of the
        # model while training
        MeanAbsError_thr += MAE_thr
        MeanAbsError += MAE

    test_MAE = MeanAbsError / len(dev_loader)
    test_MAE_thr = MeanAbsError_thr / len(dev_loader)
    output = {'Predicted_values': y_hat, 'Measured_values': y, 'R2': R2_plot,
              'MAE': test_MAE, 'MAE_thr': test_MAE_thr}
    # Log validation MAE/MAE_thr to Neptune
    run["dev/mae"].append(test_MAE)
    run["dev/mae_thr"].append(test_MAE_thr)
    return output

# Create an output folder if it doesn't already exist
if not osp.exists(args.output_dir):
    os.makedirs(args.output_dir)

# Model training
for epoch in range(1, args.n_epochs + 1):
    loss, MAE = train(epoch)
    test_output = test()
    print(
        'Epoch: {:02d}, Train_loss: {:.4f}, Train_MAE: {:.4f}, Test_MAE: {'
        ':.4f}, Test_MAE_thr: {:.4f}'.format(
            epoch, loss, MAE, test_output['MAE'], test_output['MAE_thr']))
    # Log epoch information to Neptune
    run["epoch"] = epoch
    run["monitor/loss"] = loss.cpu().item()
    run["monitor/mae"] = MAE
    run["monitor/val_mae"] = test_output['MAE']
    run["monitor/val_mae_thr"] = test_output['MAE_thr']

    if epoch % args.interm_save_every == 0:  # To save intermediate predictions
        output_path = osp.join(
            osp.dirname(osp.realpath(__file__)),
            args.output_dir,
            f'deepRetinotopy_ecc_{args.hemisphere}_output_epoch{epoch}.pt'
        )
        torch.save({'Epoch': epoch,
                    'Predicted_values': test_output['Predicted_values'],
                    'Measured_values': test_output['Measured_values'],
                    'R2': test_output['R2'], 'Loss': loss,
                    'Dev_MAE': test_output['MAE']},
                   output_path)
        # Log intermediate results to Neptune
        run[f"intermediate/epoch_{epoch}/dev_mae"] = test_output["MAE"]
        run[f"intermediate/epoch_{epoch}/dev_mae_thr"] = test_output["MAE_thr"]
        # Example: Save some predicted values (comment out if large data)
        # run[f"intermediate/epoch_{epoch}/y_hat"] = [y.cpu().numpy() for y in test_output["Predicted_values"]][:5]

# Saving model's learned parameters
final_model_path = osp.join(
    osp.dirname(osp.realpath(__file__)),
    args.output_dir,
    f'deepRetinotopy_ecc_{args.hemisphere}_model.pt')
torch.save(model.state_dict(), final_model_path)
run["artifacts/model_state_dict"].upload(final_model_path)

# Neptune run stop
run.stop()