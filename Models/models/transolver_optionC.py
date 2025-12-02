"""
Transolver Option C: Pure Transolver structure (100% identical to original Transolver)
MLP preprocessing + Transolver blocks (Physics Attention + MLP)
Edge information is not used (Physics Attention works on node features only)
"""
import torch
import torch.nn as nn
import numpy as np
from models.physics_attention import Physics_Attention


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    """
    Truncated normal distribution initialization (same as timm.models.layers.trunc_normal_)
    Fills the input Tensor with values drawn from a truncated normal distribution.
    """
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        sqrt_2 = torch.tensor(np.sqrt(2.), dtype=x.dtype, device=x.device)
        return (1. + torch.erf(x / sqrt_2)) / 2.

    with torch.no_grad():
        # Convert to tensor if needed
        mean = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device)
        std = torch.tensor(std, dtype=tensor.dtype, device=tensor.device)
        a = torch.tensor(a, dtype=tensor.dtype, device=tensor.device)
        b = torch.tensor(b, dtype=tensor.dtype, device=tensor.device)
        
        # Get bounds
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        # Uniformly fill tensor with values from [l, u], then transform to [a, b]
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        sqrt_2 = torch.tensor(np.sqrt(2.), dtype=tensor.dtype, device=tensor.device)
        tensor.mul_(std * sqrt_2)
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


ACTIVATION = {'gelu': nn.GELU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid, 'relu': nn.ReLU, 
              'leaky_relu': nn.LeakyReLU(0.1), 'softplus': nn.Softplus, 'ELU': nn.ELU, 'silu': nn.SiLU}


class MLP(nn.Module):
    """MLP module from Transolver"""
    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act='gelu', res=True):
        super(MLP, self).__init__()

        if act in ACTIVATION.keys():
            act = ACTIVATION[act]
        else:
            raise NotImplementedError
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_layers = n_layers
        self.res = res
        self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden), act())
        self.linear_post = nn.Linear(n_hidden, n_output)
        self.linears = nn.ModuleList([nn.Sequential(nn.Linear(n_hidden, n_hidden), act()) for _ in range(n_layers)])

    def forward(self, x):
        x = self.linear_pre(x)
        for i in range(self.n_layers):
            if self.res:
                x = self.linears[i](x) + x
            else:
                x = self.linears[i](x)
        x = self.linear_post(x)
        return x


class Transolver_block(nn.Module):
    """Transformer encoder block from Transolver"""
    def __init__(
            self,
            num_heads: int,
            hidden_dim: int,
            dropout: float,
            act='gelu',
            mlp_ratio=4,
            last_layer=False,
            out_dim=1,
            slice_num=32,
    ):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.Attn = Physics_Attention(
            hidden_dim, 
            heads=num_heads, 
            dim_head=hidden_dim // num_heads,
            dropout=dropout, 
            slice_num=slice_num
        )
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim, n_layers=0, res=False, act=act)
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, fx):
        fx = self.Attn(self.ln_1(fx)) + fx
        fx = self.mlp(self.ln_2(fx)) + fx
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        else:
            return fx


class deepRetinotopy_OptionC(torch.nn.Module):
    """
    Pure Transolver structure (100% identical to original Transolver_Irregular_Mesh)
    - MLP preprocessing
    - Transolver blocks (Physics Attention + MLP)
    - Edge information is not used (Physics Attention works on node features only)
    """
    def __init__(self, num_features, 
                 space_dim=3,
                 n_layers=5,
                 n_hidden=128,
                 dropout=0.0,
                 n_head=8,
                 act='gelu',
                 mlp_ratio=1,
                 slice_num=32,
                 ref=8,
                 unified_pos=False):
        super(deepRetinotopy_OptionC, self).__init__()
        self.space_dim = space_dim
        self.ref = ref
        self.unified_pos = unified_pos
        self.n_hidden = n_hidden
        self.num_features = num_features  # Store for reference

        # Preprocessing MLP with fixed input dimension: num_features + space_dim
        preprocess_input_dim = num_features + space_dim
        self.preprocess = MLP(
            preprocess_input_dim,
            n_hidden * 2,
            n_hidden,
            n_layers=0,
            res=False,
            act=act
        )

        # Transolver blocks
        self.blocks = nn.ModuleList([
            Transolver_block(
                num_heads=n_head, 
                hidden_dim=n_hidden,
                dropout=dropout,
                act=act,
                mlp_ratio=mlp_ratio,
                out_dim=1,
                slice_num=slice_num,
                last_layer=(_ == n_layers - 1)
            )
            for _ in range(n_layers)
        ])
        
        # Placeholder parameter (same as Transolver)
        self.placeholder = nn.Parameter((1 / (n_hidden)) * torch.rand(n_hidden, dtype=torch.float))
        
        # Initialize weights (including preprocess)
        self.apply(self._init_weights)

    def initialize_weights(self):
        """Initialize weights same as Transolver"""
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Weight initialization from Transolver"""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_grid(self, x, batchsize=1):
        """
        Generate grid reference positions (same as Transolver)
        x: B N space_dim (coordinates)
        Returns: B N (ref*ref) (distance to grid points)
        """
        device = x.device
        gridx = torch.tensor(np.linspace(0, 1, self.ref), dtype=torch.float, device=device)
        gridx = gridx.reshape(1, self.ref, 1, 1).repeat([batchsize, 1, self.ref, 1])
        gridy = torch.tensor(np.linspace(0, 1, self.ref), dtype=torch.float, device=device)
        gridy = gridy.reshape(1, 1, self.ref, 1).repeat([batchsize, self.ref, 1, 1])
        grid_ref = torch.cat((gridx, gridy), dim=-1).reshape(batchsize, self.ref * self.ref, 2)  # B (ref*ref) 2

        # Compute distance from each node to each grid point
        pos = torch.sqrt(torch.sum((x[:, :, None, :] - grid_ref[:, None, :, :]) ** 2, dim=-1)). \
            reshape(batchsize, x.shape[1], self.ref * self.ref).contiguous()
        return pos

    def forward(self, data):
        """
        Forward pass (adapted for torch_geometric Data format)
        data: torch_geometric Data object with x (node features) and pos (coordinates)
        """
        # Extract coordinates and features
        # data.x: (N, actual_features) - node features (may include coordinates)
        # data.pos: (N, space_dim) - node coordinates (if available)
        
        x = data.x  # Node features: (N, actual_features)
        N = x.shape[0]

        # Get coordinates from data.pos if available
        if hasattr(data, 'pos') and data.pos is not None:
            coords = data.pos  # (N, space_dim)
            x_features = x  # (N, num_features)
        else:
            coords = x[:, :self.space_dim]  # (N, space_dim)
            x_features = x[:, self.space_dim:] if x.shape[1] > self.space_dim else x  # (N, num_features)

        # Always use the fixed feature/coord concat scheme since preprocess expects num_features+space_dim input
        if x_features.dim() == 1:
            x_features = x_features.unsqueeze(1)
        if coords.dim() == 1:
            coords = coords.unsqueeze(1)

        x_batch = x_features.unsqueeze(0) if x_features.dim() == 2 else x_features  # (1, N, F)
        coords_batch = coords.unsqueeze(0) if coords.dim() == 2 else coords         # (1, N, space_dim)

        # Concatenate (Features, Coordinates) for each node
        fx = torch.cat([x_batch, coords_batch], dim=-1)   # (1, N, num_features + space_dim)
        # Check: If input has more dims, squeeze down to (1, N, C)
        if fx.dim() == 2:
            fx = fx.unsqueeze(0)

        # MLP preprocessing
        fx = self.preprocess(fx)  # (1, N, n_hidden)

        # Add placeholder (same as Transolver)
        fx = fx + self.placeholder[None, None, :]

        # Transolver blocks
        for block in self.blocks:
            fx = block(fx)
        
        # Output: (1, N, 1) -> (N,)
        return fx.squeeze(0).squeeze(-1)  # (N,)

