import argparse
from tqdm import tqdm

import torch
from torch.cuda.amp import autocast, GradScaler
import os

from datasets import Dataset
from utils import Logger, get_parameter_groups, get_lr_scheduler_with_warmup
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import dgl
import hashlib
from baseline_model import ResidualModuleWrapper, FeedForwardModule, compute_spline_basis


# call with nohup python /home/samustac/WORK/USC/STAC/CauchyNet/week_1/slpitnet_euler.py   --name SGWT_l14   --dataset minesweeper   --model SGWT   --num_layers 14   --device cuda:0   > sgwt
#_l14_minesweeper.out 2>&1 &

def plot_filter_evolution(model, step, run_id, save_dir='fig_dump'):
    """
    Plots the frequency response of the learned filters.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    # We only visualize the first layer's filter banks for simplicity
    first_sgwt_layer = None
    for m in model.residual_modules:
        if isinstance(m.module, SGWTModule):
            first_sgwt_layer = m.module
            break
            
    if first_sgwt_layer is None:
        return

    device = next(model.parameters()).device
    
    # 1. Create a smooth domain for lambda [0, 1]
    lam_smooth = torch.linspace(0, 1, 200, device=device)
    lam_np = lam_smooth.cpu().numpy()  # <--- add this
    
    # 2. Compute Spline Basis for this smooth domain
    # We reuse the module's parameters to match the training setup
    K = first_sgwt_layer.number_of_filters
    degree = first_sgwt_layer.spline_degree
    basis_matrix = compute_spline_basis(lam_smooth, K, degree) # [200, K]
    
    # 3. Plot responses: global + parts in one figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Global
    ax = axes[0]
    with torch.no_grad():
        f = first_sgwt_layer.spectral_wavelet_filter
        resp = basis_matrix @ f.coeffs
        if resp.ndim == 1:
            avg_response = resp.cpu().numpy()
        else:
            avg_response = resp.mean(dim=1).cpu().numpy()
        ax.plot(lam_np, avg_response, linewidth=2.0, label='Global Filter')
        ax.fill_between(lam_np, avg_response, alpha=0.1)    
    ax.set_title(f'Global Filters (Run {run_id}, Step {step})')
    ax.set_xlabel('Normalized Frequency (λ)')
    ax.set_ylabel('Avg Magnitude')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Helper to plot a part
    def plot_part(ax, part_key, filt_list, lam_norm):
        if lam_norm is None or first_sgwt_layer.B_part.get(part_key) is None:
            ax.set_title(f'Part {part_key}: N/A')
            ax.axis('off')
            return
        basis_part = first_sgwt_layer.B_part[part_key]
        lam_part = lam_norm.to(device)
        lam_part_np = lam_part.cpu().numpy()
        with torch.no_grad():
            f = first_sgwt_layer.spectral_wavelet_filter
            resp = basis_matrix @ f.coeffs
            if resp.ndim == 1:
                avg_response = resp.cpu().numpy()
            else:
                avg_response = resp.mean(dim=1).cpu().numpy()
            ax.plot(lam_np, avg_response, linewidth=2.0, label='Global Filter')
            ax.fill_between(lam_np, avg_response, alpha=0.1)        
        ax.set_title(f'Part {part_key} Filters')
        ax.set_xlabel('Normalized Frequency (λ)')
        ax.set_ylabel('Avg Magnitude')
        ax.grid(True, alpha=0.3)
        ax.legend()

    plot_part(axes[1], 'S', first_sgwt_layer.spectral_wavelet_filter_S, lam_S_normalized)
    plot_part(axes[2], 'T', first_sgwt_layer.spectral_wavelet_filter_T, lam_T_normalized)

    plt.tight_layout()
    filename = os.path.join(save_dir, f'filters_run{run_id}_step{step:04d}.png')
    plt.savefig(filename, dpi=150)
    plt.close()
        

class Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, module, hidden_dim_multiplier=1, num_heads=0,
                 dropout=0.2, inner_dropout=0.2, num_layers=2, share_filters=False, initial_step_global=0.03, 
                 initial_step_sides=0.12, num_filters_global=6, num_filters_S=6, num_filters_T=6, 
                 initial_step_rmw=1.0, filter_init_mode='band_pass'):
        super().__init__()
        
        self.residual_modules = nn.ModuleList()
        if share_filters:
            shared_filters = SharedSpectralFilters(dim=hidden_dim, number_of_filters=num_filters_global, number_filters_S=num_filters_S, number_filters_T=num_filters_T, filter_init_mode=filter_init_mode)        
        for ii in range(num_layers):
            if not share_filters:
                shared_filters = SharedSpectralFilters(dim=hidden_dim, number_of_filters=num_filters_global, number_filters_S=num_filters_S, number_filters_T=num_filters_T, filter_init_mode=filter_init_mode)
    
            shared_ff = FeedForwardModule(dim=hidden_dim,
                                hidden_dim_multiplier=hidden_dim_multiplier,
                                dropout=dropout)
            
            residual_module = ResidualModuleWrapper(
                module=module,
                dim=hidden_dim,
                step_size=initial_step_rmw,
                hidden_dim_multiplier=hidden_dim_multiplier,
                num_heads=num_heads,
                dropout=dropout,
                inner_dropout=inner_dropout,
                shared_filters=shared_filters, 
                shared_feedforward=shared_ff, 
                initial_step_global=initial_step_global,
                initial_step_sides=initial_step_sides,
            )
            self.residual_modules.append(residual_module)

        self.input_linear = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.act = nn.GELU()

        self.output_normalization = nn.LayerNorm(hidden_dim)
        self.output_linear = nn.Linear(in_features=hidden_dim, out_features=output_dim)

    def forward(self, graph, x):
        x = self.input_linear(x)
        x = self.dropout(x)
        x = self.act(x)

        for residual_module in self.residual_modules:
            x = residual_module(graph, x)

        x = self.output_normalization(x)
        x = self.output_linear(x).squeeze(1)

        return x    

U_eig = None
eigvals = None
lam_normalized = None
lam_max = None
basis_cauchy = None  # U_full^T @ blockdiag(U_S, U_T)
U_S_local = None
U_T_local = None
eig_S_local = None
eig_T_local = None
lam_S_normalized = None
lam_T_normalized = None
idx_S_global = None
idx_T_global = None

import math

import torch
import torch.nn.functional as F
import math

def load_spectral_components(path, device):
    """Load the spectral tensors exported by export_factorization.py."""
    global U_eig, eigvals, lam_max, lam_normalized, basis_cauchy
    global U_S_local, U_T_local, eig_S_local, eig_T_local, lam_S_normalized, lam_T_normalized
    global idx_S_global, idx_T_global
    import numpy as np
    d = np.load(path, allow_pickle=True)
    t = lambda k: torch.from_numpy(np.ascontiguousarray(d[k])).float().to(device)
    eigvals = t("eigvals")
    lam_max = eigvals.max()
    lam_normalized = eigvals / lam_max
    basis_cauchy = t("basis_cauchy")
    U_S_local = t("U_S")
    U_T_local = t("U_T")
    eig_S_local = t("eig_S")
    eig_T_local = t("eig_T")
    lam_S_normalized = eig_S_local / (eig_S_local.max() + 1e-9)
    lam_T_normalized = eig_T_local / (eig_T_local.max() + 1e-9)
    idx_S_global = torch.from_numpy(np.ascontiguousarray(d["idx_S"])).long().to(device)
    idx_T_global = torch.from_numpy(np.ascontiguousarray(d["idx_T"])).long().to(device)
    U_eig = basis_cauchy  # sentinel: spectral components are initialized
    print(f"Loaded Cauchy factorization: {path} "
          f"(n={basis_cauchy.shape[0]}, |S|={len(idx_S_global)}, "
          f"|T|={len(idx_T_global)}, "
          f"laplacian={d['laplacian']}, target_cut={int(d['target_cut'])})")

class SpectralWaveletFilter(nn.Module):
    def __init__(self, in_features, num_scales, init_mode='low_pass'):
        super().__init__()
        self.in_features = in_features 
        self.num_scales = num_scales
        self.init_mode = init_mode
        self.coeffs = nn.Parameter(torch.Tensor(num_scales, 1))
        self.inner_dim = 1
        self.combinator = nn.Parameter(torch.ones(self.inner_dim, in_features))
        
        self.reset_parameters()

    def _get_target_coeffs(self, K, order=2):
        x_centers = torch.linspace(0, 1, K)
        
        if self.init_mode == 'low_pass':
            # Decays from 1 to 0
            y_target = (1.0 - x_centers) ** order
        elif self.init_mode == 'high_pass':
            # Grows from 0 to 1
            y_target = x_centers ** order
        elif self.init_mode == 'band_pass':
            # Gaussian bump in middle
            y_target = torch.exp(-((x_centers - 0.5)**2) / 0.1)
        elif self.init_mode == 'flat':
            y_target = torch.ones(K) * 0.25
        else:
            y_target = torch.rand(K)
            
        return y_target.float()
    
    def reset_parameters(self):
        # 1. Get Ideal Shape
        ideal_coeffs = self._get_target_coeffs(K=self.num_scales, order=2)
        
        # 2. Add noise
        #nn.init.xavier_uniform_(self.coeffs)
        #self.coeffs = nn.Parameter(ideal_coeffs.unsqueeze(1))
        #nn.init.xavier_uniform_(self.combinator)
        #self.combinator.data = torch.ones_like(self.combinator.data)
        self.combinator.data = torch.ones_like(self.combinator.data)
         #self.coeffs.data *= 0.1
        # 3. Add Bias
        with torch.no_grad():
            self.coeffs.data = torch.zeros_like(self.coeffs.data)
            self.coeffs.data += ideal_coeffs.unsqueeze(1)   
            
    def forward(self, x_hat, basis_matrix):
        # i want to generate N_DIM filters using combinator and coeffs
        basis_tmp = torch.matmul(basis_matrix, self.coeffs)  # [N, inner_dim]
        return x_hat * basis_tmp  # [N, in_features]

class SharedSpectralFilters(nn.Module):
    def __init__(self, dim, number_of_filters=6, number_filters_S=6, number_filters_T=6, filter_init_mode='band_pass'):
        super().__init__()
        self.number_of_filters = number_of_filters
        self.number_filters_S = number_filters_S
        self.number_filters_T = number_filters_T

        init_mode_options = [filter_init_mode]
        init_mode_options_sides = ['band_pass']

        self.spectral_wavelet_filter = SpectralWaveletFilter(in_features=dim, num_scales=number_of_filters, init_mode=init_mode_options[0])
        self.spectral_wavelet_filter_S = SpectralWaveletFilter(in_features=dim, num_scales=number_filters_S, init_mode=init_mode_options_sides[0])
        self.spectral_wavelet_filter_T = SpectralWaveletFilter(in_features=dim, num_scales=number_filters_T, init_mode=init_mode_options_sides[0])


def apply_bank(x_hat_local, basis_local, filters):
    basis_tmp = basis_local @ filters.coeffs @ filters.combinator # [N_local, dim]
    x_out = x_hat_local * basis_tmp 
    return x_out


class SGWTModule(nn.Module):
    def __init__(self, dim, hidden_dim_multiplier, dropout, inner_dropout, shared_filters, shared_feedforward, 
                 initial_step_global, initial_step_sides, **kwargs):
        super().__init__()
        self.feed_forward_module = shared_feedforward
                
        self.dim = dim
        self.spline_degree = 3
        self.B_global = None
        self.B_part = {'S': None, 'T': None}

        self.number_of_filters = shared_filters.number_of_filters
        self.number_filters_S = shared_filters.number_filters_S
        self.number_filters_T = shared_filters.number_filters_T
        self.spectral_wavelet_filter = shared_filters.spectral_wavelet_filter
        self.spectral_wavelet_filter_S = shared_filters.spectral_wavelet_filter_S
        self.spectral_wavelet_filter_T = shared_filters.spectral_wavelet_filter_T

        # Per-layer mixing weights (kept unique per layer)      
        # map back from sigmoid to logit space for initialization
        if initial_step_sides == 1.0:
            self.flag_direct = True
        else:
            self.flag_direct = False
            logit_sides = math.log(initial_step_sides / (1 - initial_step_sides))
            self._step_S_logit = nn.Parameter(torch.tensor(logit_sides))  # sigmoid(-2) ≈ 0.12
            self._step_T_logit = nn.Parameter(torch.tensor(logit_sides))
        if initial_step_global == 1.0:
            self.flag_direct_global = True
        else:
            self.flag_direct_global = False
            logit_global = math.log(initial_step_global / (1 - initial_step_global))              
            self._step_global_logit = nn.Parameter(torch.tensor(logit_global))  # sigmoid(-1) = 0.12            
        self.drop1 = nn.Dropout(p=inner_dropout)
        self.act = nn.SiLU()
                
    @property
    def step_S(self):
        return torch.sigmoid(self._step_S_logit)

    @property
    def step_T(self):
        return torch.sigmoid(self._step_T_logit)
    
    @property
    def step_global(self):
        return torch.sigmoid(self._step_global_logit)
    
    def bootstrap_filters(self):
        """Initial basis computation (used only on first call for plotting/debug).
        During training, forward() recomputes bases with current knots."""
        if lam_normalized is not None:
            self.B_global = self._compute_spline_basis_wrapper(
                lam_normalized, K=self.number_of_filters)
        if lam_S_normalized is not None:
            self.B_part['S'] = self._compute_spline_basis_wrapper(
                lam_S_normalized, K=self.number_filters_S)
        if lam_T_normalized is not None:
            self.B_part['T'] = self._compute_spline_basis_wrapper(
                lam_T_normalized, K=self.number_filters_T)
        
    def _compute_spline_basis_wrapper(self, lam, K):
        """
        Wrapper to compute Spline basis on the eigenvalues.
        Uses percentile-based knots to match eigenvalue distribution.
        """
        lam = lam.flatten()
        lam = torch.clamp(lam, 0.0, 1.0)

        # Percentile-based knots
        qs = torch.linspace(0, 1, K - self.spline_degree + 1, device=lam.device)
        #internal_knots = torch.quantile(lam, qs)
        internal_knots = torch.linspace(0, 1, K - self.spline_degree + 1, device=lam.device)
        
        # ensure endpoints
        internal_knots[0] = 0.0
        internal_knots[-1] = 1.0
        knots = torch.cat([
            torch.zeros(self.spline_degree, device=lam.device),
            internal_knots,
            torch.ones(self.spline_degree, device=lam.device)
        ])

        basis = compute_spline_basis(lam, K=K, degree=self.spline_degree, knots=knots)
            
        # Optional: Plotting for verification
        if not hasattr(self, '_plotted_spline'):
            self._plotted_spline = True
            try:
                import matplotlib.pyplot as plt
                import os
                if not os.path.exists('fig_dump'): os.makedirs('fig_dump')
                
                plt.figure(figsize=(8, 5))
                sorted_idx = torch.argsort(lam)
                lam_sorted = lam[sorted_idx].cpu().numpy()
                basis_sorted = basis[sorted_idx].cpu().numpy()
                
                # Plot first few basis functions
                for k in range(K): 
                    plt.plot(lam_sorted, basis_sorted[:, k], label=f'Spline {k}')
                    
                plt.title(f'B-Spline Basis (d={self.spline_degree}) on Eigenvalues')
                plt.xlabel('Normalized Lambda')
                plt.legend()
                plt.savefig('fig_dump/spline_basis.png', dpi=300)
                plt.close()
                print("Saved spline basis plot to fig_dump/spline_basis.png")
            except Exception as e:
                print(f"Skipped plotting: {e}")
        return basis
    
    def forward(self, graph, x):
        if U_eig is None:
            raise RuntimeError("spectral components not loaded - "
                               "call load_spectral_components() first")
        if self.B_global is None:
            self.bootstrap_filters()

        # Recompute bases with current learnable knots each forward pass
        B_global = self._compute_spline_basis_wrapper(
            lam_normalized, K=self.number_of_filters)
        B_S = self._compute_spline_basis_wrapper(
            lam_S_normalized, K=self.number_filters_S)
        B_T = self._compute_spline_basis_wrapper(
            lam_T_normalized, K=self.number_filters_T)

        # 1) Partition the signal
        x_S = x[idx_S_global]
        x_T = x[idx_T_global]

        # 2) Transform to local spectral domains
        x_S_hat = torch.matmul(U_S_local.T, x_S)
        x_T_hat = torch.matmul(U_T_local.T, x_T)

        x_S_new = apply_bank(
            x_S_hat,
            B_S,
            self.spectral_wavelet_filter_S, # Best results if shared; remember to flip this
        )
        
        x_T_new = apply_bank(
            x_T_hat,
            B_T,
            self.spectral_wavelet_filter_T, 
        )

        if self.flag_direct is False:
            step_S = self.step_S
            step_T = self.step_T

            scaled_S = x_S_new * step_S + x_S_hat
            scaled_T = x_T_new * step_T + x_T_hat
        else:
            scaled_S = x_S_new
            scaled_T = x_T_new
        
        block_hat = torch.cat([scaled_S, scaled_T], dim=0)
        
        x_global_hat = torch.matmul(basis_cauchy, block_hat)
        
        x_new = apply_bank(
            x_global_hat,
            B_global,
            self.spectral_wavelet_filter, 
        )
        
        # Inverse transform
        x_filt_1 = torch.matmul(basis_cauchy.T, x_new)
        x_filtered = torch.zeros_like(x)
        x_filtered[idx_S_global] = torch.matmul(U_S_local, x_filt_1[:len(idx_S_global)])
        x_filtered[idx_T_global] = torch.matmul(U_T_local, x_filt_1[len(idx_S_global):])
        x_filtered = self.drop1(x_filtered)
        if self.flag_direct_global is False:
            x_filtered = self.act(x_filtered) * self.step_global + x
        else:
            x_filtered = self.act(x_filtered)
        x = self.feed_forward_module(graph, x_filtered)
        return x

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--factorization', type=str, required=True,
                        help='npz produced by export_factorization.py')
    parser.add_argument('--name', type=str, default=None, help='Experiment name. If None, model name is used.')
    parser.add_argument('--save_dir', type=str, default='experiments', help='Base directory for saving information.')
    parser.add_argument('--dataset', type=str, default='minesweeper',
                        choices=['roman-empire', 'amazon-ratings', 'minesweeper', 'tolokers', 'questions',
                                 'squirrel', 'squirrel-directed', 'squirrel-filtered', 'squirrel-filtered-directed',
                                 'chameleon', 'chameleon-directed', 'chameleon-filtered', 'chameleon-filtered-directed',
                                 'actor', 'texas', 'texas-4-classes', 'cornell', 'wisconsin'])
    # model architecture
    parser.add_argument('--model', type=str, default='SGWT',
                        choices=['GCN', 'SGWT'])
    parser.add_argument('--num_layers', type=int, default=16)
    parser.add_argument('--hidden_dim', type=int, default=32)
    parser.add_argument('--hidden_dim_multiplier', type=float, default=1)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--normalization', type=str, default='LayerNorm', choices=['None', 'LayerNorm', 'BatchNorm'])
    # regularization
    parser.add_argument('--dropout', type=float, default=0.25)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    
    # training parameters
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--num_steps', type=int, default=1500)
    parser.add_argument('--num_warmup_steps', type=int, default=None,
                        help='If None, warmup_proportion is used instead.')
    parser.add_argument('--warmup_proportion', type=float, default=0, help='Only used if num_warmup_steps is None.')
    
    # node feature augmentation
    parser.add_argument('--use_sgc_features', default=False, action='store_true')
    parser.add_argument('--use_identity_features', default=False, action='store_true')
    parser.add_argument('--use_adjacency_features', default=False, action='store_true')
    parser.add_argument('--do_not_use_original_features', default=False, action='store_true')
    parser.add_argument('--num_runs', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--amp', default=False, action='store_true')
    parser.add_argument('--verbose', default=False, action='store_true')
    parser.add_argument('--lr_mult_spectral', type=float, default=64.0)
    parser.add_argument('--lr_mult_step', type=float, default=0.0)
    parser.add_argument('--inner_dropout', type=float, default=0.0)
    parser.add_argument('--share_filters', default=True, action='store_true')
    parser.add_argument("--initial_step_global", type=float, default=0.10, help="Initial global filter step size (between 0 and 1).")
    parser.add_argument("--initial_step_sides", type=float, default=0.9, help="Initial filter step size (between 0 and 1).")
    parser.add_argument("--initial_step_rmw", type=float, default=1.0, help="Initial step size for residual mixing weights (between 0 and 1).")
    parser.add_argument("--filters_global", type=int, default=6, help="Number of global filters.") # best uses 4
    parser.add_argument("--filters_S", type=int, default=6, help="Number of filters on S side.") # best uses 4
    parser.add_argument("--filters_T", type=int, default=6, help="Number of filters on T side.") # best uses 4
    parser.add_argument("--filter_init_mode", type=str, default='high_pass', choices=['low_pass', 'high_pass', 'band_pass', 'flat', 'random'], help="Initialization mode for spectral wavelet filters.")
    
    args = parser.parse_args()

    if args.name is None:
        args.name = args.model

    return args

def train_step(model, dataset, optimizer, scheduler, scaler, args, amp=False):
    model.train()
    optimizer.zero_grad()

    with autocast(enabled=amp):
        logits = model(graph=dataset.graph, x=dataset.node_features)
        # 1. Main task loss
        loss = dataset.loss_fn(input=logits[dataset.train_idx], target=dataset.labels[dataset.train_idx])

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()
    
@torch.no_grad()
def evaluate(model, dataset, amp=False):
    model.eval()

    with autocast(enabled=amp):
        logits = model(graph=dataset.graph, x=dataset.node_features)

    metrics = dataset.compute_metrics(logits)

    return metrics

model_classes = {
    'SGWT': SGWTModule,
}

def main():
    args = get_args()

    torch.manual_seed(0)

    dataset = Dataset(name=args.dataset,
                      add_self_loops=(args.model in ['GCN', 'GAT', 'GT']),
                      device=args.device,
                      use_sgc_features=args.use_sgc_features,
                      use_identity_features=args.use_identity_features,
                      use_adjacency_features=args.use_adjacency_features,
                      do_not_use_original_features=args.do_not_use_original_features)

    load_spectral_components(args.factorization, device=args.device)

    logger = Logger(args, metric=dataset.metric, num_data_splits=dataset.num_data_splits)

    for run in range(1, args.num_runs + 1):
        model = Model(
            num_layers=args.num_layers,
            input_dim=dataset.num_node_features,
            hidden_dim=args.hidden_dim,
            output_dim=dataset.num_targets,
            hidden_dim_multiplier=args.hidden_dim_multiplier,
            num_heads=args.num_heads,
            module=model_classes[args.model],
            dropout=args.dropout,
            inner_dropout=args.inner_dropout,
            share_filters=args.share_filters, 
            initial_step_global=args.initial_step_global,
            initial_step_sides=args.initial_step_sides,
            initial_step_rmw=args.initial_step_rmw,
            num_filters_global=args.filters_global,
            num_filters_S=args.filters_S,
            num_filters_T=args.filters_T, 
            filter_init_mode=args.filter_init_mode
        )

        model.to(args.device)

        print("Weight decay:", args.weight_decay)
        parameter_groups = get_parameter_groups(model, base_lr=args.lr, filter_lr_multiplier=args.lr_mult_spectral, step_lr_multiplier=args.lr_mult_step)
        optimizer = torch.optim.AdamW(parameter_groups, lr=args.lr, weight_decay=args.weight_decay)
        scaler = GradScaler(enabled=args.amp)
        scheduler = get_lr_scheduler_with_warmup(optimizer=optimizer, num_warmup_steps=args.num_warmup_steps,
                                         num_steps=args.num_steps, warmup_proportion=args.warmup_proportion)
        logger.start_run(run=run, data_split=dataset.cur_data_split + 1)
        with tqdm(total=args.num_steps, desc=f'Run {run}', disable=args.verbose) as progress_bar:
            best_val = float("-inf")
            best_step = -1
            best_test_at_val = float("nan")            
            for step in range(1, args.num_steps + 1):
                train_step(model=model, dataset=dataset, optimizer=optimizer, scheduler=scheduler,
                           scaler=scaler, args=args, amp=args.amp)
                metrics = evaluate(model=model, dataset=dataset, amp=args.amp)
                logger.update_metrics(metrics=metrics, step=step)
                progress_bar.update()
                
                val_key = "val ROC AUC" if "val ROC AUC" in metrics else "val accuracy"
                test_key = "test ROC AUC" if "test ROC AUC" in metrics else "test accuracy"
                if val_key in metrics and test_key in metrics:
                    if metrics[val_key] > best_val:
                        best_val = metrics[val_key]
                        best_step = step
                        best_test_at_val = metrics[test_key]

                progress_bar.set_postfix({
                    **{metric: f'{value:.2f}' for metric, value in metrics.items()},
                    "bval": f"{best_val:.4f}",
                    "btest": f"{best_test_at_val:.4f}",
                })
                
                #progress_bar.set_postfix({metric: f'{value:.4f}' for metric, value in metrics.items()})
                # --- Plotting Hook (optional; module not shipped) ---
                try:
                    from plot_spectral_filters import plot_filters_with_eigenvalues
                except ImportError:
                    plot_filters_with_eigenvalues = None
                if plot_filters_with_eigenvalues is not None and (
                        step % 200 == 0 or step == 1 or step == args.num_steps):
                    import numpy as np
                    lam_plot = lam_normalized.cpu().numpy() if lam_normalized is not None else None
                    lam_plot = lam_plot[:-2]
                    lam_plot = lam_plot / (lam_plot.max() * (1.05))
                    lam_plot = np.concatenate([lam_plot, [1.0]])  # ensure we see the endpoint
                    lam_plot = np.clip(lam_plot, 0.0, 1.0) if lam_plot is not None else None
                    lam_plot = torch.from_numpy(lam_plot).float() if lam_plot is not None else None
                    #plot_filters_with_eigenvalues(model, step=step, run_id=run,
                    #    lam_original=lam_original,
                    #    lam_display=lam_display,
                    #    lam_S_normalized=lam_S_normalized,
                    #    lam_T_normalized=lam_T_normalized)
                    
        logger.finish_run()
        model.cpu()
        # === Save model checkpoint ===
        ckpt_dir = "checkpoints_platonov"
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, f"{args.dataset}_model_run{run}.pt")
        torch.save({
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "spectral_cache_path": args.factorization,
            "input_dim": dataset.num_node_features,
            "output_dim": dataset.num_targets,
        }, ckpt_path)
        print(f"Saved model checkpoint to {ckpt_path}")        
        dataset.next_data_split()

    logger.print_metrics_summary()

if __name__ == '__main__':
    main()
    
