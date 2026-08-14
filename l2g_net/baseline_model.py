import torch
import torch.nn as nn


class ResidualModuleWrapper(nn.Module):
    def __init__(self, module, dim, step_size=0.5, **kwargs):
        super().__init__()
        self.normalization =  nn.LayerNorm(dim)
        self.module = module(dim=dim, **kwargs)
        self.step_pass = nn.Parameter(torch.tensor(step_size))  # starts at 1.0

    def forward(self, graph, x):
        x_res = self.normalization(x)
        x_res = self.module(graph, x_res)
        x = x + x_res * self.step_pass
        return x

class FeedForwardModule(nn.Module):
    def __init__(self, dim, hidden_dim_multiplier, dropout, input_dim_multiplier=1, **kwargs):
        super().__init__()
        input_dim = int(dim * input_dim_multiplier)
        hidden_dim = int(dim * hidden_dim_multiplier)
        self.linear_1 = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.dropout_1 = nn.Dropout(p=dropout)
        self.act = nn.GELU()
        self.linear_2 = nn.Linear(in_features=hidden_dim, out_features=dim)
        self.dropout_2 = nn.Dropout(p=dropout)

    def forward(self, graph, x):
        x = self.linear_1(x)
        x = self.dropout_1(x)
        x = self.act(x)
        x = self.linear_2(x)
        x = self.dropout_2(x)
        return x
    
    
def compute_spline_basis(x, K, degree=2, knots=None):
    """
    Computes B-Spline basis functions N_{i,d}(x) using Cox-de Boor recursion.
    
    Args:
        x: [N] tensor of eigenvalues, normalized to [0, 1]
        K: Number of basis functions (output size)
        degree: Degree of the B-splines (2=Quadratic, 3=Cubic)
        
    Returns:
        Basis matrix of shape [N, K]
    """
    device = x.device
    N = x.shape[0]
    
    # We want K basis functions.
    # For open uniform B-splines, we need (K + degree + 1) knots.
    # Knots: [0...0, t_1, ..., t_m, 1...1]
    
    # Number of internal intervals = K - degree
    # If K=5, degree=2, we need 3 internal intervals.
    # Internal knots needed: K - degree + 1 points (including 0 and 1)
    
    if K <= degree:
        raise ValueError(f"Number of filters K ({K}) must be > degree ({degree})")

    # Generate knot vector
    # Internal spacing: linearly spaced between 0 and 1
    # num_points = (K - degree + 1) -> creates (K - degree) intervals
    if knots is None:
        # Uniform knots (original behavior)
        internal_knots = torch.linspace(0, 1, K - degree + 1, device=device)
        knots = torch.cat([
            torch.zeros(degree, device=device),
            internal_knots,
            torch.ones(degree, device=device)
        ])
    else:
        knots = knots.to(device)
    
    x = x.unsqueeze(1) # [N, 1]
    
    # --- Cox-de Boor Recursion ---
    
    # Degree 0 (Step functions)
    # B_{i,0}(x) = 1 if knots[i] <= x < knots[i+1], else 0
    # Note: We must handle x=1 carefully (usually belongs to last interval)
    
    kv = knots
    d0_basis = []
    
    # We compute for all relevant intervals. 
    # Total knots = M. Degree 0 basis count = M - 1.
    num_d0 = knots.shape[0] - 1
    
    lower = kv[:-1].unsqueeze(0) # [1, M-1]
    upper = kv[1:].unsqueeze(0)  # [1, M-1]
    
    # Standard interval check
    basis = (x >= lower) & (x < upper)
    
    # Fix the right boundary (x=1.0)
    # Include x=1 in the very last interval where (upper == 1)
    # We find the last interval that ends at 1.0 and perform an OR check
    mask_one = (x >= 1.0 - 1e-7)
    # The last valid interval is usually the one before the padded 1s
    # or just the last one in the list.
    # For open splines, the active domain ends at index K.
    if mask_one.any():
        basis[mask_one.squeeze(), -1] = 1.0 
        # (A slight simplification, but numerically stable for normalized graphs)

    basis = basis.float()
    
    # Recursion for degrees 1..d
    for d in range(1, degree + 1):
        # B_{i,d} has M - 1 - d basis functions
        b_prev = basis
        current_size = b_prev.shape[1] - 1
        
        # Term 1: (x - u_i) / (u_{i+d} - u_i) * B_{i, d-1}
        # Numerator
        term1_num = (x - kv[:current_size].unsqueeze(0))
        # Denominator
        term1_den = (kv[d:d+current_size] - kv[:current_size]).unsqueeze(0)
        
        # Term 2: (u_{i+d+1} - x) / (u_{i+d+1} - u_{i+1}) * B_{i+1, d-1}
        # Numerator
        term2_num = (kv[d+1:d+1+current_size].unsqueeze(0) - x)
        # Denominator
        term2_den = (kv[d+1:d+1+current_size] - kv[1:1+current_size]).unsqueeze(0)
        
        # Safe Division (0/0 = 0)
        term1_den[term1_den == 0] = 1.0
        term2_den[term2_den == 0] = 1.0
        
        term1 = (term1_num / term1_den) * b_prev[:, :current_size]
        term2 = (term2_num / term2_den) * b_prev[:, 1:]
        
        basis = term1 + term2
    return basis
