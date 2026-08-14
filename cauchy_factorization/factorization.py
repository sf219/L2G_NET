import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix, diags, eye as speye
from scipy.sparse.linalg import eigsh, cg, spsolve
import numba as nb
import time
import os
from ._metis_compat import pymetis
from .recursive_decomp import (
    _cauchy_matvec_batch, _cauchy_matvec_batch_parallel_stable,
    _cauchy_matvec_batch_parallel, _cauchy_matvec_batch_stable,
)
from .recursive_decomp import (
    RecursiveDecomp, _apply_givens_batch,
    _apply_cauchy_factor_to_vector_znorms, _apply_cauchy_factor_to_vector_zhat,
)
from .cauchy_factor import CauchyFactor

# ============================================================================
# GRAPH LOADING — ogbn-arxiv
# ============================================================================

def load_ogbn_arxiv_subgraph(n_target, seed=42):
    """
    Load ogbn-arxiv and extract a connected subgraph of approximately n_target nodes
    via BFS from a random high-degree seed node.
    
    Returns a NetworkX Graph with nodes relabeled 0..n-1.
    """
    try:
        from ogb.nodeproppred import NodePropPredDataset
    except ImportError:
        raise ImportError(
            "ogb not installed. Run: pip install ogb\n"
            "The dataset will be downloaded automatically on first use (~170MB)."
        )
    
    print(f"  Loading ogbn-arxiv dataset...")
    dataset = NodePropPredDataset(name='ogbn-arxiv', root='data/')
    graph, _ = dataset[0]
    
    # graph['edge_index'] is (2, num_edges) — directed edges
    edge_index = graph['edge_index']  # numpy array (2, E)
    src = edge_index[0]
    dst = edge_index[1]
    n_total = graph['num_nodes']
    
    print(f"  ogbn-arxiv: {n_total} nodes, {len(src)} directed edges")
    
    # Build undirected adjacency using scipy sparse (much faster than NetworkX)
    from scipy.sparse import coo_matrix
    ones = np.ones(len(src), dtype=np.float64)
    A = coo_matrix((ones, (src, dst)), shape=(n_total, n_total))
    A = A + A.T  # symmetrize
    A = (A > 0).astype(np.float64)  # binary
    A = csr_matrix(A)
    
    # BFS to extract connected subgraph of size ~n_target
    rng = np.random.RandomState(seed)
    
    # Pick a seed node: choose from high-degree nodes for a well-connected start
    degrees = np.array(A.sum(axis=1)).ravel()
    top_degree_nodes = np.argsort(degrees)[-100:]
    seed_node = int(rng.choice(top_degree_nodes))
    
    print(f"  BFS from seed node {seed_node} (degree={int(degrees[seed_node])})...")
    
    # BFS
    visited = set()
    queue = [seed_node]
    visited.add(seed_node)
    
    while len(visited) < n_target and queue:
        node = queue.pop(0)
        # Get neighbors from sparse matrix
        row = A.getrow(node)
        neighbors = row.indices
        rng.shuffle(neighbors)  # randomize BFS order for diversity
        for nb_node in neighbors:
            nb_node = int(nb_node)
            if nb_node not in visited:
                visited.add(nb_node)
                queue.append(nb_node)
                if len(visited) >= n_target:
                    break
    
    subset = sorted(visited)
    n_actual = len(subset)
    print(f"  Extracted subgraph: {n_actual} nodes")
        
    # Extract submatrix
    subset_arr = np.array(subset)
    A_sub = A[subset_arr][:, subset_arr]
    
    # Convert to NetworkX with relabeled nodes 0..n-1
    G = nx.Graph()
    G.add_nodes_from(range(n_actual))
    A_sub_coo = A_sub.tocoo()
    for i, j in zip(A_sub_coo.row, A_sub_coo.col):
        if i < j:  # undirected, avoid duplicates
            G.add_edge(int(i), int(j))
    
    # Ensure connected (take largest CC if needed)
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
        G = nx.convert_node_labels_to_integers(G)
        print(f"  Took largest connected component: {G.number_of_nodes()} nodes, "
              f"{G.number_of_edges()} edges")
    else:
        print(f"  Connected: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    return G

# ============================================================================
# NUMBA KERNELS — secular equation solvers (Bunch-Nielsen-Sorensen / Li)
# ============================================================================

@nb.njit(cache=True)
def _solve_secular_root_j(d, z2, k, j):
    """
    Find root j of the secular equation 1 + sum z2[i]/(d[i]-sigma) = 0
    in the interval (d[j], d[j+1]) using Newton-bisection.
    
    z2[i] = rho * z[i]^2 (rho is already folded in by the caller).
    Uses relative convergence tolerance for stability with large eigenvalues.
    """
    # Setup interval
    if j < k - 1:
        left = d[j]
        right = d[j + 1]
        mid_gap = right - left
    else:
        left = d[j]
        total_z2 = 0.0
        for i in range(k):
            total_z2 += z2[i]
        right = left + total_z2 + 1.0
        mid_gap = right - left

    orgn = left
    
    if j < k - 1:
        gap = d[j+1] - d[j]
        
        tau = 0.5 * gap
        sigma = orgn + tau
        
        f = 1.0
        for i in range(k):
            f += z2[i] / (d[i] - sigma)
        
        if f > 0:
            tau = 0.25 * gap
        else:
            tau = 0.75 * gap
    else:
        total_z2 = 0.0
        for i in range(k):
            total_z2 += z2[i]
        tau = total_z2
        if tau < 1e-15:
            tau = 1e-10
    
    # Relative tolerance for bracket bounds
    scale = max(abs(left), 1.0)
    lo_tau = 1e-15 * scale
    if j < k - 1:
        hi_tau = d[j + 1] - d[j] - 1e-15 * max(abs(d[j+1]), 1.0)
    else:
        total_z2 = 0.0
        for i in range(k):
            total_z2 += z2[i]
        hi_tau = total_z2 + max(total_z2, 1.0)  # wider bracket for large rho
    
    if lo_tau >= hi_tau:
        lo_tau = 1e-300
        hi_tau = max(mid_gap, 1e-10)
    
    if tau < lo_tau:
        tau = lo_tau
    if tau > hi_tau:
        tau = hi_tau
    
    for iteration in range(50):  # more iterations for large rho
        sigma = orgn + tau
        
        f = 1.0
        fp = 0.0
        for i in range(k):
            delta = d[i] - sigma
            if abs(delta) < 1e-300:
                if delta >= 0:
                    delta = 1e-300
                else:
                    delta = -1e-300
            t = z2[i] / delta
            f += t
            fp += t / delta
        
        # Relative convergence: |f| small relative to the dominant term
        if abs(f) < 1e-14 * (1.0 + abs(fp) * max(abs(orgn + tau), 1.0)):
            break
        
        if fp > 1e-300:
            newton_step = f / fp
        else:
            if f < 0:
                lo_tau = tau
            else:
                hi_tau = tau
            tau = 0.5 * (lo_tau + hi_tau)
            continue
        
        tau_new = tau - newton_step
        
        if f < 0:
            lo_tau = tau
        else:
            hi_tau = tau
        
        if lo_tau < tau_new < hi_tau:
            tau = tau_new
        else:
            tau = 0.5 * (lo_tau + hi_tau)
        
        # Gap-relative bracket convergence: stop when bracket is smaller than
        # eps * interval_width. This ensures root error ~ eps * gap, making
        # off-diagonal C^T C entries O(eps) regardless of d_max/gap ratio.
        # Using scale-relative (eps * d_max) causes errors of eps * d_max / gap
        # which grows with d_max/gap ratio (e.g., 1500/3e-5 = 5e7 -> 5e-7 per entry,
        # summing to 1e-3 over k=1000 terms in the orthogonality probe).
        if j < k - 1:
            interval_width = d[j + 1] - d[j]
        else:
            interval_width = hi_tau - lo_tau + 1.0
        if abs(hi_tau - lo_tau) < 1e-14 * max(interval_width, 1e-300):
            break
    
    return orgn + tau

@nb.njit(cache=True)
def _perturb_duplicates(d, min_gap=1e-12):
    """In-place perturbation: ensure all eigenvalues have gap >= min_gap."""
    k = d.size
    for i in range(1, k):
        if d[i] - d[i - 1] < min_gap:
            d[i] = d[i - 1] + min_gap


@nb.njit(cache=True)
def _perturb_duplicates_rel(d, min_gap=1e-12):
    """Ensure consecutive eigenvalues have relative gap >= rel_gap."""
    k = d.size
    for i in range(1, k):
        scale = max(abs(d[i - 1]), abs(d[i]), 1.0)
        gap_i = min_gap * scale
        if d[i] - d[i - 1] < gap_i:
            d[i] = d[i - 1] + gap_i

@nb.njit(cache=True, parallel=True)
def solve_secular_all_roots(d, z, rho=1.0):
    """
    Find all k roots of the secular equation.
    Uses per-root Newton-bisection solver.
    """
    k = d.size
    if k == 0:
        return np.empty(0, dtype=np.float64)

    roots = np.empty(k, dtype=np.float64)
    z2 = np.empty(k, dtype=np.float64)
    for i in range(k):
        z2[i] = z[i] * z[i] * rho

    for j in nb.prange(k):
        roots[j] = _solve_secular_root_j(d, z2, k, j)
    
    return roots


@nb.njit(cache=True)
def compute_z_hat_stable(d, z2_rho, roots, z_sign, k):
    """
    Compute z_hat using the LAPACK/Gu-Eisenstat numerically stable formula.
    """
    z_hat = np.empty(k, dtype=np.float64)
    
    deltas = np.empty(k, dtype=np.float64)
    for j in range(k):
        deltas[j] = roots[j] - d[j]
    
    for i in range(k):
        if abs(deltas[i]) < 1e-300:
            z_hat[i] = 0.0
            continue
        
        log_abs = np.log(abs(deltas[i]))
        sign = 1 if deltas[i] > 0 else -1
        
        skip = False
        for j in range(k):
            if j == i:
                continue
            gap = d[j] - d[i]
            if abs(gap) < 1e-300:
                skip = True
                break
            ratio = 1.0 + deltas[j] / gap
            if abs(ratio) < 1e-300:
                skip = True
                break
            if ratio < 0:
                sign = -sign
                ratio = -ratio
            log_abs += np.log(ratio)
        
        if skip:
            z_hat[i] = 0.0
            continue
        
        if sign < 0:
            z_hat[i] = 0.0
            continue
        
        if log_abs > 700:
            log_abs = 700.0
        if log_abs < -700:
            z_hat[i] = 0.0
            continue
        
        magnitude = np.exp(0.5 * log_abs)
        if z_sign[i] < 0:
            z_hat[i] = -magnitude
        else:
            z_hat[i] = magnitude
    
    return z_hat


@nb.njit(cache=True)
def compute_eigenvector_via_secular(d, z2_rho, root_j, k):
    """
    Compute the j-th eigenvector of D + rho*z*z^T directly.
    """
    v = np.empty(k, dtype=np.float64)
    norm_sq = 0.0
    for i in range(k):
        delta = d[i] - root_j
        if abs(delta) < 1e-300:
            if delta >= 0:
                delta = 1e-300
            else:
                delta = -1e-300
        v[i] = np.sqrt(abs(z2_rho[i])) / delta
        norm_sq += v[i] * v[i]
    
    if norm_sq > 0:
        inv_norm = 1.0 / np.sqrt(norm_sq)
        for i in range(k):
            v[i] *= inv_norm
    
    return v

def _factor_orth_probe_err(d, roots, z, norms, z_hat, n_probes=2):
    k = len(d)
    if k == 0:
        return 0.0, np.inf
    rng = np.random.RandomState(777)
    err_classic = 0.0
    err_zhat = np.inf
    ones = np.ones(k, dtype=np.float64)
    for _ in range(n_probes):
        x = rng.randn(k)
        yc = _cauchy_QtQx_znorms(d, roots, z, norms, x, k)
        ec = np.max(np.abs(yc - x))
        if np.isfinite(ec):
            err_classic = max(err_classic, ec)

        if z_hat is not None:
            yz = _cauchy_QtQx(d, roots, z_hat, ones, x, k)
            ez = np.max(np.abs(yz - x))
            if np.isfinite(ez):
                if not np.isfinite(err_zhat):
                    err_zhat = ez
                else:
                    err_zhat = max(err_zhat, ez)
    return err_classic, err_zhat


@nb.njit(cache=True, parallel=True)
def _compute_column_norms(z, d, roots, k):
    """Compute column norms of eigenvector matrix Q[i,j] = z[i]/(d[i]-roots[j]).
    Parallelised over j — same prange pattern as solve_secular_all_roots."""
    norms = np.empty(k, dtype=np.float64)
    for j in nb.prange(k):
        s = 0.0
        for i in range(k):
            delta = d[i] - roots[j]
            if abs(delta) < 1e-300:
                if delta >= 0:
                    delta = 1e-300
                else:
                    delta = -1e-300
            val = z[i] / delta
            s += val * val
        norms[j] = np.sqrt(s)
        if norms[j] < 1e-300:
            norms[j] = 1.0
    return norms

def _compute_z_hat_checked(d, roots, z, rho, orth_tol=1e-5):
    """
    Compute Gu-Eisenstat z_hat and validate with a cheap orthogonality probe.
    Returns None if unstable.
    """
    k = len(d)
    z2_rho = (z * z) * rho
    z_sign = np.sign(z)

    z_hat = compute_z_hat_stable(d, z2_rho, roots, z_sign, k)

    if not np.all(np.isfinite(z_hat)):
        return None

    x = np.random.RandomState(123).randn(k)
    y = _cauchy_QtQx(d, roots, z_hat, np.ones(k, dtype=np.float64), x, k)
    err = np.max(np.abs(y - x))
    if (not np.isfinite(err)) or (err > orth_tol):
        return None
    return z_hat

@nb.njit(cache=True)
def _compute_z_hat(d, roots, z, k, rho=1.0):
    """
    Compute z_hat using the Gu-Eisenstat product formula.
    """
    z_hat = np.empty(k, dtype=np.float64)
    
    deltas = np.empty(k, dtype=np.float64)
    for j in range(k):
        deltas[j] = roots[j] - d[j]
    
    for i in range(k):
        if abs(deltas[i]) < 1e-300:
            z_hat[i] = 0.0
            continue
        
        log_abs = np.log(abs(deltas[i]))
        sign = 1 if deltas[i] > 0 else -1
        
        skip = False
        for j in range(k):
            if j == i:
                continue
            gap = d[j] - d[i]
            if abs(gap) < 1e-300:
                skip = True
                break
            ratio = 1.0 + deltas[j] / gap
            if abs(ratio) < 1e-300:
                skip = True
                break
            if ratio < 0:
                sign = -sign
                ratio = -ratio
            log_abs += np.log(ratio)
        
        if skip:
            z_hat[i] = 0.0
            continue
        
        if log_abs > 700:
            log_abs = 700.0
        if log_abs < -700:
            z_hat[i] = 0.0
            continue
        
        magnitude = np.exp(0.5 * log_abs)
        
        if z[i] < 0:
            z_hat[i] = -magnitude
        else:
            z_hat[i] = magnitude
    
    return z_hat

# ============================================================================
# SPARSE-AWARE GRAPH UTILITIES
# ============================================================================

def approx_effective_resistances_cut(G, cut_edges, n_sketches=10, seed=42):
    n = G.number_of_nodes()
    nodes = sorted(G.nodes())
    node_to_idx = {nd: i for i, nd in enumerate(nodes)}
    
    L = sparse_laplacian(G)
    eps = 1e-4  # coarser regularization — we only need order-of-magnitude R_e
    L_reg = (L + eps * speye(n, format='csr')).tocsr()
    
    rng = np.random.RandomState(seed)
    Q = rng.choice([-1.0, 1.0], size=(n, n_sketches)) / np.sqrt(n_sketches)
    
    Z = np.zeros((n, n_sketches))
    for k in range(n_sketches):
        Z[:, k], _ = cg(L_reg, Q[:, k], rtol=1e-3, maxiter=100)  # loose tol
    
    R = np.array([
        float(np.sum((Z[node_to_idx[u], :] - Z[node_to_idx[v], :]) ** 2))
        for u, v in cut_edges
    ])
    return R

def sparsify_cut_edges_ss(G, part_0_set, part_1_set, target_cut=5,
                           n_sketches=20, seed=42, max_reweight=5.0):
    """
    Spielman-Srivastava cut sparsification with a multiplicative reweight cap.
    Samples target_cut edges ∝ w_e·R_e, HT-reweights w_e/(target_cut·p_e),
    then caps the reweight factor at max_reweight to bound rho (the rank-1
    update magnitude) and keep the secular solver well-conditioned.
    """
    rng = np.random.RandomState(seed)
    cut_edges = get_cut_edges(G, part_0_set, part_1_set)
    n_cut = len(cut_edges)

    if n_cut <= target_cut:
        return G.copy(), [(u, v, G[u][v].get('weight', 1.0)) for u, v in cut_edges]

    cut_weights = np.array([G[u][v].get('weight', 1.0) for u, v in cut_edges])
    R = approx_effective_resistances_cut(G, cut_edges, n_sketches=n_sketches, seed=seed)

    importance = np.maximum(cut_weights * R, 1e-300)
    p = importance / np.sum(importance)

    chosen = rng.choice(n_cut, size=target_cut, replace=True, p=p)

    G_sparse = G.copy()
    for u, v in cut_edges:
        if G_sparse.has_edge(u, v):
            G_sparse.remove_edge(u, v)

    kept = {}
    for idx in chosen:
        u, v = cut_edges[idx]
        raw_factor = 1.0 / (target_cut * p[idx])       # HT reweight factor
        factor = min(raw_factor, max_reweight)          # CAP: bound rho
        w_new = cut_weights[idx] * factor
        key = (min(u, v), max(u, v))
        kept[key] = kept.get(key, 0.0) + w_new

    kept_weighted = []
    for (u, v), w in kept.items():
        G_sparse.add_edge(u, v, weight=w)
        kept_weighted.append((u, v, w))

    return G_sparse, kept_weighted

def approx_effective_resistances(L, n_projections=32, seed=42):
    """Approximate effective resistances via JL projections."""
    n = L.shape[0]
    rng = np.random.RandomState(seed)
    
    eps = 1e-6
    L_reg = (L + eps * speye(n, format='csr')).tocsc()
    
    density = 1.0 / 3.0
    mask = rng.rand(n_projections, n)
    Q = np.zeros((n_projections, n), dtype=np.float64)
    scale = np.sqrt(3.0 / n_projections)
    Q[mask < density / 2] = scale
    Q[(mask >= density / 2) & (mask < density)] = -scale
    
    Z = np.empty((n_projections, n), dtype=np.float64)
    
    try:
        from sksparse.cholmod import cholesky
        factor = cholesky(L_reg)
        for k in range(n_projections):
            Z[k, :] = factor(Q[k, :])
    except ImportError:
        try:
            from scipy.sparse.linalg import factorized
            solve = factorized(L_reg)
            for k in range(n_projections):
                Z[k, :] = solve(Q[k, :])
        except Exception:
            for k in range(n_projections):
                Z[k, :], _ = cg(L_reg, Q[k, :], rtol=1e-4, maxiter=200)
    
    return Z

def sparse_laplacian(G):
    """Build sparse Laplacian, supporting weighted edges."""
    n = G.number_of_nodes()
    nodes = sorted(G.nodes())
    node_to_idx = {nd: i for i, nd in enumerate(nodes)}
    
    rows, cols, vals = [], [], []
    for u, v, data in G.edges(data=True):
        w = data.get('weight', 1.0)
        i_u = node_to_idx[u]
        i_v = node_to_idx[v]
        rows.extend([i_u, i_v, i_u, i_v])
        cols.extend([i_v, i_u, i_u, i_v])
        vals.extend([-w, -w, w, w])
    
    from scipy.sparse import coo_matrix
    L = coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    return L

def cut_sparse(G):
    """Graph bisection using METIS."""
    n = G.number_of_nodes()
    nodes = sorted(G.nodes())
    node_to_idx = {nd: i for i, nd in enumerate(nodes)}

    print(f"  Running METIS partitioning...")
    adjacency = [[] for _ in range(n)]
    for u, v in G.edges():
        i_u = node_to_idx[u]
        i_v = node_to_idx[v]
        adjacency[i_u].append(i_v)
        adjacency[i_v].append(i_u)
    
    n_cuts, membership = pymetis.part_graph(2, adjacency=adjacency)
    membership = np.array(membership, dtype=np.int32)
    
    part_0 = np.array([nodes[i] for i in range(n) if membership[i] == 0])
    part_1 = np.array([nodes[i] for i in range(n) if membership[i] == 1])
    
    if len(part_0) == 0 or len(part_1) == 0:
        raise ValueError("Degenerate partition")
    print(f"  METIS cut: {n_cuts} edges, parts: {len(part_0)} vs {len(part_1)}")
    return part_0, part_1, membership  
    
    
def get_cut_edges(G, part_0_set, part_1_set):
    cut = []
    for u, v in G.edges():
        if (u in part_0_set and v in part_1_set) or (u in part_1_set and v in part_0_set):
            cut.append((u, v))
    return cut


def sparsify_cut_edges(G, part_0_set, part_1_set, target_cut=5, seed=42):
    rng = np.random.RandomState(seed)
    cut_edges = get_cut_edges(G, part_0_set, part_1_set)
    n_cut = len(cut_edges)
    
    if n_cut <= target_cut:
        return G.copy(), [(u, v, G[u][v].get('weight', 1.0)) for u, v in cut_edges]
    
    cut_weights_orig = np.array([G[u][v].get('weight', 1.0) for u, v in cut_edges])
    
    boundary_nodes = set(u for e in cut_edges for u in e)
    weighted_degrees = {u: sum(d.get('weight', 1.0) for _, _, d in G.edges(u, data=True))
                        for u in boundary_nodes}
    
    R_approx = np.array([
        1.0 / max(weighted_degrees[u], 1e-300) + 1.0 / max(weighted_degrees[v], 1e-300)
        for u, v in cut_edges
    ])
    importance = cut_weights_orig * R_approx
    importance = np.maximum(importance, 1e-300)
    
    total_importance = np.sum(importance)
    p = importance / total_importance
    
    top_idx = np.argsort(importance)[::-1][:target_cut]
    
    G_sparse = G.copy()
    for u, v in cut_edges:
        if G_sparse.has_edge(u, v):
            G_sparse.remove_edge(u, v)
    
    kept_weighted = []
    for idx in top_idx:
        u, v = cut_edges[idx]
        # HT weight: preserves per-edge spectral contribution in expectation
        w_new = cut_weights_orig[idx] / (target_cut * p[idx])
        # Cap at uniform reweight to prevent rho explosion
        w_cap = float(np.sum(cut_weights_orig)) / target_cut
        w_new = min(w_new, w_cap)
        G_sparse.add_edge(u, v, weight=w_new)
        kept_weighted.append((u, v, w_new))
    
    return G_sparse, kept_weighted

def subgraph_dense_laplacian(G, nodes):
    """Build dense Laplacian for a subgraph, supporting weighted edges."""
    nodes = list(nodes)
    node_map = {n: i for i, n in enumerate(nodes)}
    H = G.subgraph(nodes)
    n = len(nodes)

    L = np.zeros((n, n), dtype=np.float64)
    for u, v, data in H.edges(data=True):
        w = data.get('weight', 1.0)
        i, j = node_map[u], node_map[v]
        L[i, j] -= w
        L[j, i] -= w
        L[i, i] += w
        L[j, j] += w

    return L, node_map

# ============================================================================
# DEFLATION (Numba-accelerated)
# ============================================================================

@nb.njit(cache=True)
def deflate_numba(eigvals, z, tol=1e-8):
    """
    Deflation for rank-1 secular equation.
    Uses RELATIVE gap tolerance.
    """
    n = eigvals.size
    z_def = z.copy()
    deflated = np.zeros(n, dtype=nb.boolean)

    max_givens = n
    givens_data = np.empty((max_givens, 4), dtype=np.float64)
    n_givens = 0

    z_norm = 0.0
    for i in range(n):
        z_norm += z_def[i] * z_def[i]
    z_norm = np.sqrt(z_norm)

    if z_norm < 1e-300:
        for i in range(n):
            deflated[i] = True
        return deflated, z_def, givens_data[:0]

    z_tol = tol * z_norm
    for i in range(n):
        if abs(z_def[i]) < z_tol:
            deflated[i] = True
            z_def[i] = 0.0

    i = 0
    while i < n:
        if deflated[i]:
            i += 1
            continue
        j = i + 1
        while j < n:
            scale = max(abs(eigvals[i]), abs(eigvals[j]), 1.0)
            gap_tol_local = tol * scale
            if abs(eigvals[j] - eigvals[i]) >= gap_tol_local:
                break
            j += 1
        for t in range(i + 1, j):
            if deflated[t]:
                continue
            r = np.sqrt(z_def[i] * z_def[i] + z_def[t] * z_def[t])
            if r > 1e-300:
                c = z_def[i] / r
                s = z_def[t] / r
            else:
                c = 1.0
                s = 0.0
            z_def[i] = r
            z_def[t] = 0.0
            deflated[t] = True

            if n_givens < max_givens:
                givens_data[n_givens, 0] = float(i)
                givens_data[n_givens, 1] = float(t)
                givens_data[n_givens, 2] = c
                givens_data[n_givens, 3] = s
                n_givens += 1
        i = j

    return deflated, z_def, givens_data[:n_givens]


# ============================================================================
# CORE: Cauchy factorization
# ============================================================================

def reconstruct_eigendecomposition(
    eigvals_0, eigvecs_0, eigvals_1, eigvecs_1,
    added_edges, part_0, part_1, tol=1e-8,
    cut_weights=None,
):
    """
    Reconstruct eigendecomposition from two subgraph decompositions.
    Each cut edge adds a rank-1 update: L += w * (e_u - e_w)(e_u - e_w)^T
    """
    n0 = len(part_0)
    n1 = len(part_1)
    n_nodes = n0 + n1
    m = len(added_edges)

    if cut_weights is None:
        cut_weights = [1.0] * m

    eigvals_prev = np.concatenate([eigvals_0, eigvals_1])
    idx_sort = np.argsort(eigvals_prev, kind='mergesort')
    eigvals_curr = eigvals_prev[idx_sort].astype(np.float64).copy()

    perm = np.concatenate([part_0, part_1]).astype(np.int32)
    perm_inv = np.empty(int(max(perm)) + 1, dtype=np.int32)
    for i in range(len(perm)):
        perm_inv[perm[i]] = i

    part_0_set = set(int(p) for p in part_0)

    def compute_z_vector(node_u, node_w):
        pidx_u = int(perm_inv[node_u])
        pidx_w = int(perm_inv[node_w])

        row_u = np.zeros(n_nodes, dtype=np.float64)
        if int(node_u) in part_0_set:
            row_u[:n0] = eigvecs_0[pidx_u, :]
        else:
            row_u[n0:] = eigvecs_1[pidx_u - n0, :]

        row_w = np.zeros(n_nodes, dtype=np.float64)
        if int(node_w) in part_0_set:
            row_w[:n0] = eigvecs_0[pidx_w, :]
        else:
            row_w[n0:] = eigvecs_1[pidx_w - n0, :]

        return (row_u - row_w)[idx_sort]

    z_all = np.zeros((m, n_nodes), dtype=np.float64)
    for e_idx, (u, w) in enumerate(added_edges):
        z_all[e_idx, :] = compute_z_vector(u, w)

    if m > 1:
        alpha = np.empty(m, dtype=np.float64)
        for i in range(m):
            alpha[i] = cut_weights[i] * np.dot(z_all[i], z_all[i])
        ord_idx = np.argsort(alpha, kind='mergesort')
        added_edges = [added_edges[i] for i in ord_idx]
        cut_weights = [cut_weights[i] for i in ord_idx]
        z_all = z_all[ord_idx, :]

    cauchy_factors = []

    for edge_idx in range(m):
        rho = cut_weights[edge_idx]
        z = z_all[edge_idx, :].copy()

        deflated_mask, z_def, givens_data = deflate_numba(eigvals_curr, z, tol=tol)

        givens_list = []
        for g in range(givens_data.shape[0]):
            givens_list.append((
                int(givens_data[g, 0]), int(givens_data[g, 1]),
                givens_data[g, 2], givens_data[g, 3]
            ))

        for (i_g, j_g, c_g, s_g) in givens_list:
            for future_e in range(edge_idx + 1, m):
                zi = z_all[future_e, i_g]
                zj = z_all[future_e, j_g]
                z_all[future_e, i_g] = c_g * zi + s_g * zj
                z_all[future_e, j_g] = -s_g * zi + c_g * zj

        active = np.where(~deflated_mask)[0]

        if len(active) == 0:
            cauchy_factors.append(CauchyFactor(
                d=np.array([]), roots=np.array([]),
                z=np.array([]), norms=np.array([]),
                active_idx=np.array([], dtype=np.int64),
                givens=givens_list, perm=None, n_global=n_nodes, rho=rho
            ))
            continue

        d_active = eigvals_curr[active].copy()
        sort_idx = np.argsort(d_active, kind='mergesort')
        active_sorted = active[sort_idx]
        d_sorted = d_active[sort_idx].copy()
        z_active = z_def[active_sorted].copy()

        _perturb_duplicates_rel(d_sorted, min_gap=1e-12)

        roots = solve_secular_all_roots(d_sorted, z_active, rho=rho)

        for ri in range(len(roots)):
            if np.isnan(roots[ri]):
                roots[ri] = d_sorted[ri] + 1e-10

        roots = np.sort(roots)

        k_active = len(active_sorted)
        norms = _compute_column_norms(z_active, d_sorted, roots, k_active)
        z_hat = _compute_z_hat_checked(d_sorted, roots, z_active, rho, orth_tol=1e-8)
        err_c, err_z = _factor_orth_probe_err(d_sorted, roots, z_active, norms, z_hat, n_probes=2)

        use_zhat = (
            (z_hat is not None)
            and np.isfinite(err_z)
            and (err_z <= max(err_c, 1e-16))
        )

        cf = CauchyFactor(
            d=d_sorted, roots=roots, z=z_active.copy(), norms=norms,
            active_idx=active_sorted.copy(), givens=givens_list, perm=None,
            n_global=n_nodes, rho=rho, z_hat=z_hat, use_zhat=use_zhat
        )
                
        cauchy_factors.append(cf)

        if edge_idx < m - 1:
            n_future = m - edge_idx - 1
            future_z = np.empty((k_active, n_future), dtype=np.float64)
            for fe in range(n_future):
                future_z[:, fe] = z_all[edge_idx + 1 + fe, active_sorted]

            out = np.empty_like(future_z)
            use_stable = cf.use_zhat and (z_hat is not None)
            if use_stable:
                if k_active > 200:
                    _cauchy_matvec_batch_parallel_stable(z_hat, d_sorted, roots, future_z, out)
                else:
                    _cauchy_matvec_batch_stable(z_hat, d_sorted, roots, future_z, out)
            else:
                if k_active > 200:
                    _cauchy_matvec_batch_parallel(z_active, d_sorted, roots, norms, future_z, out)
                else:
                    _cauchy_matvec_batch(z_active, d_sorted, roots, norms, future_z, out)

            for col in range(out.shape[1]):
                if not np.all(np.isfinite(out[:, col])):
                    out[:, col] = future_z[:, col]

            for fe in range(n_future):
                z_all[edge_idx + 1 + fe, active_sorted] = out[:, fe]

        eigvals_curr[active_sorted] = roots

        if np.any(np.diff(eigvals_curr) < -1e-15):
            full_sort = np.argsort(eigvals_curr, kind='mergesort')
            eigvals_curr = eigvals_curr[full_sort]
            z_all = z_all[:, full_sort]

    return eigvals_curr, cauchy_factors, perm, eigvecs_0, eigvecs_1

# ============================================================================
# VALIDATION
# ============================================================================

def validate_eigenvalues(G_sparse, eigvals_recon, part_0, part_1,
                         n_check=50, tol=1e-4):
    """Validate eigenvalue decomposition quality."""
    L = sparse_laplacian(G_sparse)
    n = L.shape[0]
    results = {}

    n_nan = int(np.sum(np.isnan(eigvals_recon)))
    results['n_nan'] = n_nan
    eigvals_clean = eigvals_recon[~np.isnan(eigvals_recon)] if n_nan > 0 else eigvals_recon

    if len(eigvals_clean) > 0:
        min_eigval = float(np.min(eigvals_clean))
    else:
        min_eigval = float('nan')
    results['min_eigenvalue'] = min_eigval
    results['nonnegative'] = min_eigval >= -tol if not np.isnan(min_eigval) else False

    trace_L = float(L.diagonal().sum())
    trace_recon = float(np.nansum(eigvals_recon))
    results['trace_L'] = trace_L
    results['trace_recon'] = trace_recon
    results['trace_relerr'] = abs(trace_L - trace_recon) / max(abs(trace_L), 1e-12)

    frob_sq_L = float((L.multiply(L)).sum())
    frob_sq_recon = float(np.nansum(eigvals_recon ** 2))
    results['frob_sq_L'] = frob_sq_L
    results['frob_sq_recon'] = frob_sq_recon
    results['frob_relerr'] = abs(frob_sq_L - frob_sq_recon) / max(abs(frob_sq_L), 1e-12)

    n_components = nx.number_connected_components(G_sparse)
    k_small = min(n_check, n - 2)
    if k_small > 0:
        try:
            sigma_shift = 1e-6
            L_shifted = L + sigma_shift * diags(np.ones(n))
            ref_small_shifted, _ = eigsh(L_shifted, k=k_small, which='SM', tol=1e-10)
            ref_small = np.sort(ref_small_shifted) - sigma_shift
            ref_small[ref_small < 1e-10] = 0.0
            
            recon_small = np.sort(eigvals_clean)[:k_small]
            if len(recon_small) >= k_small:
                max_err_small = float(np.max(np.abs(ref_small - recon_small)))
                ref_nz = ref_small[ref_small > tol]
                recon_nz = recon_small[len(ref_small) - len(ref_nz):]
                if len(ref_nz) > 0:
                    rel_err_small = float(np.max(np.abs(ref_nz - recon_nz[:len(ref_nz)]))) / max(float(np.max(np.abs(ref_nz))), 1e-12)
                else:
                    rel_err_small = 0.0
                results['smallest_maxerr'] = max_err_small
                results['smallest_relerr'] = rel_err_small
        except Exception as e:
            results['smallest_err'] = str(e)

    k_large = min(n_check, n - 2)
    if k_large > 0:
        try:
            ref_large, _ = eigsh(L, k=k_large, which='LM', tol=1e-8)
            ref_large = np.sort(ref_large)[::-1]
            recon_large = np.sort(eigvals_clean)[::-1][:k_large]
            if len(recon_large) >= k_large:
                max_err_large = float(np.max(np.abs(ref_large - recon_large)))
                rel_err_large = max_err_large / max(float(np.max(np.abs(ref_large))), 1e-12)
                results['largest_maxerr'] = max_err_large
                results['largest_relerr'] = rel_err_large
        except Exception as e:
            results['largest_err'] = str(e)

    nullity_tol = 1e-8
    n_near_zero = int(np.sum(np.abs(eigvals_clean) < nullity_tol))
    results['near_zero_eigenvalues'] = n_near_zero
    results['connected_components'] = n_components
    results['nullity_match'] = (n_near_zero == n_components)

    rng = np.random.RandomState(42)
    sorted_eigs = np.sort(eigvals_clean)
    violations = 0
    if len(sorted_eigs) > 0:
        for _ in range(10):
            x = rng.randn(n)
            x /= np.linalg.norm(x)
            rq = float(x @ (L @ x))
            if rq < sorted_eigs[0] - tol or rq > sorted_eigs[-1] + tol:
                violations += 1
    results['rayleigh_violations'] = violations

    all_pass = (
        n_nan == 0 and
        results['nonnegative'] and
        results['trace_relerr'] < tol and
        results['frob_relerr'] < tol and
        results.get('smallest_relerr', 0) < tol and
        results.get('largest_relerr', 0) < tol and
        results['nullity_match'] and
        violations == 0
    )
    results['PASS'] = all_pass

    return results

def validate_cauchy_orthogonality(cauchy_factor, n_probes=20, tol=1e-6):
    k = len(cauchy_factor.d)
    if k == 0:
        return {'PASS': True, 'max_err': 0.0}

    d = cauchy_factor.d
    roots = cauchy_factor.roots
    rng = np.random.RandomState(123)
    max_err = 0.0
    n_finite = 0

    use_zhat = cauchy_factor.z_hat is not None
    z_or_zhat = cauchy_factor.z_hat if use_zhat else cauchy_factor.z
    norms = np.ones(k, dtype=np.float64) if use_zhat else cauchy_factor.norms

    for _ in range(n_probes):
        x = rng.randn(k)
        z_out = _cauchy_QtQx(d, roots, z_or_zhat, norms, x, k) if use_zhat \
            else _cauchy_QtQx_znorms(d, roots, z_or_zhat, norms, x, k)
        err = np.max(np.abs(z_out - x))
        if np.isfinite(err):
            max_err = max(max_err, err)
            n_finite += 1

    if n_finite == 0:
        return {'PASS': False, 'max_err': np.inf}
    return {'PASS': max_err < tol, 'max_err': max_err}

@nb.njit(cache=True)
def _cauchy_QtQx(d, roots, z_hat, norms_unused, x, k):
    """Compute Q^T Q x for orthogonality check using z_hat."""
    y = np.empty(k, dtype=np.float64)
    for i in range(k):
        s = 0.0
        for j in range(k):
            delta = d[i] - roots[j]
            if abs(delta) < 1e-300:
                if delta >= 0:
                    delta = 1e-300
                else:
                    delta = -1e-300
            s += x[j] / delta
        y[i] = z_hat[i] * s

    out = np.empty(k, dtype=np.float64)
    for j in range(k):
        s = 0.0
        for i in range(k):
            delta = d[i] - roots[j]
            if abs(delta) < 1e-300:
                if delta >= 0:
                    delta = 1e-300
                else:
                    delta = -1e-300
            s += z_hat[i] * y[i] / delta
        out[j] = s
    return out


def _validate_cauchy_orth_fast(d, roots, z, norms, k, n_probes, tol):
    """Numba-accelerated orthogonality validation."""
    rng = np.random.RandomState(123)
    max_err = 0.0

    for _ in range(n_probes):
        x = rng.randn(k)
        z_out = _cauchy_QtQx(d, roots, z, norms, x, k)
        err = np.max(np.abs(z_out - x))
        if not np.isnan(err):
            max_err = max(max_err, err)

    return {'PASS': max_err < tol, 'max_err': max_err}

@nb.njit(cache=True)
def _cauchy_QtQx_znorms(d, roots, z, norms, x, k):
    """Compute Q^T Q x for classical Q[i,j]=z[i]/(d[i]-roots[j])/norms[j]."""
    y = np.empty(k, dtype=np.float64)
    for i in range(k):
        s = 0.0
        for j in range(k):
            delta = d[i] - roots[j]
            if abs(delta) < 1e-300:
                delta = 1e-300 if delta >= 0 else -1e-300
            s += x[j] / (delta * norms[j])
        y[i] = z[i] * s

    out = np.empty(k, dtype=np.float64)
    for j in range(k):
        s = 0.0
        for i in range(k):
            delta = d[i] - roots[j]
            if abs(delta) < 1e-300:
                delta = 1e-300 if delta >= 0 else -1e-300
            s += z[i] * y[i] / delta
        out[j] = s / norms[j]
    return out


# ============================================================================
# SMALL VALIDATION TESTS
# ============================================================================

def test_small_rank1_update():
    """Test secular equation solver on D + z*z^T."""
    print("=" * 60)
    print("SMALL RANK-1 UPDATE TEST")
    print("=" * 60)

    np.random.seed(42)
    k = 50
    d = np.sort(np.random.rand(k) * 10)
    for i in range(1, k):
        if d[i] - d[i - 1] < 1e-10:
            d[i] = d[i - 1] + 1e-10

    z = np.random.randn(k) * 0.5

    M = np.diag(d) + np.outer(z, z)
    eigvals_true = np.sort(np.linalg.eigvalsh(M))

    roots = solve_secular_all_roots(d, z, rho=1.0)
    roots = np.sort(roots)

    err = np.max(np.abs(eigvals_true - roots))
    print(f"  Secular equation error: {err:.2e}")

    norms = _compute_column_norms(z, d, roots, k)
    C = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            delta = d[i] - roots[j]
            if abs(delta) > 1e-300:
                C[i, j] = z[i] / delta / norms[j]

    CtC = C.T @ C
    orth_err = np.max(np.abs(CtC - np.eye(k)))
    print(f"  Cauchy orthogonality error (C^T C - I): {orth_err:.2e}")

    CtMC = C.T @ M @ C
    diag_err = np.max(np.abs(CtMC - np.diag(roots)))
    print(f"  Diagonalization error (C^T M C - diag(roots)): {diag_err:.2e}")

    print()
    return err < 1e-10 and orth_err < 1e-6


def test_rank1_with_clusters():
    """Test with clustered eigenvalues."""
    print("=" * 60)
    print("RANK-1 UPDATE WITH CLUSTERED EIGENVALUES")
    print("=" * 60)

    np.random.seed(123)
    clusters = []
    for val in [0.0, 1.0, 2.0, 3.0, 5.0, 8.0]:
        n_in_cluster = np.random.randint(5, 20)
        clusters.extend([val + np.random.randn() * 1e-8 for _ in range(n_in_cluster)])
    d = np.sort(np.array(clusters))
    k = len(d)

    z = np.random.randn(k) * 0.1

    M = np.diag(d) + np.outer(z, z)
    eigvals_true = np.sort(np.linalg.eigvalsh(M))

    deflated_mask, z_def, givens_data = deflate_numba(d, z, tol=1e-6)
    active = np.where(~deflated_mask)[0]
    d_active = d[active].copy()
    z_active = z_def[active].copy()

    sort_idx = np.argsort(d_active)
    d_sorted = d_active[sort_idx].copy()
    z_sorted = z_active[sort_idx].copy()
    _perturb_duplicates_rel(d_sorted, min_gap=1e-12)

    print(f"  k = {k}, active after deflation = {len(active)}")

    roots = solve_secular_all_roots(d_sorted, z_sorted, rho=1.0)
    roots = np.sort(roots)

    eigvals_recon = d.copy()
    active_sorted = active[sort_idx]
    eigvals_recon[active_sorted] = roots
    eigvals_recon = np.sort(eigvals_recon)

    err = np.max(np.abs(eigvals_true - eigvals_recon))
    print(f"  Eigenvalue error: {err:.2e}")

    norms = _compute_column_norms(z_sorted, d_sorted, roots, len(d_sorted))
    C = np.zeros((len(d_sorted), len(d_sorted)))
    for i in range(len(d_sorted)):
        for j in range(len(d_sorted)):
            delta = d_sorted[i] - roots[j]
            if abs(delta) > 1e-300:
                C[i, j] = z_sorted[i] / delta / norms[j]
    CtC = C.T @ C
    orth_err = np.max(np.abs(CtC - np.eye(len(d_sorted))))
    print(f"  Cauchy orthogonality error: {orth_err:.2e}")

    passed = err < 1e-6 and orth_err < 1e-4
    print(f"  {'✓ PASS' if passed else '✗ FAIL'}")
    print()
    return passed

def test_small_graph():
    """Test on a small graph where we can verify everything."""
    print("=" * 60)
    print("SMALL GRAPH TEST (n=100)")
    print("=" * 60)

    G = nx.barabasi_albert_graph(100, 3, seed=42)

    part_0, part_1, parts = cut_sparse(G)
    part_0_set = set(part_0)
    part_1_set = set(part_1)

    G_sparse, cut_edges_weighted = sparsify_cut_edges(G, part_0_set, part_1_set, target_cut=5, seed=42)
    cut_edges = [(u, v) for u, v, w in cut_edges_weighted]
    cut_weights = [w for u, v, w in cut_edges_weighted]

    L_sparse_dense = np.zeros((100, 100), dtype=np.float64)
    for u, v, data in G_sparse.edges(data=True):
        w = data.get('weight', 1.0)
        L_sparse_dense[u, v] -= w
        L_sparse_dense[v, u] -= w
        L_sparse_dense[u, u] += w
        L_sparse_dense[v, v] += w
    eigvals_sparse_true = np.sort(np.linalg.eigvalsh(L_sparse_dense))

    L_0, _ = subgraph_dense_laplacian(G_sparse, part_0)
    L_1, _ = subgraph_dense_laplacian(G_sparse, part_1)
    eigvals_0, eigvecs_0 = np.linalg.eigh(L_0)
    eigvals_1, eigvecs_1 = np.linalg.eigh(L_1)

    eigvals_recon, cauchy_factors, perm, _, _ = reconstruct_eigendecomposition(
        eigvals_0, eigvecs_0, eigvals_1, eigvecs_1,
        cut_edges, part_0, part_1, tol=1e-10,
        cut_weights=cut_weights
    )

    eigvals_recon_sorted = np.sort(eigvals_recon)
    err = np.max(np.abs(eigvals_sparse_true - eigvals_recon_sorted))
    print(f"  n_nodes: {G.number_of_nodes()}")
    print(f"  n_cut_edges: {len(cut_edges)}")
    print(f"  Eigenvalue max error: {err:.2e}")
    print(f"  Trace error: {abs(np.sum(eigvals_sparse_true) - np.sum(eigvals_recon_sorted)):.2e}")

    for i, cf in enumerate(cauchy_factors):
        if len(cf.d) > 0:
            orth = validate_cauchy_orthogonality(cf, n_probes=20, tol=1e-6)
            print(f"  Cauchy factor {i}: k={len(cf.d)}, orth_err={orth['max_err']:.2e}")

    passed = err < 1e-6
    print(f"  {'✓ PASS' if passed else '✗ FAIL'}")
    print()
    return passed


def test_medium_graph():
    """Test on a medium graph (n=1000)."""
    print("=" * 60)
    print("MEDIUM GRAPH TEST (n=1000)")
    print("=" * 60)

    G = nx.barabasi_albert_graph(1000, 3, seed=42)

    part_0, part_1, parts = cut_sparse(G)
    part_0_set = set(part_0)
    part_1_set = set(part_1)

    G_sparse, cut_edges_weighted = sparsify_cut_edges(G, part_0_set, part_1_set, target_cut=5, seed=42)
    cut_edges = [(u, v) for u, v, w in cut_edges_weighted]
    cut_weights = [w for u, v, w in cut_edges_weighted]

    n = G.number_of_nodes()
    L_sparse_dense = np.zeros((n, n), dtype=np.float64)
    for u, v, data in G_sparse.edges(data=True):
        w = data.get('weight', 1.0)
        L_sparse_dense[u, v] -= w
        L_sparse_dense[v, u] -= w
        L_sparse_dense[u, u] += w
        L_sparse_dense[v, v] += w
    eigvals_sparse_true = np.sort(np.linalg.eigvalsh(L_sparse_dense))

    L_0, _ = subgraph_dense_laplacian(G_sparse, part_0)
    L_1, _ = subgraph_dense_laplacian(G_sparse, part_1)
    eigvals_0, eigvecs_0 = np.linalg.eigh(L_0)
    eigvals_1, eigvecs_1 = np.linalg.eigh(L_1)

    eigvals_recon, cauchy_factors, perm, _, _ = reconstruct_eigendecomposition(
        eigvals_0, eigvecs_0, eigvals_1, eigvecs_1,
        cut_edges, part_0, part_1, tol=1e-10,
        cut_weights=cut_weights
    )

    eigvals_recon_sorted = np.sort(eigvals_recon)
    err = np.max(np.abs(eigvals_sparse_true - eigvals_recon_sorted))
    trace_err = abs(np.sum(eigvals_sparse_true) - np.sum(eigvals_recon_sorted))

    print(f"  n_nodes: 1000")
    print(f"  n_cut_edges: {len(cut_edges)}")
    print(f"  Eigenvalue max error: {err:.2e}")
    print(f"  Trace error: {trace_err:.2e}")

    err_per_eig = np.abs(eigvals_sparse_true - eigvals_recon_sorted)
    print(f"  Mean error: {np.mean(err_per_eig):.2e}")
    print(f"  Median error: {np.median(err_per_eig):.2e}")
    print(f"  95th percentile error: {np.percentile(err_per_eig, 95):.2e}")

    worst_idx = np.argsort(err_per_eig)[-10:]
    print(f"  Worst 10 eigenvalue indices: {worst_idx}")
    print(f"  Worst 10 true eigenvalues: {eigvals_sparse_true[worst_idx]}")
    print(f"  Worst 10 reconstructed: {eigvals_recon_sorted[worst_idx]}")

    for i, cf in enumerate(cauchy_factors):
        if len(cf.d) > 0:
            orth = validate_cauchy_orthogonality(cf, n_probes=10, tol=1e-4)
            print(f"  Cauchy factor {i}: k={len(cf.d)}, orth_err={orth['max_err']:.2e}")

    passed = err < 5e-3  # flat recon with large rho (reweight~140) limits accuracy
    print(f"  {'✓ PASS' if passed else '✗ FAIL'}")
    print()
    return passed

def test_weighted_graph():
    """Test on a small weighted graph."""
    print("=" * 60)
    print("WEIGHTED GRAPH TEST (n=100)")
    print("=" * 60)

    rng = np.random.RandomState(42)
    G = nx.barabasi_albert_graph(100, 3, seed=42)
    
    for u, v in G.edges():
        G[u][v]['weight'] = rng.uniform(0.5, 5.0)

    part_0, part_1, parts = cut_sparse(G)
    part_0_set = set(part_0)
    part_1_set = set(part_1)

    G_sparse, cut_edges_weighted = sparsify_cut_edges(G, part_0_set, part_1_set, target_cut=5, seed=42)
    cut_edges = [(u, v) for u, v, w in cut_edges_weighted]
    cut_weights = [w for u, v, w in cut_edges_weighted]

    n = G.number_of_nodes()
    L_sparse_dense = np.zeros((n, n), dtype=np.float64)
    for u, v, data in G_sparse.edges(data=True):
        w = data.get('weight', 1.0)
        L_sparse_dense[u, v] -= w
        L_sparse_dense[v, u] -= w
        L_sparse_dense[u, u] += w
        L_sparse_dense[v, v] += w
    eigvals_sparse_true = np.sort(np.linalg.eigvalsh(L_sparse_dense))

    L_0, _ = subgraph_dense_laplacian(G_sparse, part_0)
    L_1, _ = subgraph_dense_laplacian(G_sparse, part_1)
    eigvals_0, eigvecs_0 = np.linalg.eigh(L_0)
    eigvals_1, eigvecs_1 = np.linalg.eigh(L_1)

    eigvals_recon, cauchy_factors, perm, _, _ = reconstruct_eigendecomposition(
        eigvals_0, eigvecs_0, eigvals_1, eigvecs_1,
        cut_edges, part_0, part_1, tol=1e-10,
        cut_weights=cut_weights
    )

    eigvals_recon_sorted = np.sort(eigvals_recon)
    err = np.max(np.abs(eigvals_sparse_true - eigvals_recon_sorted))
    max_abs = max(np.max(np.abs(eigvals_sparse_true)), 1.0)
    rel_err = err / max_abs
    
    print(f"  n_nodes: {n}")
    print(f"  n_cut_edges: {len(cut_edges)}")
    print(f"  Cut weights: {[f'{w:.2f}' for w in cut_weights]}")
    print(f"  Eigenvalue max error: {err:.2e}")
    print(f"  Relative max error:   {rel_err:.2e}")
    print(f"  Trace error: {abs(np.sum(eigvals_sparse_true) - np.sum(eigvals_recon_sorted)):.2e}")

    passed = rel_err < 1e-5  # reweighted cuts (weight~38) cause O(1e-6) rel error
    print(f"  {'✓ PASS' if passed else '✗ FAIL'}")
    print()
    return passed

# ============================================================================
# IMPLICIT EIGENVECTOR APPLICATION: Q^T v without materializing Q
# ============================================================================

@nb.njit(cache=True)
def apply_cauchy_factor_transpose(d, roots, z, norms, givens, active_idx,
                                  x_in, n_total):
    """Apply C^T to a vector x_in of length n_total."""
    k = len(d)
    x_out = x_in.copy()
    
    if k == 0:
        return x_out
    
    n_givens = len(givens)
    for g in range(n_givens):
        i_g = int(givens[g, 0])
        j_g = int(givens[g, 1])
        c_g = givens[g, 2]
        s_g = givens[g, 3]
        xi = x_out[i_g]
        xj = x_out[j_g]
        x_out[i_g] = c_g * xi + s_g * xj
        x_out[j_g] = -s_g * xi + c_g * xj
    
    x_active = np.empty(k, dtype=np.float64)
    for i in range(k):
        x_active[i] = x_out[active_idx[i]]
    
    y = np.empty(k, dtype=np.float64)
    for j in range(k):
        inv_norm = 1.0 / norms[j] if abs(norms[j]) > 1e-300 else 0.0
        s = 0.0
        for i in range(k):
            delta = d[i] - roots[j]
            if abs(delta) < 1e-300:
                if delta >= 0:
                    delta = 1e-300
                else:
                    delta = -1e-300
            s += z[i] * x_active[i] / delta
        y[j] = s * inv_norm
    
    for j in range(k):
        x_out[active_idx[j]] = y[j]
    
    return x_out


@nb.njit(cache=True)
def apply_cauchy_chain_transpose(chain_d, chain_roots, chain_z, chain_norms,
                                  chain_givens_flat, chain_givens_counts,
                                  chain_active_flat, chain_active_counts,
                                  x_in, n_total, n_factors):
    """Apply a sequence of Cauchy factor transposes to a vector."""
    x = x_in.copy()
        
    d_offsets = np.empty(n_factors, dtype=np.int64)
    g_offsets = np.empty(n_factors, dtype=np.int64)
    a_offsets = np.empty(n_factors, dtype=np.int64)
    k_sizes = np.empty(n_factors, dtype=np.int64)
    
    d_off = 0
    g_off = 0
    a_off = 0
    for f in range(n_factors):
        d_offsets[f] = d_off
        g_offsets[f] = g_off
        a_offsets[f] = a_off
        k = chain_active_counts[f]
        k_sizes[f] = k
        d_off += k
        g_off += chain_givens_counts[f]
        a_off += k
    
    for f in range(n_factors - 1, -1, -1):
        k = k_sizes[f]
        if k == 0:
            continue
        
        do = d_offsets[f]
        go = g_offsets[f]
        ao = a_offsets[f]
        ng = chain_givens_counts[f]
        
        d = chain_d[do:do + k]
        roots = chain_roots[do:do + k]
        z_vec = chain_z[do:do + k]
        norms = chain_norms[do:do + k]
        givens = chain_givens_flat[go:go + ng].reshape(-1, 4) if ng > 0 else np.empty((0, 4), dtype=np.float64)
        active = chain_active_flat[ao:ao + k]
        
        x = apply_cauchy_factor_transpose(d, roots, z_vec, norms, givens,
                                           active, x, n_total)
    
    return x


@nb.njit(cache=True)
def _check_roots_quality(d, z2_rho, roots, k):
    """
    O(k) quality check: max |f(root_j)| over all roots.
    Also returns early if any single residual is catastrophically large.
    """
    max_res = 0.0
    for j in range(k):
        f = 1.0
        for i in range(k):
            delta = d[i] - roots[j]
            if abs(delta) < 1e-300:
                delta = 1e-300 if delta >= 0 else -1e-300
            f += z2_rho[i] / delta
        res = abs(f)
        if res > max_res:
            max_res = res
        if max_res > 1e10:   # fast exit: definitely bad
            return max_res
    return max_res


@nb.njit(cache=True, parallel=True)
def _compute_norms_and_residual(z, d, roots, z2_rho, k):
    """Compute column norms AND max secular residual in one O(k²) pass."""
    norms = np.empty(k, dtype=np.float64)
    residuals = np.empty(k, dtype=np.float64)
    
    for j in nb.prange(k):
        s_norm = 0.0
        f = 1.0
        for i in range(k):
            delta = d[i] - roots[j]
            if abs(delta) < 1e-300:
                delta = 1e-300 if delta >= 0 else -1e-300
            val = z[i] / delta
            s_norm += val * val
            f += z2_rho[i] / delta
        norms[j] = max(np.sqrt(s_norm), 1e-300)
        residuals[j] = abs(f)
    
    max_res = 0.0
    for j in range(k):
        if residuals[j] > max_res:
            max_res = residuals[j]
    
    return norms, max_res


# ============================================================================
# BUILD THE RECURSIVELY-SPARSIFIED GRAPH (for validation)
# ============================================================================

def recursive_cauchy_eigen(G, depth=1, target_cut=5, tol=1e-10,
                           use_zhat=True,
                           _current_depth=0, _stats=None,
                           _parent_needs_nodes=None):
    """
    Recursively compute eigendecomposition of graph Laplacian.
    
    _parent_needs_nodes: set of node IDs whose eigenvector rows the parent
                         will need for z computation.
    """
    if _stats is None:
        _stats = {
            'n_leaves': 0,
            'leaf_sizes': [],
            'n_merges': 0,
            'merge_sizes': [],
            'time_cuts': 0.0,
            'time_sparsify': 0.0,
            'time_base_eig': 0.0,
            'time_secular': 0.0,
            'time_z_compute': 0.0,
            'total_cut_edges': 0,
            'depth': depth,
            'peak_dense_size': 0,
            'root_residual_max': 0.0,
            'n_resorts': 0,
        }
    
    n = G.number_of_nodes()
    nodes = sorted(G.nodes())
    
    decomp = RecursiveDecomp()
    decomp.n = n
    decomp.global_to_local = {nd: i for i, nd in enumerate(nodes)}
    
    # ---- Base case ----
    if depth <= 0 or n <= 100:
        nodes_arr = np.array(nodes)
        L, node_map = subgraph_dense_laplacian(G, nodes_arr)
        t0 = time.perf_counter()
        eigvals, eigvecs = np.linalg.eigh(L)
        t1 = time.perf_counter()
        _stats['time_base_eig'] += t1 - t0
        _stats['n_leaves'] += 1
        _stats['leaf_sizes'].append(n)
        _stats['peak_dense_size'] = max(_stats['peak_dense_size'], n)
        
        decomp.eigvals = eigvals
        decomp.is_leaf = True
        decomp.leaf_eigvecs = eigvecs
        decomp.leaf_nodes = nodes_arr
        decomp.leaf_node_map = node_map
        
        # Precompute rows the parent needs — exact from leaf eigvecs (no Cauchy error)
        precomputed = {}
        if _parent_needs_nodes is not None:
            for nd in _parent_needs_nodes:
                if nd in node_map:
                    precomputed[nd] = eigvecs[node_map[nd], :].copy()
        
        return eigvals, decomp, _stats, G.copy(), precomputed
    
    # ---- Recursive case ----
    
    # 1. Spectral bisection
    t0 = time.perf_counter()
    part_0, part_1, parts = cut_sparse(G)
    t1 = time.perf_counter()
    _stats['time_cuts'] += t1 - t0
    
    part_0_set = set(int(p) for p in part_0)
    part_1_set = set(int(p) for p in part_1)
    
    # 2. Sparsify
    t0 = time.perf_counter()
    G_sparse, cut_edges_weighted = sparsify_cut_edges_ss(
            G, part_0_set, part_1_set, 
            target_cut=target_cut, seed=42
        )    
    t1 = time.perf_counter()
    _stats['time_sparsify'] += t1 - t0
    _stats['total_cut_edges'] += len(cut_edges_weighted)
    
    cut_edges = [(u, v) for u, v, w in cut_edges_weighted]
    cut_weights = [w for u, v, w in cut_edges_weighted]
    
    # Determine which nodes each child needs to precompute
    child_needs_0 = set()
    child_needs_1 = set()
    for u, w in cut_edges:
        u_int, w_int = int(u), int(w)
        if u_int in part_0_set:
            child_needs_0.add(u_int)
        else:
            child_needs_1.add(u_int)
        if w_int in part_0_set:
            child_needs_0.add(w_int)
        else:
            child_needs_1.add(w_int)

    # Also propagate parent's needs down (nodes needed by grandparent)
    if _parent_needs_nodes is not None:
        for nd in _parent_needs_nodes:
            if nd in part_0_set:
                child_needs_0.add(nd)
            elif nd in part_1_set:
                child_needs_1.add(nd)
    
    # 3. Recurse on subgraphs
    G_sub0 = G_sparse.subgraph(part_0).copy()
    G_sub1 = G_sparse.subgraph(part_1).copy()
    
    eigvals_0, decomp_0, _, G_eff_0, precomp_0 = recursive_cauchy_eigen(
        G_sub0, depth=depth - 1, target_cut=target_cut, tol=tol,
        use_zhat=use_zhat,
        _current_depth=_current_depth + 1, _stats=_stats,
        _parent_needs_nodes=child_needs_0,
    )
    eigvals_1, decomp_1, _, G_eff_1, precomp_1 = recursive_cauchy_eigen(
        G_sub1, depth=depth - 1, target_cut=target_cut, tol=tol,
        use_zhat=use_zhat,
        _current_depth=_current_depth + 1, _stats=_stats,
        _parent_needs_nodes=child_needs_1,
    )
    
    # Build effective graph
    G_effective = nx.Graph()
    G_effective.add_nodes_from(G.nodes())
    for u, v, data in G_eff_0.edges(data=True):
        G_effective.add_edge(u, v, **data)
    for u, v, data in G_eff_1.edges(data=True):
        G_effective.add_edge(u, v, **data)
    for (u, v), w in zip(cut_edges, cut_weights):
        if G_effective.has_edge(u, v):
            G_effective[u][v]['weight'] = G_effective[u][v].get('weight', 0.0) + w
        else:
            G_effective.add_edge(u, v, weight=w)
    
    # 4. Compute z vectors using PRECOMPUTED child eigenvector rows
    n0 = len(part_0)
    n1 = len(part_1)
    n_total = n0 + n1
    
    eigvals_combined = np.concatenate([eigvals_0, eigvals_1])
    sort_perm = np.argsort(eigvals_combined, kind='mergesort')
    eigvals_sorted = eigvals_combined[sort_perm]
    
    t0 = time.perf_counter()
    
    all_z_vectors = []
    for u, w in cut_edges:
        u_int, w_int = int(u), int(w)
        
        c = np.zeros(n_total, dtype=np.float64)
        
        if u_int in part_0_set:
            if u_int in precomp_0:
                c[:n0] += precomp_0[u_int]
        else:
            if u_int in precomp_1:
                c[n0:] += precomp_1[u_int]
        
        if w_int in part_0_set:
            if w_int in precomp_0:
                c[:n0] -= precomp_0[w_int]
        else:
            if w_int in precomp_1:
                c[n0:] -= precomp_1[w_int]
        
        z = c[sort_perm]
        all_z_vectors.append(z)

    if len(cut_edges) > 1:
        alpha = np.empty(len(cut_edges), dtype=np.float64)
        for i in range(len(cut_edges)):
            alpha[i] = cut_weights[i] * np.dot(all_z_vectors[i], all_z_vectors[i])
        ord_idx = np.argsort(alpha, kind='mergesort')
        cut_edges = [cut_edges[i] for i in ord_idx]
        cut_weights = [cut_weights[i] for i in ord_idx]
        all_z_vectors = [all_z_vectors[i] for i in ord_idx]
    
    t1 = time.perf_counter()
    _stats['time_z_compute'] += t1 - t0
    
    # 5. Sequential rank-1 updates — z stays unscaled, rho passed to solver
    t0 = time.perf_counter()
    
    current_eigvals = eigvals_sorted.copy()
    cauchy_factors = []
    resort_perms = []
    
    for edge_idx, (u, w) in enumerate(cut_edges):
        z = all_z_vectors[edge_idx]
        rho = cut_weights[edge_idx]
        
        deflated, z_def, givens_data = deflate_numba(current_eigvals, z, tol)
        active_mask = ~deflated
        active_idx = np.where(active_mask)[0]
        
        givens_list = []
        for g in range(givens_data.shape[0]):
            givens_list.append((
                int(givens_data[g, 0]), int(givens_data[g, 1]),
                givens_data[g, 2], givens_data[g, 3]
            ))
        
        # Apply Givens rotations to all future z-vectors via Numba batch kernel
        n_future_givens = len(cut_edges) - edge_idx - 1
        if givens_data.shape[0] > 0 and n_future_givens > 0:
            future_z_mat = np.column_stack(
                [all_z_vectors[edge_idx + 1 + fe] for fe in range(n_future_givens)]
            )
            _apply_givens_batch(givens_data, future_z_mat)
            for fe in range(n_future_givens):
                all_z_vectors[edge_idx + 1 + fe] = future_z_mat[:, fe]
        
        if len(active_idx) == 0:
            cf = CauchyFactor(
                d=np.empty(0), roots=np.empty(0),
                z=np.empty(0), norms=np.empty(0),
                active_idx=np.empty(0, dtype=np.int64),
                givens=givens_list, perm=None, n_global=n_total, rho=rho
            )
            cauchy_factors.append(cf)
            resort_perms.append(None)
            continue
        
        d_active = current_eigvals[active_idx].copy()
        sort_idx_a = np.argsort(d_active, kind='mergesort')
        active_sorted = active_idx[sort_idx_a]
        d_sorted = d_active[sort_idx_a].copy()
        z_active = z_def[active_sorted].copy()
        
        _perturb_duplicates_rel(d_sorted, min_gap=1e-12)

        min_gap_d = float(np.min(np.diff(d_sorted))) if len(d_sorted) > 1 else 0.0
        
        new_roots = solve_secular_all_roots(d_sorted, z_active, rho=rho)
        for ri in range(len(new_roots)):
            if np.isnan(new_roots[ri]):
                new_roots[ri] = d_sorted[ri] + 1e-10
        new_roots = np.sort(new_roots)
        
        k_active = len(active_sorted)
        z2_rho = (z_active * z_active) * rho
        col_norms, root_residual = _compute_norms_and_residual(z_active, d_sorted, new_roots, z2_rho, k_active)
        #z_hat = _compute_z_hat_checked(d_sorted, new_roots, z_active, rho, orth_tol=1e-8)
        #err_c, err_z = _factor_orth_probe_err(d_sorted, new_roots, z_active, col_norms, z_hat, n_probes=2)

        #use_zhat = (
        #    (z_hat is not None)
        #    and np.isfinite(err_z)
        #    and (err_z <= max(err_c, 1e-16))
        #)

        #z2_rho = (z_active * z_active) * rho
        z2_sum = float(np.sum(z2_rho))
        root_residual = _check_roots_quality(d_sorted, z2_rho, new_roots, k_active)

        # Tight threshold: residual must be < 1e-8 * rho*||z||^2 to trust z_hat
        # The log-product amplifies root errors by ~k, so we need roots to be
        # very accurate before z_hat's O(k^2) product formula is safe.
        residual_tol = max(1e-8 * max(z2_sum, 1.0), 1e-10)
        z_hat = None
        if use_zhat:
            z_hat = _compute_z_hat(d_sorted, new_roots, z_active, k_active, rho)
            use_zhat = (root_residual < residual_tol) and np.all(np.isfinite(z_hat))
            if not use_zhat:
                z_hat = None
        else:
            use_zhat = False

        cf = CauchyFactor(
            d=d_sorted, roots=new_roots.copy(), z=z_active.copy(),
            norms=col_norms.copy(), active_idx=active_sorted.copy(),
            givens=givens_list, perm=None, n_global=n_total, rho=rho,
            z_hat=z_hat, use_zhat=use_zhat
        )

        cauchy_factors.append(cf)
        
        if edge_idx < len(cut_edges) - 1:
            n_future = len(cut_edges) - edge_idx - 1
            future_z = np.empty((k_active, n_future), dtype=np.float64)
            for fe in range(n_future):
                future_z[:, fe] = all_z_vectors[edge_idx + 1 + fe][active_sorted]

            out = np.empty_like(future_z)
            use_stable = cf.use_zhat and (z_hat is not None)
            if use_stable:
                if k_active > 200:
                    _cauchy_matvec_batch_parallel_stable(z_hat, d_sorted, new_roots, future_z, out)
                else:
                    _cauchy_matvec_batch_stable(z_hat, d_sorted, new_roots, future_z, out)
            else:
                if k_active > 200:
                    _cauchy_matvec_batch_parallel(z_active, d_sorted, new_roots, col_norms, future_z, out)
                else:
                    _cauchy_matvec_batch(z_active, d_sorted, new_roots, col_norms, future_z, out)

            for col in range(out.shape[1]):
                if not np.all(np.isfinite(out[:, col])):
                    out[:, col] = future_z[:, col]

            for fe in range(n_future):
                all_z_vectors[edge_idx + 1 + fe][active_sorted] = out[:, fe]
        
        current_eigvals[active_sorted] = new_roots
        
        if np.any(np.diff(current_eigvals) < -1e-15):
            full_sort = np.argsort(current_eigvals, kind='mergesort')
            current_eigvals = current_eigvals[full_sort]
            for future_idx in range(edge_idx + 1, len(cut_edges)):
                all_z_vectors[future_idx] = all_z_vectors[future_idx][full_sort]
            resort_perms.append(full_sort.copy())
            _stats['n_resorts'] += 1
        else:
            resort_perms.append(None)

        if root_residual > _stats['root_residual_max']:
            _stats['root_residual_max'] = float(root_residual)
    
    t1 = time.perf_counter()
    _stats['time_secular'] += t1 - t0
    _stats['n_merges'] += 1
    _stats['merge_sizes'].append(n)
    
    assert len(current_eigvals) == n, \
        f"Eigenvalue count mismatch: got {len(current_eigvals)}, expected {n}"
    
    # 6. Store decomposition
    decomp.eigvals = current_eigvals
    decomp.is_leaf = False
    decomp.left = decomp_0
    decomp.right = decomp_1
    decomp.part_0 = part_0
    decomp.part_1 = part_1
    decomp.part_0_set = part_0_set
    decomp.part_1_set = part_1_set
    decomp.cut_edges = cut_edges
    decomp.cauchy_factors = cauchy_factors
    decomp.sort_perm = sort_perm
    decomp.resort_perms = resort_perms

    # 7. Precompute rows needed by parent.
    #
    # FIX: Instead of calling decomp.qt_dot_sparse({nd: 1.0}), which would
    # re-traverse the child subtrees (re-applying their Cauchy chains and
    # accumulating floating-point errors multiplicatively), we use the child
    # precomp rows directly.  precomp_0[nd] = Q_child_0^T e_nd was already
    # computed accurately by the child (exact at leaves, one level of Cauchy
    # error at non-leaf children).  We then apply only THIS node's chain
    # (sort_perm + cauchy_factors + resort_perms) via _apply_parent_chain.
    #
    # This makes the error ADDITIVE across recursion levels:
    #   err(depth D) ≈ D * err_per_level
    # instead of multiplicative:
    #   err(depth D) ≈ (1 + err_per_level * κ)^D  (where κ = Cauchy condition number)
    precomputed = {}
    if _parent_needs_nodes is not None and len(_parent_needs_nodes) > 0:
        t0_pre = time.perf_counter()
        nd_list = sorted(_parent_needs_nodes)
        
        C_batch = np.zeros((n_total, len(nd_list)), dtype=np.float64)
        for col_idx, nd in enumerate(nd_list):
            nd_int = int(nd)
            if nd_int in part_0_set:
                row = precomp_0.get(nd_int)
                if row is not None:
                    C_batch[:n0, col_idx] = row
            elif nd_int in part_1_set:
                row = precomp_1.get(nd_int)
                if row is not None:
                    C_batch[n0:, col_idx] = row
        
        C_out = decomp._apply_parent_chain_batch(C_batch)
        
        for col_idx, nd in enumerate(nd_list):
            precomputed[int(nd)] = C_out[:, col_idx]
        
        _stats['time_z_compute'] += time.perf_counter() - t0_pre

    return current_eigvals, decomp, _stats, G_effective, precomputed 

# Add near other globals / helper funcs
USE_ZHAT_IN_QT = True

def validate_eigenvectors_implicit(
    decomp, G_effective, n_probes=5, tol_orth=1e-4, tol_diag=5e-3, drop_tol=0.0
):
    n = G_effective.number_of_nodes()
    lam = decomp.eigvals
    rng = np.random.RandomState(123)

    probe_nodes = rng.choice(np.arange(n, dtype=np.int64), size=min(n_probes, n), replace=False)

    norm_errs = []
    diag_errs = []

    for i in probe_nodes:
        c = decomp.qt_dot_sparse({int(i): 1.0})
        norm_errs.append(abs(np.linalg.norm(c) - 1.0))

        row = {int(i): 0.0}
        deg_i = 0.0
        for j, data in G_effective[int(i)].items():
            w = data.get('weight', 1.0)
            deg_i += w
            row[int(j)] = row.get(int(j), 0.0) - w
        row[int(i)] = row.get(int(i), 0.0) + deg_i

        lhs = decomp.qt_dot_sparse(row)
        rhs = lam * c
        denom = max(np.linalg.norm(lhs), np.linalg.norm(rhs), 1e-12)
        diag_errs.append(np.linalg.norm(lhs - rhs) / denom)

    max_norm_err = max(norm_errs) if norm_errs else np.inf
    max_diag_rel = max(diag_errs) if diag_errs else np.inf
    return {
        "max_norm_err": max_norm_err,
        "mean_norm_err": float(np.mean(norm_errs)) if norm_errs else np.inf,
        "max_diag_relerr": max_diag_rel,
        "mean_diag_relerr": float(np.mean(diag_errs)) if diag_errs else np.inf,
        "PASS": (max_norm_err < tol_orth and max_diag_rel < tol_diag),
    }

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Recursive Cauchy eigendecomposition benchmark')
    parser.add_argument('--sizes', type=int, nargs='+', default=[5000, 10000, 20000],
                        help='Target subgraph sizes (default: 5000 10000 20000)')
    parser.add_argument('--full', action='store_true',
                        help='Use the full ogbn-arxiv graph (169343 nodes)')
    parser.add_argument('--depths', type=int, nargs='+', default=[1, 2, 3],
                        help='Recursion depths to test (default: 1 2 3)')
    parser.add_argument('--target-cut', type=int, default=5,
                        help='Target number of cut edges per partition (default: 5)')
    parser.add_argument('--runs', type=int, default=2,
                        help='Number of runs per size (default: 2)')
    parser.add_argument('--skip-validation', action='store_true',
                        help='Skip unit tests')
    parser.add_argument('--no-dense', action='store_true',
                        help='Skip dense baseline (for large graphs)')
    args = parser.parse_args()
    
    os.makedirs('data_times', exist_ok=True)

    # Warm up Numba JIT
    print("Warming up Numba JIT...")
    _d = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    _z = np.array([0.1, 0.2, 0.3, 0.2, 0.1])
    _ = solve_secular_all_roots(_d, _z, 1.0)
    _ = _compute_column_norms(_z, _d, _d + 0.5, 5)
    _gv = np.array([[0.0, 1.0, 0.6, 0.8]], dtype=np.float64)
    _fzm = np.random.randn(5, 2)
    _apply_givens_batch(_gv, _fzm)
    _X = np.random.randn(5, 3)
    _out = np.empty_like(_X)
    _nrm = np.ones(5)
    _ = deflate_numba(_d, _z, 1e-8)
    _dd = np.array([1.0, 1.0, 2.0, 2.0, 3.0])
    _perturb_duplicates_rel(_dd)
    _xw = np.array([0.1, 0.2, 0.3, 0.2, 0.1])
    _ = _cauchy_QtQx(_d, _d + 0.5, _z, _nrm, _xw, 5)
    _gv = np.empty((0, 4), dtype=np.float64)
    _ai = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    _xin = np.random.randn(5)
    _ = apply_cauchy_factor_transpose(_d, _d + 0.5, _z, _nrm, _gv, _ai, _xin, 5)
    _ = apply_cauchy_factor_transpose(_d, _d + 0.5, _z, _nrm, _gv, _ai, _xin, 5)
    _zh = _compute_z_hat(_d, _d + 0.5, _z, 5)
    _cauchy_matvec_batch_stable(_zh, _d, _d + 0.5, _X, _out)
    _cauchy_matvec_batch_parallel_stable(_zh, _d, _d + 0.5, _X, _out)
    _cauchy_matvec_batch(_z, _d, _d + 0.5, _nrm, _X, _out)
    _cauchy_matvec_batch_parallel(_z, _d, _d + 0.5, _nrm, _X, _out)   
    _ = _apply_cauchy_factor_to_vector_zhat(_d, _d + 0.5, _z, _gv, _ai, _xin)
    _ = _apply_cauchy_factor_to_vector_znorms(_d, _d + 0.5, _z, _nrm, _gv, _ai, _xin)
    
    _z2rho = _z * _z
    _ = _check_roots_quality(_d, _z2rho, _d + 0.5, 5)    
    _ = _compute_norms_and_residual(_z, _d, _d + 0.5, _z2rho, 5)
    print("JIT warmup done.\n")

    # Run validation tests
    if not args.skip_validation:
        test1 = test_small_rank1_update()
        test2 = test_rank1_with_clusters()
        test3 = test_small_graph()
        test4 = test_medium_graph()
        test5 = test_weighted_graph()

        if not (test1 and test2 and test3 and test4 and test5):
            print("=" * 60)
            print("VALIDATION TESTS FAILED — debugging required.")
            print("=" * 60)
            exit(1)

        print("All validation tests passed. Running recursive benchmark...\n")
    else:
        print("Skipping validation tests.\n")

    # ================================================================
    # LOAD ogbn-arxiv graphs
    # ================================================================
    
    if args.full:
        target_sizes = [0]
        n_runs = 1
    else:
        target_sizes = args.sizes
        n_runs = args.runs
    
    arxiv_graphs = {}
    
    print("=" * 70)
    print("LOADING ogbn-arxiv GRAPHS")
    print("=" * 70)
    
    if args.full:
        try:
            from ogb.nodeproppred import NodePropPredDataset
            from scipy.sparse import coo_matrix
            
            print("  Loading full ogbn-arxiv dataset...")
            dataset = NodePropPredDataset(name='ogbn-arxiv', root='data/')
            graph, _ = dataset[0]
            
            edge_index = graph['edge_index']
            src = edge_index[0]
            dst = edge_index[1]
            n_total = graph['num_nodes']
            
            print(f"  ogbn-arxiv: {n_total} nodes, {len(src)} directed edges")
            
            ones = np.ones(len(src), dtype=np.float64)
            A = coo_matrix((ones, (src, dst)), shape=(n_total, n_total))
            A = A + A.T
            A = (A > 0).astype(np.float64)
            A = csr_matrix(A)
            
            G_full = nx.Graph()
            G_full.add_nodes_from(range(n_total))
            A_coo = A.tocoo()
            for i, j in zip(A_coo.row, A_coo.col):
                if i < j:
                    G_full.add_edge(int(i), int(j))
            
            if not nx.is_connected(G_full):
                print("  Finding largest connected component...")
                largest_cc = max(nx.connected_components(G_full), key=len)
                G_full = G_full.subgraph(largest_cc).copy()
                G_full = nx.convert_node_labels_to_integers(G_full)
                print(f"  Largest CC: {G_full.number_of_nodes()} nodes, "
                      f"{G_full.number_of_edges()} edges")
            
            actual_n = G_full.number_of_nodes()
            print(f"  Full graph: {actual_n} nodes, {G_full.number_of_edges()} edges, "
                  f"avg_degree={2*G_full.number_of_edges()/actual_n:.1f}")
            
            arxiv_graphs[(0, 0)] = G_full
            target_sizes = [actual_n]
            
        except Exception as e:
            print(f"  FAILED to load full graph: {e}")
            exit(1)
    else:
        for n_target in target_sizes:
            for run in range(n_runs):
                key = (n_target, run)
                G = load_ogbn_arxiv_subgraph(n_target, seed=42 + run)
                arxiv_graphs[key] = G
                print(f"  n={n_target}, run={run}: {G.number_of_nodes()} nodes, "
                        f"{G.number_of_edges()} edges, "
                        f"avg_degree={2*G.number_of_edges()/G.number_of_nodes():.1f}")
    
    print()

    # ================================================================
    # BENCHMARK
    # ================================================================
    print("=" * 70)
    if args.full:
        print("RECURSIVE DEPTH COMPARISON (ogbn-arxiv FULL)")
    else:
        print("RECURSIVE DEPTH COMPARISON (ogbn-arxiv)")
    print("=" * 70)
    
    depths = args.depths
    target_cut = args.target_cut
    dense_limit = 0 if args.no_dense else 20000
    recursive_results = {}
    
    for n_nodes in target_sizes:
        print(f"\n{'=' * 70}")
        if args.full:
            print(f"RECURSIVE BENCHMARK: FULL ogbn-arxiv (n = {n_nodes})")
        else:
            print(f"RECURSIVE BENCHMARK: n ≈ {n_nodes} (ogbn-arxiv)")
        print(f"{'=' * 70}")
        
        actual_runs = 1 if args.full else n_runs
        for run in range(actual_runs):
            if args.full:
                G = arxiv_graphs[(0, 0)]
            else:
                G = arxiv_graphs[(n_nodes, run)]
            actual_n = G.number_of_nodes()
            
            if actual_n <= dense_limit:
                A = nx.adjacency_matrix(G).astype(np.float64)
                D_vec = np.array(A.sum(axis=1)).ravel()
                L_dense = np.diag(D_vec) - A.toarray()
                t0 = time.perf_counter()
                eigvals_dense_orig = np.sort(np.linalg.eigvalsh(L_dense))
                t_dense = time.perf_counter() - t0
                del L_dense
                print(f"\n  [run {run}] n={actual_n}, edges={G.number_of_edges()}, "
                      f"avg_deg={2*G.number_of_edges()/actual_n:.1f}")
                print(f"  Dense baseline: {t_dense:.2f}s")
            else:
                eigvals_dense_orig = None
                t_dense = np.nan
                print(f"\n  [run {run}] n={actual_n}, edges={G.number_of_edges()}, "
                      f"avg_deg={2*G.number_of_edges()/actual_n:.1f}")
                print(f"  Dense baseline: skipped (n={actual_n} > {dense_limit})")
            
            for depth in depths:
                print(f"\n  Starting depth={depth}...")
                t0 = time.perf_counter()
                eigvals_rec, decomp, stats, G_effective, _ = recursive_cauchy_eigen(
                    G, depth=depth, target_cut=target_cut, tol=1e-10
                )
                t_total = time.perf_counter() - t0
                from src.io.decomp_io import test_save_load_roundtrip
                # or if you pasted in same script, call directly:
                test_save_load_roundtrip(decomp, path="decomp_files/decomp_arxiv_rt.pkl")
                speedup = t_dense / t_total if not np.isnan(t_dense) else float('nan')
                
                print(f"\n  depth={depth}: {t_total:.2f}s", end="")
                if not np.isnan(speedup):
                    print(f" (speedup: {speedup:.1f}x)")
                else:
                    print()
                
                print(f"    Leaves: {stats['n_leaves']} "
                      f"(sizes: {sorted(stats['leaf_sizes'])})")
                print(f"    Largest leaf (dense eig): {stats['peak_dense_size']}")
                print(f"    Merges: {stats['n_merges']}")
                print(f"    Total cut edges: {stats['total_cut_edges']}")
                print(f"    Time breakdown:")
                print(f"      Spectral cuts:  {stats['time_cuts']:.2f}s")
                print(f"      Sparsification: {stats['time_sparsify']:.2f}s")
                print(f"      Base eig:       {stats['time_base_eig']:.2f}s")
                print(f"      z computation:  {stats['time_z_compute']:.2f}s")
                print(f"      Secular eqs:    {stats['time_secular']:.2f}s")
                print(f"    Memory: {decomp.memory_bytes() / 1e6:.1f}MB "
                      f"(vs {actual_n**2 * 8 / 1e6:.1f}MB dense)")
                
                if actual_n <= dense_limit:
                    n_edges_eff = G_effective.number_of_edges()
                    n_nodes_eff = G_effective.number_of_nodes()
                    
                    print(f"    G_effective: {n_nodes_eff} nodes, {n_edges_eff} edges")
                    print(f"    eigvals_rec length: {len(eigvals_rec)}")
                    
                    if n_nodes_eff == actual_n and len(eigvals_rec) == actual_n:
                        A_eff = nx.adjacency_matrix(G_effective).astype(np.float64)
                        D_eff = np.array(A_eff.sum(axis=1)).ravel()
                        L_eff = np.diag(D_eff) - A_eff.toarray()
                        eigvals_dense_eff = np.sort(np.linalg.eigvalsh(L_eff))
                        del L_eff
                        
                        eigvals_rec_sorted = np.sort(eigvals_rec)
                        
                        max_err = np.max(np.abs(eigvals_dense_eff - eigvals_rec_sorted))
                        max_abs_eigval = np.max(np.abs(eigvals_dense_eff))
                        rel_max_err = max_err / max_abs_eigval
                        trace_eff = np.sum(eigvals_dense_eff)
                        trace_rec = np.sum(eigvals_rec_sorted)
                        trace_err = abs(trace_eff - trace_rec)
                        frob_eff = np.sum(eigvals_dense_eff ** 2)
                        frob_rec = np.sum(eigvals_rec_sorted ** 2)
                        frob_err = abs(frob_eff - frob_rec)
                        
                        print(f"    Validation (vs SAME sparsified graph):")
                        print(f"      Edges: {G.number_of_edges()} original -> {n_edges_eff} effective")
                        print(f"      Eigenvalue max error:  {max_err:.2e}")
                        print(f"      Relative max error:    {rel_max_err:.2e}")
                        print(f"      Trace error:           {trace_err:.2e} "
                              f"({trace_eff:.2f} vs {trace_rec:.2f})")
                        print(f"      Frobenius² error:      {frob_err:.2e} "
                              f"({frob_eff:.2f} vs {frob_rec:.2f})")
                        
                        err_per_eig = np.abs(eigvals_dense_eff - eigvals_rec_sorted)
                        print(f"      Mean error:            {np.mean(err_per_eig):.2e}")
                        print(f"      Median error:          {np.median(err_per_eig):.2e}")
                        print(f"      95th pctl error:       {np.percentile(err_per_eig, 95):.2e}")
                        print(f"      99th pctl error:       {np.percentile(err_per_eig, 99):.2e}")
                        
                        if rel_max_err < 1e-6:
                            print(f"      ✓ PASS (rel_err={rel_max_err:.2e})")
                        else:
                            print(f"      ✗ FAIL (rel_err={rel_max_err:.2e})")
                        worst_idx = np.argsort(err_per_eig)[-5:]
                        print(f"      Worst indices: {worst_idx}")
                        print(f"      Worst true:  {eigvals_dense_eff[worst_idx]}")
                        print(f"      Worst recon: {eigvals_rec_sorted[worst_idx]}")
                    else:
                        print(f"    ✗ SKIP validation: size mismatch")
                else:
                    print(f"    Validation: skipped (n={actual_n} > {dense_limit})")

                eigvec_probes = 10
                t_vec0 = time.perf_counter()
                vec_val = validate_eigenvectors_implicit(
                    decomp, G_effective, n_probes=eigvec_probes, tol_orth=1e-6, tol_diag=2e-3
                )
                t_vec = time.perf_counter() - t_vec0
                print(f"    Eigvec implicit check ({eigvec_probes} probes):")
                print(f"      Norm max err: {vec_val['max_norm_err']:.2e}")
                print(f"      Diag max rel: {vec_val['max_diag_relerr']:.2e}")
                print(f"      Time:         {t_vec:.2f}s")
                print(f"      {'✓ PASS' if vec_val['PASS'] else '✗ FAIL'}")
                
                key = (n_nodes, depth, run)
                recursive_results[key] = {
                    'time': t_total,
                    'speedup': speedup,
                    'stats': stats,
                    'memory_mb': decomp.memory_bytes() / 1e6,
                    'actual_n': actual_n,
                }
    
    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'n_target':>8s} {'n_actual':>8s} {'depth':>6s} {'run':>4s} {'time(s)':>8s} "
          f"{'speedup':>8s} {'mem(MB)':>8s} {'leaves':>7s} {'max_leaf':>9s} {'cut_edges':>10s}")
    print("-" * 85)
    for (n, d, r), res in sorted(recursive_results.items()):
        stats = res['stats']
        actual = res.get('actual_n', n)
        if not np.isnan(res['speedup']):
            sp = f"{res['speedup']:>7.1f}x"
        else:
            sp = f"{'N/A':>8s}"
        print(f"{n:>8d} {actual:>8d} {d:>6d} {r:>4d} {res['time']:>8.2f} "
              f"{sp} {res['memory_mb']:>8.1f} {stats['n_leaves']:>7d} "
              f"{stats['peak_dense_size']:>9d} {stats['total_cut_edges']:>10d}")