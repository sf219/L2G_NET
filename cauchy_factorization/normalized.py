"""
recursive_decomp_normalized.py
==============================
Recursive Cauchy eigendecomposition for the *normalized* graph Laplacian

    L_norm = D_eff^{-1/2} L_eff D_eff^{-1/2}

where D_eff is the degree matrix of the FULLY SPARSIFIED effective graph —
i.e., the graph whose Laplacian we are actually decomposing.

Why this matters
----------------
At every recursion level we drop some cut edges (sparsification).  The matrix
we end up factoring is the Laplacian of the effective graph G_eff, not of the
original G.  The normalized Laplacian of G_eff is

    L_norm = D_eff^{-1/2} L_eff D_eff^{-1/2}

whose diagonal entry d_i^{eff} = sum of weights of edges incident to i in
G_eff.  Using degrees from any other graph (e.g. the pre-sparsification G)
would normalize by the wrong matrix, breaking the algebraic decomposition.

Strategy
--------
We do a lightweight "dry-run" pass first:
    build_effective_graph(G, depth, target_cut)
which mirrors the recursion structure exactly (same METIS cuts, same
sparsification choices, same random seeds) but only builds the NetworkX
graph — no eigenvalue computation.  From that we compute global_deg_eff
once and pass it into the actual eigendecomposition.

Key changes vs. the combinatorial version
------------------------------------------
1.  build_effective_graph — dry run that returns G_eff.

2.  global_deg : ndarray (n,) — weighted degrees in G_eff, computed once
    from the dry-run result and passed down every recursive call unchanged.

3.  Subgraph base matrices:

        L̃_S[i,j] = L_eff,S[i,j] / sqrt(d_i^{eff} * d_j^{eff})

    where L_eff,S is the block of L_eff restricted to partition S
    (intra-partition sparsified edges only).  Scaled by EFFECTIVE global
    degrees, so cut-edge contributions to D_eff are already absorbed in the
    diagonal.

4.  z-vectors for kept cut edge (u,v,w):

        z = [Q_0|Q_1]^T  D_eff^{-1/2} (e_u - e_v)
          = (d_u^{eff})^{-1/2} (Q^T e_u) - (d_v^{eff})^{-1/2} (Q^T e_v)

    Children return plain eigvec rows; the parent applies d_eff^{-1/2}.

5.  Validation builds L_norm from G_eff and global_deg_eff consistently.

Everything else (deflation, secular equation, Cauchy factors, qt_dot,
save/load) is IDENTICAL to the combinatorial version.

----------------------------------------------------------------------------
SPIELMAN–SRIVASTAVA LOEWNER-ORDER BOOTSTRAP (diagnostic add-on)
----------------------------------------------------------------------------
Optional per-level measurement (loewner_check=True).  At each recursion step,
before recursing, we compare the level's COMBINATORIAL Laplacian before vs
after the cut sparsification and estimate the constants in the two-sided
Loewner sandwich

    (1 - eps_minus) L_orig  ⪯  L_sparse  ⪯  (1 + eps_plus) L_orig

via randomized generalized power iteration with n_probes random restarts (the
"bootstrap").  Two flavours:

    loewner_kind='comb' : the combinatorial bound above.
    loewner_kind='norm' : each side normalized by ITS OWN degree
                          (Do^{-1/2} L_o Do^{-1/2} vs Ds^{-1/2} L_s Ds^{-1/2}).
                          Different kernels => genuine normalized perturbation,
                          NOT a congruence of the combinatorial bound.

This is measurement only — it reports the estimated bounds and draws no
conclusion about adequacy.  It is OFF by default and does not touch the
decomposition path or benchmark timings when disabled.
"""

import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix, diags, eye as speye, coo_matrix
from scipy.sparse.linalg import eigsh, cg
import numba as nb
import time
import os
from ._metis_compat import pymetis

from .recursive_decomp import (
    RecursiveDecomp,
    _apply_givens_batch,
    _cauchy_matvec_batch,
    _cauchy_matvec_batch_parallel,
    _cauchy_matvec_batch_stable,
    _cauchy_matvec_batch_parallel_stable,
    _apply_cauchy_factor_to_vector_znorms,
    _apply_cauchy_factor_to_vector_zhat,
)
from .cauchy_factor import CauchyFactor

def _import_cauchy():
    """
    Lazy import from testing_fast_cauchy_factor.
    Called once at runtime to avoid the circular import that occurs when
    testing_fast_cauchy_factor imports from itself at module load time.
    """
    from . import factorization as _tfc
    return _tfc

# Symbols populated on first use by _ensure_imports()
solve_secular_all_roots    = None
deflate_numba              = None
_perturb_duplicates_rel    = None
_compute_column_norms      = None
_compute_z_hat             = None
_compute_z_hat_checked     = None
_check_roots_quality       = None
_compute_norms_and_residual = None
_cauchy_QtQx               = None
_cauchy_QtQx_znorms        = None
_factor_orth_probe_err     = None
validate_cauchy_orthogonality = None
cut_sparse                 = None
get_cut_edges              = None
sparsify_cut_edges         = None
sparse_laplacian           = None
subgraph_dense_laplacian   = None
sparsify_cut_edges_ss         = None

def _ensure_imports():
    """Populate module-level names from testing_fast_cauchy_factor on first call."""
    global solve_secular_all_roots, deflate_numba, _perturb_duplicates_rel
    global _compute_column_norms, _compute_z_hat, _compute_z_hat_checked
    global _check_roots_quality, _compute_norms_and_residual
    global _cauchy_QtQx, _cauchy_QtQx_znorms, _factor_orth_probe_err
    global validate_cauchy_orthogonality, cut_sparse, get_cut_edges
    global sparsify_cut_edges, sparse_laplacian, subgraph_dense_laplacian, sparsify_cut_edges_ss

    if solve_secular_all_roots is not None:
        return  # already done

    m = _import_cauchy()
    solve_secular_all_roots       = m.solve_secular_all_roots
    deflate_numba                 = m.deflate_numba
    _perturb_duplicates_rel       = m._perturb_duplicates_rel
    _compute_column_norms         = m._compute_column_norms
    _compute_z_hat                = m._compute_z_hat
    _compute_z_hat_checked        = m._compute_z_hat_checked
    _check_roots_quality          = m._check_roots_quality
    _compute_norms_and_residual   = m._compute_norms_and_residual
    _cauchy_QtQx                  = m._cauchy_QtQx
    _cauchy_QtQx_znorms           = m._cauchy_QtQx_znorms
    _factor_orth_probe_err        = m._factor_orth_probe_err
    validate_cauchy_orthogonality = m.validate_cauchy_orthogonality
    cut_sparse                    = m.cut_sparse
    get_cut_edges                 = m.get_cut_edges
    sparsify_cut_edges            = m.sparsify_cut_edges
    sparsify_cut_edges_ss         = m.sparsify_cut_edges_ss
    sparse_laplacian              = m.sparse_laplacian
    subgraph_dense_laplacian      = m.subgraph_dense_laplacian


# ============================================================================
# DRY-RUN: build the fully sparsified effective graph
# ============================================================================

def build_effective_graph(G, depth, target_cut):
    _ensure_imports()
    """
    Mirror the recursion of recursive_cauchy_eigen_normalized exactly
    (same METIS calls, same sparsify_cut_edges calls, same seeds) but
    perform NO eigenvalue computation.

    Returns G_eff : NetworkX graph with the same node set as G and edge
    weights equal to those that survive all levels of sparsification.
    """
    n = G.number_of_nodes()

    if depth <= 0 or n <= 100:
        return G.copy()

    part_0, part_1, _ = cut_sparse(G)
    part_0_set = set(int(p) for p in part_0)
    part_1_set = set(int(p) for p in part_1)

    G_sparse, cut_edges_weighted = sparsify_cut_edges_ss(
        G, part_0_set, part_1_set, target_cut=target_cut, seed=42
    )

    G_sub0 = G_sparse.subgraph(part_0).copy()
    G_sub1 = G_sparse.subgraph(part_1).copy()

    G_eff_0 = build_effective_graph(G_sub0, depth - 1, target_cut)
    G_eff_1 = build_effective_graph(G_sub1, depth - 1, target_cut)

    G_eff = nx.Graph()
    G_eff.add_nodes_from(G.nodes())
    for u, v, data in G_eff_0.edges(data=True):
        G_eff.add_edge(u, v, **data)
    for u, v, data in G_eff_1.edges(data=True):
        G_eff.add_edge(u, v, **data)
    for u, v, w in cut_edges_weighted:
        if G_eff.has_edge(u, v):
            G_eff[u][v]['weight'] = G_eff[u][v].get('weight', 0.0) + w
        else:
            G_eff.add_edge(u, v, weight=w)

    return G_eff


# ============================================================================
# GLOBAL-DEGREE HELPERS
# ============================================================================

def compute_global_degrees(G, n_nodes=None):
    """
    Weighted degree vector from G_eff.  Every node's degree reflects only
    the edges that survive all levels of sparsification — matching D_eff
    exactly.
    """
    if n_nodes is None:
        n_nodes = max(G.nodes()) + 1 if G.number_of_nodes() > 0 else 0
    deg = np.zeros(n_nodes, dtype=np.float64)
    for u, v, data in G.edges(data=True):
        w = data.get('weight', 1.0)
        deg[u] += w
        deg[v] += w
    deg = np.where(deg < 1e-300, 1.0, deg)   # guard isolated nodes
    return deg


def normalized_subgraph_laplacian(G_sub, nodes, global_deg):
    """
    Dense block of L_norm = D_eff^{-1/2} L_eff D_eff^{-1/2} restricted to
    `nodes` (intra-partition edges of the sparsified subgraph only).

        M[i,j] = L_sub[i,j] / sqrt(global_deg[u] * global_deg[v])

    Returns (M, node_map).
    """
    nodes = list(nodes)
    node_map = {nd: i for i, nd in enumerate(nodes)}
    k = len(nodes)
    M = np.zeros((k, k), dtype=np.float64)

    H = G_sub.subgraph(nodes)
    for u, v, data in H.edges(data=True):
        w = data.get('weight', 1.0)
        i, j = node_map[u], node_map[v]
        du, dv = global_deg[u], global_deg[v]
        off = w / np.sqrt(du * dv)
        M[i, j] -= off
        M[j, i] -= off
        M[i, i] += w / du
        M[j, j] += w / dv

    return M, node_map


def normalized_full_laplacian_dense(G, global_deg):
    """Dense D_eff^{-1/2} L_eff D_eff^{-1/2}."""
    nodes = sorted(G.nodes())
    idx = {nd: i for i, nd in enumerate(nodes)}
    n = len(nodes)
    M = np.zeros((n, n), dtype=np.float64)
    for u, v, data in G.edges(data=True):
        w = data.get('weight', 1.0)
        i, j = idx[u], idx[v]
        du, dv = global_deg[u], global_deg[v]
        off = w / np.sqrt(du * dv)
        M[i, j] -= off; M[j, i] -= off
        M[i, i] += w / du; M[j, j] += w / dv
    return M


def normalized_full_laplacian_sparse(G, global_deg):
    """Sparse D_eff^{-1/2} L_eff D_eff^{-1/2}."""
    nodes = sorted(G.nodes())
    idx = {nd: i for i, nd in enumerate(nodes)}
    n = len(nodes)
    rows, cols, vals = [], [], []
    for u, v, data in G.edges(data=True):
        w = data.get('weight', 1.0)
        i_u, i_v = idx[u], idx[v]
        du, dv = global_deg[u], global_deg[v]
        off = w / np.sqrt(du * dv)
        rows += [i_u, i_v, i_u, i_v]
        cols += [i_v, i_u, i_u, i_v]
        vals += [-off, -off, w / du, w / dv]
    return coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()


# ============================================================================
# SPIELMAN–SRIVASTAVA LOEWNER-ORDER BOOTSTRAP (diagnostic only)
# ============================================================================

def _cg_solve(A, b, rtol=1e-6, maxiter=500):
    """scipy version-agnostic CG (rtol vs tol kw)."""
    try:
        z, _ = cg(A, b, rtol=rtol, maxiter=maxiter)
    except TypeError:
        z, _ = cg(A, b, tol=rtol, maxiter=maxiter)
    return z


def _level_laplacian(G, nodes, node_map):
    """Sparse COMBINATORIAL Laplacian of G on the given local node ordering."""
    rows, cols, vals = [], [], []
    diag = np.zeros(len(nodes))
    for u, v, data in G.edges(data=True):
        w = data.get('weight', 1.0)
        i, j = node_map[u], node_map[v]
        rows += [i, j]; cols += [j, i]; vals += [-w, -w]
        diag[i] += w; diag[j] += w
    k = len(nodes)
    rows += list(range(k)); cols += list(range(k)); vals += list(diag)
    return coo_matrix((vals, (rows, cols)), shape=(k, k)).tocsr()


def loewner_bootstrap(L_orig, L_sparse, n_probes=8, n_iter=30,
                      reg=1e-8, maxiter=500, seed=0):
    """
    Randomized estimate of the generalized spectrum of (L_sparse, L_orig) on
    range(L_orig), i.e. the constants in

        (1 - eps_minus) L_orig  ⪯  L_sparse  ⪯  (1 + eps_plus) L_orig.

    lam_max = max_x (xᵀL_s x)/(xᵀL_o x)  via power iter on  L_orig⁺ L_sparse
    lam_min = min_x (xᵀL_s x)/(xᵀL_o x)  via power iter on  L_sparse⁺ L_orig

    "Bootstrap": n_probes random restarts; lam_max/lam_min are the extreme
    over restarts, mean/std also reported. Diagnostic only — reports the
    estimated bounds, draws no conclusion about adequacy.
    """
    n = L_orig.shape[0]
    rng = np.random.RandomState(seed)
    A_o = (L_orig + reg * speye(n)).tocsc()
    A_s = (L_sparse + reg * speye(n)).tocsc()

    def deflate(x):
        return x - x.mean()                      # remove single constant mode

    def gen_rayleigh(x):
        xo = float(x @ (L_orig @ x))
        xs = float(x @ (L_sparse @ x))
        return xs / xo if xo > 1e-300 else np.nan

    lam_max_trials, lam_min_trials = [], []
    for _ in range(n_probes):
        # lam_max:  x <- L_orig^+ (L_sparse x)
        x = deflate(rng.randn(n)); x /= np.linalg.norm(x) + 1e-300
        for _ in range(n_iter):
            z = deflate(_cg_solve(A_o, L_sparse @ x, maxiter=maxiter))
            nz = np.linalg.norm(z)
            if nz < 1e-300:
                break
            x = z / nz
        lam_max_trials.append(gen_rayleigh(x))

        # lam_min:  x <- L_sparse^+ (L_orig x)
        x = deflate(rng.randn(n)); x /= np.linalg.norm(x) + 1e-300
        for _ in range(n_iter):
            z = deflate(_cg_solve(A_s, L_orig @ x, maxiter=maxiter))
            nz = np.linalg.norm(z)
            if nz < 1e-300:
                break
            x = z / nz
        lam_min_trials.append(gen_rayleigh(x))

    mx = np.array([v for v in lam_max_trials if np.isfinite(v)])
    mn = np.array([v for v in lam_min_trials if np.isfinite(v)])
    lam_max = float(np.max(mx)) if mx.size else np.nan
    lam_min = float(np.min(mn)) if mn.size else np.nan

    return {
        'lam_max': lam_max, 'lam_min': lam_min,
        'eps_plus': lam_max - 1.0, 'eps_minus': 1.0 - lam_min,
        'kappa': lam_max / lam_min if lam_min > 0 else np.inf,
        'lam_max_mean': float(np.mean(mx)) if mx.size else np.nan,
        'lam_max_std':  float(np.std(mx))  if mx.size else np.nan,
        'lam_min_mean': float(np.mean(mn)) if mn.size else np.nan,
        'lam_min_std':  float(np.std(mn))  if mn.size else np.nan,
        'n_probes': int(n_probes),
    }


def loewner_bootstrap_norm(L_o, L_s, n_probes=6, n_iter=20, reg=1e-8,
                           maxiter=500, seed=0):
    """
    Normalized-Laplacian Loewner bootstrap. L_o, L_s are the COMBINATORIAL
    level Laplacians (same ordering). Each is normalized by its OWN degree:

        Ln_o = Do^{-1/2} L_o Do^{-1/2},   Ln_s = Ds^{-1/2} L_s Ds^{-1/2}

    Estimates the constants in
        (1 - eps_minus) Ln_o  ⪯  Ln_s  ⪯  (1 + eps_plus) Ln_o
    on range(Ln_o) (upper) / range(Ln_s) (lower) via randomized generalized
    power iteration, n_probes restarts.

    Ln_o and Ln_s have DIFFERENT kernels (Do^{1/2}1 vs Ds^{1/2}1), so this is
    the genuine normalized-operator perturbation — NOT a congruence of the
    combinatorial bound. A single shared D would reproduce the combinatorial
    numbers exactly. Diagnostic only; reports the estimate, draws no conclusion.
    """
    n = L_o.shape[0]
    rng = np.random.RandomState(seed)

    d_o = np.asarray(L_o.diagonal(), dtype=np.float64)
    d_s = np.asarray(L_s.diagonal(), dtype=np.float64)
    iso_o, iso_s = d_o <= 1e-300, d_s <= 1e-300
    Dih_o = diags(1.0 / np.sqrt(np.where(iso_o, 1.0, d_o)))
    Dih_s = diags(1.0 / np.sqrt(np.where(iso_s, 1.0, d_s)))
    Ln_o = (Dih_o @ L_o @ Dih_o).tocsr()
    Ln_s = (Dih_s @ L_s @ Dih_s).tocsr()

    # kernels = D^{1/2} 1  (zeroed on isolated nodes)
    null_o = np.sqrt(np.where(iso_o, 0.0, d_o)); null_o /= np.linalg.norm(null_o) + 1e-300
    null_s = np.sqrt(np.where(iso_s, 0.0, d_s)); null_s /= np.linalg.norm(null_s) + 1e-300

    A_o = (Ln_o + reg * speye(n)).tocsc()
    A_s = (Ln_s + reg * speye(n)).tocsc()

    def proj(x, u):
        return x - u * float(u @ x)

    def rayleigh(x):
        xo = float(x @ (Ln_o @ x))
        return (float(x @ (Ln_s @ x)) / xo) if xo > 1e-300 else np.nan

    def power(A, M_num, null):           # max gen-eigval of (A+reg)^+ M_num
        x = proj(rng.randn(n), null); x /= np.linalg.norm(x) + 1e-300
        for _ in range(n_iter):
            rhs = proj(M_num @ x, null)                       # keep RHS ⊥ null
            z = proj(_cg_solve(A, rhs, maxiter=maxiter), null)
            nz = np.linalg.norm(z)
            if nz < 1e-300:
                break
            x = z / nz
        return rayleigh(x)

    lam_max_trials, lam_min_trials = [], []
    for _ in range(n_probes):
        lam_max_trials.append(power(A_o, Ln_s, null_o))       # (Ln_o)^+ Ln_s
        # power on (Ln_s)^+ Ln_o lands at the vector with largest 1/mu,
        # i.e. smallest ratio mu = Ln_s/Ln_o -> already the min ratio.
        lam_min_trials.append(power(A_s, Ln_o, null_s))

    mx = np.array([v for v in lam_max_trials if np.isfinite(v)])
    mn = np.array([v for v in lam_min_trials if np.isfinite(v)])
    lam_max = float(np.max(mx)) if mx.size else np.nan
    lam_min = float(np.min(mn)) if mn.size else np.nan
    return {
        'lam_max': lam_max, 'lam_min': lam_min,
        'eps_plus': lam_max - 1.0, 'eps_minus': 1.0 - lam_min,
        'kappa': lam_max / lam_min if lam_min > 0 else np.inf,
        'lam_max_mean': float(np.mean(mx)) if mx.size else np.nan,
        'lam_max_std':  float(np.std(mx))  if mx.size else np.nan,
        'lam_min_mean': float(np.mean(mn)) if mn.size else np.nan,
        'lam_min_std':  float(np.std(mn))  if mn.size else np.nan,
        'n_probes': int(n_probes),
    }


# ============================================================================
# RECURSIVE NORMALIZED EIGENDECOMPOSITION
# ============================================================================

def recursive_cauchy_eigen_normalized(
    G,
    global_deg,
    depth=1,
    target_cut=5,
    tol=1e-10,
    _current_depth=0,
    _stats=None,
    _parent_needs_nodes=None,
    loewner_check=False,
    loewner_kind='norm',
    loewner_max_n=None,
    loewner_kwargs=None,
):
    """
    Recursively compute eigendecomposition of L_norm = D_eff^{-1/2} L_eff D_eff^{-1/2}.

    Call via eigen_normalized() at the top level — it handles the dry run
    and global_deg computation automatically.

    Parameters
    ----------
    G          : NetworkX graph at this recursion level.  At the top level
                 this must be G_eff (the fully sparsified graph) so the
                 sparsification choices here exactly replicate the dry run.
    global_deg : ndarray — weighted degrees from the top-level G_eff.
                 Passed unchanged to all recursive calls.
    loewner_check : bool — if True, run the Spielman–Srivastava Loewner-order
                 bootstrap at each recursive (non-leaf) level, comparing the
                 level's combinatorial Laplacian before vs after the cut
                 sparsification. Results land in _stats['loewner'].
    loewner_kind : 'comb' or 'norm' — combinatorial bound, or each side
                 normalized by its own degree.
    loewner_max_n : int or None — only measure levels with n <= this (skip
                 the expensive top levels). None measures every level.
    loewner_kwargs : dict — extra kwargs forwarded to the bootstrap
                 (n_probes, n_iter, reg, maxiter, seed).
    """
    if _stats is None:
        _stats = {
            'n_leaves': 0, 'leaf_sizes': [], 'n_merges': 0,
            'merge_sizes': [], 'time_cuts': 0.0, 'time_sparsify': 0.0,
            'time_base_eig': 0.0, 'time_secular': 0.0,
            'time_z_compute': 0.0, 'total_cut_edges': 0,
            'depth': depth, 'peak_dense_size': 0,
        }

    n = G.number_of_nodes()
    nodes = sorted(G.nodes())
    decomp = RecursiveDecomp()
    decomp.n = n
    decomp.global_to_local = {nd: i for i, nd in enumerate(nodes)}

    # ------------------------------------------------------------------ leaf
    if depth <= 0 or n <= 100:
        nodes_arr = np.array(nodes)
        M, node_map = normalized_subgraph_laplacian(G, nodes_arr, global_deg)

        t0 = time.perf_counter()
        eigvals, eigvecs = np.linalg.eigh(M)
        _stats['time_base_eig'] += time.perf_counter() - t0
        _stats['n_leaves'] += 1
        _stats['leaf_sizes'].append(n)
        _stats['peak_dense_size'] = max(_stats['peak_dense_size'], n)

        decomp.eigvals = eigvals
        decomp.is_leaf = True
        decomp.leaf_eigvecs = eigvecs
        decomp.leaf_nodes = nodes_arr
        decomp.leaf_node_map = node_map

        # Return PLAIN eigvec rows; parent applies d_eff^{-1/2} when building z
        precomputed = {}
        if _parent_needs_nodes is not None:
            for nd in _parent_needs_nodes:
                if nd in node_map:
                    precomputed[nd] = eigvecs[node_map[nd], :].copy()

        return eigvals, decomp, _stats, G.copy(), precomputed

    # ---------------------------------------------------------- recursive case

    # 1. Bisect
    t0 = time.perf_counter()
    part_0, part_1, _ = cut_sparse(G)
    _stats['time_cuts'] += time.perf_counter() - t0

    part_0_set = set(int(p) for p in part_0)
    part_1_set = set(int(p) for p in part_1)

    # 2. Sparsify — identical call (seed=42) as in build_effective_graph,
    #    so G_sparse here is exactly the subgraph of G_eff at this level.
    t0 = time.perf_counter()
    G_sparse, cut_edges_weighted = sparsify_cut_edges_ss(
        G, part_0_set, part_1_set, target_cut=target_cut, seed=42
    )
    _stats['time_sparsify'] += time.perf_counter() - t0
    _stats['total_cut_edges'] += len(cut_edges_weighted)

    # 2.5 Spielman–Srivastava Loewner-order bootstrap (diagnostic only)
    if loewner_check and (loewner_max_n is None or n <= loewner_max_n):
        t0_lw = time.perf_counter()
        L_o = _level_laplacian(G,        nodes, decomp.global_to_local)
        L_s = _level_laplacian(G_sparse, nodes, decomp.global_to_local)
        if loewner_kind == 'norm':
            lw = loewner_bootstrap_norm(L_o, L_s, **(loewner_kwargs or {}))
        else:
            lw = loewner_bootstrap(L_o, L_s, **(loewner_kwargs or {}))
        lw.update(kind=loewner_kind, level=_current_depth, n=n,
                  edges_dropped=G.number_of_edges() - G_sparse.number_of_edges())
        _stats.setdefault('loewner', []).append(lw)
        _stats['time_loewner'] = _stats.get('time_loewner', 0.0) + (time.perf_counter() - t0_lw)
        print(f"    [loewner-{loewner_kind} L{_current_depth} n={n}] "
              f"lam in [{lw['lam_min']:.4f}, {lw['lam_max']:.4f}]  "
              f"eps_- {lw['eps_minus']:.3e}  eps_+ {lw['eps_plus']:.3e}  "
              f"kappa {lw['kappa']:.4f}")

    cut_edges   = [(u, v) for u, v, w in cut_edges_weighted]
    cut_weights = [w       for u, v, w in cut_edges_weighted]

    # Which nodes must children precompute?
    child_needs_0, child_needs_1 = set(), set()
    for u, v in cut_edges:
        for nd in (int(u), int(v)):
            (child_needs_0 if nd in part_0_set else child_needs_1).add(nd)
    if _parent_needs_nodes:
        for nd in _parent_needs_nodes:
            (child_needs_0 if nd in part_0_set else child_needs_1).add(nd)

    # 3. Recurse on sparsified subgraphs
    G_sub0 = G_sparse.subgraph(part_0).copy()
    G_sub1 = G_sparse.subgraph(part_1).copy()

    eigvals_0, decomp_0, _, G_eff_0, precomp_0 = recursive_cauchy_eigen_normalized(
        G_sub0, global_deg, depth=depth - 1, target_cut=target_cut, tol=tol,
        _current_depth=_current_depth + 1, _stats=_stats,
        _parent_needs_nodes=child_needs_0,
        loewner_check=loewner_check, loewner_kind=loewner_kind,
        loewner_max_n=loewner_max_n, loewner_kwargs=loewner_kwargs,
    )
    eigvals_1, decomp_1, _, G_eff_1, precomp_1 = recursive_cauchy_eigen_normalized(
        G_sub1, global_deg, depth=depth - 1, target_cut=target_cut, tol=tol,
        _current_depth=_current_depth + 1, _stats=_stats,
        _parent_needs_nodes=child_needs_1,
        loewner_check=loewner_check, loewner_kind=loewner_kind,
        loewner_max_n=loewner_max_n, loewner_kwargs=loewner_kwargs,
    )

    # Bookkeeping graph
    G_effective = nx.Graph()
    G_effective.add_nodes_from(G.nodes())
    for u, v, data in G_eff_0.edges(data=True):
        G_effective.add_edge(u, v, **data)
    for u, v, data in G_eff_1.edges(data=True):
        G_effective.add_edge(u, v, **data)
    for (u, v), wt in zip(cut_edges, cut_weights):
        if G_effective.has_edge(u, v):
            G_effective[u][v]['weight'] += wt
        else:
            G_effective.add_edge(u, v, weight=wt)

    # 4. Build z-vectors
    #
    #   z = [Q_0|Q_1]^T D_eff^{-1/2} (e_u - e_v)
    #     =  (d_u^eff)^{-1/2} (Q^T e_u) - (d_v^eff)^{-1/2} (Q^T e_v)
    #
    # precomp_X[nd] = plain Q^T e_nd row (no D scaling); we apply it here.

    n0, n1 = len(part_0), len(part_1)
    n_total = n0 + n1

    eigvals_combined = np.concatenate([eigvals_0, eigvals_1])
    sort_perm = np.argsort(eigvals_combined, kind='mergesort')
    eigvals_sorted = eigvals_combined[sort_perm]

    t0 = time.perf_counter()
    all_z_vectors = []
    for u_nd, v_nd in cut_edges:
        u_int, v_int = int(u_nd), int(v_nd)
        c = np.zeros(n_total, dtype=np.float64)

        inv_sqrt_du = 1.0 / np.sqrt(global_deg[u_int])
        if u_int in part_0_set:
            row = precomp_0.get(u_int)
            if row is not None:
                c[:n0] += inv_sqrt_du * row
        else:
            row = precomp_1.get(u_int)
            if row is not None:
                c[n0:] += inv_sqrt_du * row

        inv_sqrt_dv = 1.0 / np.sqrt(global_deg[v_int])
        if v_int in part_0_set:
            row = precomp_0.get(v_int)
            if row is not None:
                c[:n0] -= inv_sqrt_dv * row
        else:
            row = precomp_1.get(v_int)
            if row is not None:
                c[n0:] -= inv_sqrt_dv * row

        all_z_vectors.append(c[sort_perm])

    # Sort updates weakest-first for numerical stability
    if len(cut_edges) > 1:
        alpha = np.array([
            cut_weights[i] * np.dot(all_z_vectors[i], all_z_vectors[i])
            for i in range(len(cut_edges))
        ])
        ord_idx = np.argsort(alpha)
        cut_edges     = [cut_edges[i]     for i in ord_idx]
        cut_weights   = [cut_weights[i]   for i in ord_idx]
        all_z_vectors = [all_z_vectors[i] for i in ord_idx]

    _stats['time_z_compute'] += time.perf_counter() - t0

    # 5. Sequential rank-1 secular updates — identical to combinatorial version
    t0 = time.perf_counter()
    current_eigvals = eigvals_sorted.copy()
    cauchy_factors  = []
    resort_perms    = []

    for edge_idx in range(len(cut_edges)):
        z   = all_z_vectors[edge_idx]
        rho = cut_weights[edge_idx]

        deflated, z_def, givens_data = deflate_numba(current_eigvals, z, tol)
        active_idx = np.where(~deflated)[0]

        givens_list = [
            (int(givens_data[g, 0]), int(givens_data[g, 1]),
             givens_data[g, 2],      givens_data[g, 3])
            for g in range(givens_data.shape[0])
        ]

        n_future = len(cut_edges) - edge_idx - 1
        if givens_data.shape[0] > 0 and n_future > 0:
            future_z_mat = np.column_stack(
                [all_z_vectors[edge_idx + 1 + fe] for fe in range(n_future)]
            )
            _apply_givens_batch(givens_data, future_z_mat)
            for fe in range(n_future):
                all_z_vectors[edge_idx + 1 + fe] = future_z_mat[:, fe]

        if len(active_idx) == 0:
            cauchy_factors.append(CauchyFactor(
                d=np.empty(0), roots=np.empty(0), z=np.empty(0),
                norms=np.empty(0), active_idx=np.empty(0, dtype=np.int64),
                givens=givens_list, perm=None, n_global=n_total, rho=rho,
            ))
            resort_perms.append(None)
            continue

        d_active      = current_eigvals[active_idx].copy()
        sort_idx_a    = np.argsort(d_active, kind='mergesort')
        active_sorted = active_idx[sort_idx_a]
        d_sorted      = d_active[sort_idx_a].copy()
        z_active      = z_def[active_sorted].copy()

        _perturb_duplicates_rel(d_sorted, min_gap=1e-12)

        new_roots = solve_secular_all_roots(d_sorted, z_active, rho=rho)
        for ri in range(len(new_roots)):
            if np.isnan(new_roots[ri]):
                new_roots[ri] = d_sorted[ri] + 1e-10
        new_roots = np.sort(new_roots)

        k_active = len(active_sorted)
        z2_rho   = (z_active * z_active) * rho
        col_norms, _ = _compute_norms_and_residual(
            z_active, d_sorted, new_roots, z2_rho, k_active
        )
        root_residual = _check_roots_quality(d_sorted, z2_rho, new_roots, k_active)
        z2_sum        = float(np.sum(z2_rho))
        residual_tol  = max(1e-8 * max(z2_sum, 1.0), 1e-10)

        z_hat    = _compute_z_hat(d_sorted, new_roots, z_active, k_active, rho)
        use_zhat = (root_residual < residual_tol) and np.all(np.isfinite(z_hat))
        if not use_zhat:
            z_hat = None

        cf = CauchyFactor(
            d=d_sorted, roots=new_roots.copy(), z=z_active.copy(),
            norms=col_norms.copy(), active_idx=active_sorted.copy(),
            givens=givens_list, perm=None, n_global=n_total, rho=rho,
            z_hat=z_hat, use_zhat=use_zhat,
        )
        cauchy_factors.append(cf)

        if edge_idx < len(cut_edges) - 1:
            n_fut = len(cut_edges) - edge_idx - 1
            future_z = np.empty((k_active, n_fut), dtype=np.float64)
            for fe in range(n_fut):
                future_z[:, fe] = all_z_vectors[edge_idx + 1 + fe][active_sorted]
            out = np.empty_like(future_z)
            if cf.use_zhat and z_hat is not None:
                fn = (_cauchy_matvec_batch_parallel_stable if k_active > 200
                      else _cauchy_matvec_batch_stable)
                fn(z_hat, d_sorted, new_roots, future_z, out)
            else:
                fn = (_cauchy_matvec_batch_parallel if k_active > 200
                      else _cauchy_matvec_batch)
                fn(z_active, d_sorted, new_roots, col_norms, future_z, out)
            for col in range(out.shape[1]):
                if not np.all(np.isfinite(out[:, col])):
                    out[:, col] = future_z[:, col]
            for fe in range(n_fut):
                all_z_vectors[edge_idx + 1 + fe][active_sorted] = out[:, fe]

        current_eigvals[active_sorted] = new_roots

        if np.any(np.diff(current_eigvals) < -1e-15):
            full_sort = np.argsort(current_eigvals, kind='mergesort')
            current_eigvals = current_eigvals[full_sort]
            for fi in range(edge_idx + 1, len(cut_edges)):
                all_z_vectors[fi] = all_z_vectors[fi][full_sort]
            resort_perms.append(full_sort.copy())
        else:
            resort_perms.append(None)

    _stats['time_secular'] += time.perf_counter() - t0
    _stats['n_merges']     += 1
    _stats['merge_sizes'].append(n)

    assert len(current_eigvals) == n, \
        f"Eigenvalue count mismatch: {len(current_eigvals)} vs {n}"

    # 6. Store decomposition node
    decomp.eigvals        = current_eigvals
    decomp.is_leaf        = False
    decomp.left           = decomp_0
    decomp.right          = decomp_1
    decomp.part_0         = part_0
    decomp.part_1         = part_1
    decomp.part_0_set     = part_0_set
    decomp.part_1_set     = part_1_set
    decomp.cut_edges      = cut_edges
    decomp.cauchy_factors = cauchy_factors
    decomp.sort_perm      = sort_perm
    decomp.resort_perms   = resort_perms

    # 7. Precompute plain eigvec rows for the parent (no D^{-1/2} applied)
    precomputed = {}
    if _parent_needs_nodes:
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


# ============================================================================
# CONVENIENCE TOP-LEVEL ENTRY POINT
# ============================================================================

def eigen_normalized(G, depth=1, target_cut=5, tol=1e-10,
                     loewner_check=False, loewner_kind='norm',
                     loewner_max_n=None, loewner_kwargs=None):
    _ensure_imports()
    """
    Full normalized Laplacian pipeline:
      1. Dry-run to get G_eff and compute global_deg = D_eff from it.
      2. Run the real decomposition on the original G with the same depth
         and seeds — the internal sparsification reproduces G_eff exactly,
         so every level normalizes by the correct D_eff.

    Passing G_eff back into the real run would be wrong: the real run
    would sparsify it a second time (depth more levels), producing a graph
    that no longer matches global_deg.

    loewner_check / loewner_kind / loewner_max_n / loewner_kwargs are forwarded
    to the recursion (see recursive_cauchy_eigen_normalized). When enabled, the
    per-level Loewner bootstrap results are in the returned stats['loewner'].

    Returns (eigvals, decomp, stats, G_eff, global_deg).
    """
    print("  [norm] Building effective graph (dry run)...")
    t0 = time.perf_counter()
    G_eff = build_effective_graph(G, depth=depth, target_cut=target_cut)
    print(f"  [norm] G_eff: {G_eff.number_of_nodes()} nodes, "
          f"{G_eff.number_of_edges()} edges  ({time.perf_counter()-t0:.2f}s)")

    n_global   = G_eff.number_of_nodes()
    global_deg = compute_global_degrees(G_eff, n_nodes=n_global)

    # Pass the original G — internal sparsification reproduces G_eff exactly.
    eigvals, decomp, stats, _, _ = recursive_cauchy_eigen_normalized(
        G, global_deg, depth=depth, target_cut=target_cut, tol=tol,
        loewner_check=loewner_check, loewner_kind=loewner_kind,
        loewner_max_n=loewner_max_n, loewner_kwargs=loewner_kwargs,
    )
    return eigvals, decomp, stats, G_eff, global_deg


# ============================================================================
# VALIDATION
# ============================================================================

def validate_eigenvectors_normalized(decomp, G_eff, global_deg,
                                     n_probes=5, tol_orth=1e-4, tol_diag=5e-3):
    """
    Check:
      (a) ||Q^T e_i|| ≈ 1
      (b) Q^T (L_norm[:,i]) ≈ lam * (Q^T e_i)

    Both L_norm and global_deg are built from G_eff, consistent with the
    decomposition.
    """
    n   = G_eff.number_of_nodes()
    lam = decomp.eigvals
    rng = np.random.RandomState(123)
    probe_nodes = rng.choice(np.arange(n, dtype=np.int64),
                             size=min(n_probes, n), replace=False)
    norm_errs, diag_errs = [], []

    for i in probe_nodes:
        i = int(i)
        c = decomp.qt_dot_sparse({i: 1.0})
        norm_errs.append(abs(np.linalg.norm(c) - 1.0))

        di, row = global_deg[i], {}
        diag_contrib = 0.0
        for j, data in G_eff[i].items():
            wij = data.get('weight', 1.0)
            dj  = global_deg[int(j)]
            off = -wij / np.sqrt(di * dj)
            row[int(j)] = row.get(int(j), 0.0) + off
            diag_contrib += wij / di
        row[i] = row.get(i, 0.0) + diag_contrib

        lhs   = decomp.qt_dot_sparse(row)
        rhs   = lam * c
        denom = max(np.linalg.norm(lhs), np.linalg.norm(rhs), 1e-12)
        diag_errs.append(np.linalg.norm(lhs - rhs) / denom)

    max_norm = max(norm_errs) if norm_errs else np.inf
    max_diag = max(diag_errs) if diag_errs else np.inf
    return {
        'max_norm_err':     max_norm,
        'mean_norm_err':    float(np.mean(norm_errs)) if norm_errs else np.inf,
        'max_diag_relerr':  max_diag,
        'mean_diag_relerr': float(np.mean(diag_errs)) if diag_errs else np.inf,
        'PASS': (max_norm < tol_orth and max_diag < tol_diag),
    }


# ============================================================================
# TESTS
# ============================================================================

def _make_test_graph(n, seed, weighted=False):
    G = nx.barabasi_albert_graph(n, 3, seed=seed)
    if weighted:
        rng = np.random.RandomState(seed)
        for u, v in G.edges():
            G[u][v]['weight'] = rng.uniform(0.5, 5.0)
    return G


def _ground_truth_eigvals(G_eff, global_deg):
    M = normalized_full_laplacian_dense(G_eff, global_deg)
    return np.sort(np.linalg.eigvalsh(M))


def _run_test(label, G, depth=1, target_cut=5, tol=1e-10,
              err_tol=1e-6, n_probes=10):
    _ensure_imports()
    print("=" * 60)
    print(label)
    print("=" * 60)
    n = G.number_of_nodes()

    eigvals_rec, decomp, stats, G_eff, global_deg = eigen_normalized(
        G, depth=depth, target_cut=target_cut, tol=tol,
    )

    eigvals_true  = _ground_truth_eigvals(G_eff, global_deg)
    eigvals_rec_s = np.sort(eigvals_rec)
    err_per = np.abs(eigvals_true - eigvals_rec_s)
    max_err = float(np.max(err_per))
    max_abs = max(float(np.max(np.abs(eigvals_true))), 1.0)
    rel_err = max_err / max_abs

    print(f"  n={n}, depth={depth}, target_cut={target_cut}")
    print(f"  G_eff: {G_eff.number_of_edges()} edges  "
          f"(orig: {G.number_of_edges()})")
    print(f"  Eigenvalue max err: {max_err:.2e}  (rel {rel_err:.2e})")
    print(f"  Mean err: {np.mean(err_per):.2e}   "
          f"95th pctl: {np.percentile(err_per, 95):.2e}")
    print(f"  Trace err: {abs(np.sum(eigvals_true)-np.sum(eigvals_rec_s)):.2e}")
    print(f"  L_norm max eigval (≤2): {eigvals_true[-1]:.6f}")

    if not decomp.is_leaf:
        for i, cf in enumerate(decomp.cauchy_factors):
            if len(cf.d) > 0:
                orth = validate_cauchy_orthogonality(cf, n_probes=10, tol=1e-4)
                print(f"  CF {i}: k={len(cf.d)}, orth_err={orth['max_err']:.2e}")

    vec_val = validate_eigenvectors_normalized(
        decomp, G_eff, global_deg, n_probes=n_probes,
        tol_orth=1e-4, tol_diag=5e-3,
    )
    print(f"  Eigvec norm err: {vec_val['max_norm_err']:.2e}  "
          f"diag err: {vec_val['max_diag_relerr']:.2e}")

    passed = (rel_err < err_tol) and vec_val['PASS']
    print(f"  {'✓ PASS' if passed else '✗ FAIL'}")
    print()
    return passed


def test_normalized_vs_combinatorial():
    _ensure_imports()
    print("=" * 60)
    print("NORMALIZED vs COMBINATORIAL SPECTRUM SANITY CHECK")
    print("=" * 60)
    G = nx.barabasi_albert_graph(200, 3, seed=7)
    n = G.number_of_nodes()
    global_deg = compute_global_degrees(G, n_nodes=n)

    L_comb = np.zeros((n, n))
    for u, v, data in G.edges(data=True):
        w = data.get('weight', 1.0)
        L_comb[u, v] -= w; L_comb[v, u] -= w
        L_comb[u, u] += w; L_comb[v, v] += w
    ev_comb = np.sort(np.linalg.eigvalsh(L_comb))

    L_norm = normalized_full_laplacian_dense(G, global_deg)
    ev_norm = np.sort(np.linalg.eigvalsh(L_norm))

    print(f"  Combinatorial: λ_2={ev_comb[1]:.4f}, λ_max={ev_comb[-1]:.3f}")
    print(f"  Normalized:    λ_2={ev_norm[1]:.4f}, λ_max={ev_norm[-1]:.6f}")
    print(f"  Normalized max ≤ 2: {ev_norm[-1] <= 2.0 + 1e-10}")
    print(f"  Spectra differ: {not np.allclose(ev_comb, ev_norm)}")

    passed = (ev_norm[-1] <= 2.0 + 1e-10) and (not np.allclose(ev_comb, ev_norm))
    print(f"  {'✓ PASS' if passed else '✗ FAIL'}")
    print()
    return passed


def test_small_normalized():
    return _run_test("SMALL — NORMALIZED LAPLACIAN (n=100)",
                     _make_test_graph(100, 42), depth=1, err_tol=1e-6)


def test_medium_normalized():
    return _run_test("MEDIUM — NORMALIZED LAPLACIAN (n=1000)",
                     _make_test_graph(1000, 42), depth=1, err_tol=5e-3)


def test_weighted_normalized():
    return _run_test("WEIGHTED — NORMALIZED LAPLACIAN (n=100)",
                     _make_test_graph(100, 42, weighted=True),
                     depth=1, err_tol=1e-5)


def test_deep_normalized():
    return _run_test("DEEP — NORMALIZED LAPLACIAN (n=500, depth=2)",
                     _make_test_graph(500, 99), depth=2,
                     err_tol=1e-4, n_probes=5)


def test_loewner_bootstrap():
    """Smoke test: per-level Loewner bootstrap runs and reports finite bounds."""
    _ensure_imports()
    print("=" * 60)
    print("LOEWNER BOOTSTRAP SMOKE TEST (n=500, depth=2)")
    print("=" * 60)
    G = _make_test_graph(500, 99)
    ok = True
    for kind in ('comb', 'norm'):
        _, _, stats, _, _ = eigen_normalized(
            G, depth=2, target_cut=5, loewner_check=True, loewner_kind=kind,
            loewner_max_n=None,
            loewner_kwargs={'n_probes': 4, 'n_iter': 15},
        )
        lw_list = stats.get('loewner', [])
        finite = all(np.isfinite(lw['lam_max']) and np.isfinite(lw['lam_min'])
                     for lw in lw_list)
        print(f"  kind={kind}: {len(lw_list)} levels measured, "
              f"all finite={finite}, "
              f"time_loewner={stats.get('time_loewner', 0.0):.2f}s")
        for lw in lw_list:
            print(f"    L{lw['level']} n={lw['n']}: "
                  f"lam in [{lw['lam_min']:.4f}, {lw['lam_max']:.4f}]  "
                  f"kappa={lw['kappa']:.4f}")
        ok = ok and (len(lw_list) > 0) and finite
    print(f"  {'✓ PASS' if ok else '✗ FAIL'}")
    print()
    return ok


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    import argparse
    from src.core.testing_fast_cauchy_factor import load_ogbn_arxiv_subgraph
    _ensure_imports()
    from .recursive_decomp import (
        _cauchy_matvec_batch, _cauchy_matvec_batch_parallel_stable,
        _cauchy_matvec_batch_parallel, _cauchy_matvec_batch_stable,
        _apply_givens_batch,
    )
    from decomp_io import test_save_load_roundtrip
 
    parser = argparse.ArgumentParser(
        description='Recursive Cauchy — NORMALIZED Laplacian (ogbn-arxiv)'
    )
    parser.add_argument('--sizes', type=int, nargs='+', default=[5000, 10000],
                        help='Target subgraph sizes (default: 5000 10000)')
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
    # Loewner bootstrap controls
    parser.add_argument('--loewner-check', action='store_true',
                        help='Run per-level Spielman–Srivastava Loewner bootstrap')
    parser.add_argument('--loewner-kind', choices=['comb', 'norm'], default='norm',
                        help='Combinatorial or own-degree-normalized bound')
    parser.add_argument('--loewner-max-n', type=int, default=4000,
                        help='Only measure levels with n <= this (default 4000)')
    parser.add_argument('--loewner-probes', type=int, default=6)
    parser.add_argument('--loewner-iter', type=int, default=20)
    args = parser.parse_args()

    os.makedirs('decomp_files', exist_ok=True)

    loewner_kwargs = {'n_probes': args.loewner_probes, 'n_iter': args.loewner_iter}

    # JIT warm-up
    print("Warming up Numba JIT...")
    _d  = np.array([1., 2., 3., 4., 5.])
    _z  = np.array([.1, .2, .3, .2, .1])
    _   = solve_secular_all_roots(_d, _z, 1.0)
    _   = _compute_column_norms(_z, _d, _d + 0.5, 5)
    _gv = np.array([[0., 1., .6, .8]], dtype=np.float64)
    _fm = np.random.randn(5, 2); _apply_givens_batch(_gv, _fm)
    _X  = np.random.randn(5, 3); _out = np.empty_like(_X); _nr = np.ones(5)
    _   = deflate_numba(_d, _z, 1e-8)
    _dd = np.array([1., 1., 2., 2., 3.]); _perturb_duplicates_rel(_dd)
    _   = _cauchy_QtQx(_d, _d+.5, _z, _nr, _z, 5)
    _zh = _compute_z_hat(_d, _d+.5, _z, 5)
    _cauchy_matvec_batch_stable(_zh, _d, _d+.5, _X, _out)
    _cauchy_matvec_batch_parallel_stable(_zh, _d, _d+.5, _X, _out)
    _cauchy_matvec_batch(_z, _d, _d+.5, _nr, _X, _out)
    _cauchy_matvec_batch_parallel(_z, _d, _d+.5, _nr, _X, _out)
    _z2 = _z*_z; _check_roots_quality(_d, _z2, _d+.5, 5)
    _compute_norms_and_residual(_z, _d, _d+.5, _z2, 5)
    print("JIT warmup done.\n")

    if not args.skip_validation:
        results = [
            test_normalized_vs_combinatorial(),
            test_small_normalized(),
            test_medium_normalized(),
            test_weighted_normalized(),
            test_deep_normalized(),
            test_loewner_bootstrap(),
        ]
        if not all(results):
            print("VALIDATION FAILED"); exit(1)
        print("All validation tests passed.\n")
    else:
        print("Skipping validation tests.\n")

    # ================================================================
    # LOAD ogbn-arxiv GRAPHS
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
            from scipy.sparse import coo_matrix as _coo

            print("  Loading full ogbn-arxiv dataset...")
            dataset = NodePropPredDataset(name='ogbn-arxiv', root='data/')
            graph, _ = dataset[0]

            edge_index = graph['edge_index']
            src, dst   = edge_index[0], edge_index[1]
            n_total    = graph['num_nodes']
            print(f"  ogbn-arxiv: {n_total} nodes, {len(src)} directed edges")

            ones = np.ones(len(src), dtype=np.float64)
            A = _coo((ones, (src, dst)), shape=(n_total, n_total))
            A = (A + A.T > 0).astype(np.float64)
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
                G = load_ogbn_arxiv_subgraph(n_target, seed=42 + run)
                arxiv_graphs[(n_target, run)] = G
                print(f"  n={n_target}, run={run}: {G.number_of_nodes()} nodes, "
                      f"{G.number_of_edges()} edges, "
                      f"avg_degree={2*G.number_of_edges()/G.number_of_nodes():.1f}")

    print()

    # ================================================================
    # BENCHMARK
    # ================================================================

    dense_limit = 0 if args.no_dense else 20000

    print("=" * 70)
    if args.full:
        print("RECURSIVE NORMALIZED LAPLACIAN BENCHMARK (ogbn-arxiv FULL)")
    else:
        print("RECURSIVE NORMALIZED LAPLACIAN BENCHMARK (ogbn-arxiv)")
    print("=" * 70)

    recursive_results = {}

    for n_nodes in target_sizes:
        print(f"\n{'=' * 70}")
        if args.full:
            print(f"NORMALIZED BENCHMARK: FULL ogbn-arxiv (n = {n_nodes})")
        else:
            print(f"NORMALIZED BENCHMARK: n ≈ {n_nodes} (ogbn-arxiv)")
        print(f"{'=' * 70}")

        actual_runs = 1 if args.full else n_runs
        for run in range(actual_runs):
            G = arxiv_graphs[(0, 0)] if args.full else arxiv_graphs[(n_nodes, run)]
            actual_n = G.number_of_nodes()

            # Dense baseline (normalized Laplacian)
            if actual_n <= dense_limit:
                _deg_full = compute_global_degrees(G, n_nodes=actual_n)
                L_norm_dense = normalized_full_laplacian_dense(G, _deg_full)
                t0 = time.perf_counter()
                eigvals_dense_orig = np.sort(np.linalg.eigvalsh(L_norm_dense))
                t_dense = time.perf_counter() - t0
                del L_norm_dense
                print(f"\n  [run {run}] n={actual_n}, edges={G.number_of_edges()}, "
                      f"avg_deg={2*G.number_of_edges()/actual_n:.1f}")
                print(f"  Dense normalized baseline: {t_dense:.2f}s")
            else:
                eigvals_dense_orig = None
                t_dense = np.nan
                print(f"\n  [run {run}] n={actual_n}, edges={G.number_of_edges()}, "
                      f"avg_deg={2*G.number_of_edges()/actual_n:.1f}")
                print(f"  Dense baseline: skipped (n={actual_n} > {dense_limit})")

            for depth in args.depths:
                print(f"\n  Starting depth={depth}...")
                t0 = time.perf_counter()
                eigvals_rec, decomp, stats, G_eff, global_deg = eigen_normalized(
                    G, depth=depth, target_cut=args.target_cut, tol=1e-10,
                    loewner_check=args.loewner_check,
                    loewner_kind=args.loewner_kind,
                    loewner_max_n=args.loewner_max_n,
                    loewner_kwargs=loewner_kwargs,
                )
                t_total = time.perf_counter() - t0
                speedup = t_dense / t_total if not np.isnan(t_dense) else float('nan')

                # Save decomposition — filename encodes normalized + size + depth
                if args.full:
                    save_tag = f"norm_arxiv_full_depth{depth}"
                else:
                    save_tag = f"norm_arxiv_n{actual_n}_depth{depth}_run{run}"
                save_path = os.path.join("decomp_files", f"decomp_{save_tag}.pkl")
                test_save_load_roundtrip(decomp, path=save_path)

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
                print(f"      Dry run (G_eff):  included above")
                print(f"      Spectral cuts:    {stats['time_cuts']:.2f}s")
                print(f"      Sparsification:   {stats['time_sparsify']:.2f}s")
                print(f"      Base eig:         {stats['time_base_eig']:.2f}s")
                print(f"      z computation:    {stats['time_z_compute']:.2f}s")
                print(f"      Secular eqs:      {stats['time_secular']:.2f}s")
                if 'time_loewner' in stats:
                    print(f"      Loewner boot:     {stats['time_loewner']:.2f}s")
                print(f"    Memory: {decomp.memory_bytes() / 1e6:.1f}MB "
                      f"(vs {actual_n**2 * 8 / 1e6:.1f}MB dense)")
                print(f"    G_eff: {G_eff.number_of_nodes()} nodes, "
                      f"{G_eff.number_of_edges()} edges")
                print(f"    Saved: {save_path}")

                if args.loewner_check and stats.get('loewner'):
                    print(f"    Loewner-{args.loewner_kind} per-level bounds:")
                    for lw in stats['loewner']:
                        print(f"      L{lw['level']} n={lw['n']} dropped={lw['edges_dropped']}: "
                              f"lam in [{lw['lam_min']:.4f}, {lw['lam_max']:.4f}]  "
                              f"eps_-={lw['eps_minus']:.2e} eps_+={lw['eps_plus']:.2e}  "
                              f"kappa={lw['kappa']:.4f}")

                if actual_n <= dense_limit:
                    eigvals_true  = _ground_truth_eigvals(G_eff, global_deg)
                    eigvals_rec_s = np.sort(eigvals_rec)
                    err_per  = np.abs(eigvals_true - eigvals_rec_s)
                    max_err  = float(np.max(err_per))
                    max_abs  = max(float(np.max(np.abs(eigvals_true))), 1.0)
                    rel_err  = max_err / max_abs
                    trace_err = abs(np.sum(eigvals_true) - np.sum(eigvals_rec_s))

                    print(f"    Validation (vs G_eff normalized Laplacian):")
                    print(f"      Eigenvalue max error:  {max_err:.2e}")
                    print(f"      Relative max error:    {rel_err:.2e}")
                    print(f"      Trace error:           {trace_err:.2e}")
                    print(f"      Mean error:            {np.mean(err_per):.2e}")
                    print(f"      95th pctl error:       {np.percentile(err_per, 95):.2e}")
                    print(f"      L_norm max eigval:     {eigvals_true[-1]:.6f} (≤2: "
                          f"{'✓' if eigvals_true[-1] <= 2.0 + 1e-10 else '✗'})")
                    print(f"      {'✓ PASS' if rel_err < 1e-4 else '✗ FAIL'} "
                          f"(rel_err={rel_err:.2e})")
                else:
                    print(f"    Validation: skipped (n={actual_n} > {dense_limit})")

                ev_val = validate_eigenvectors_normalized(
                    decomp, G_eff, global_deg, n_probes=10,
                    tol_orth=1e-4, tol_diag=5e-3,
                )
                print(f"    Eigvec implicit check (10 probes):")
                print(f"      Norm max err: {ev_val['max_norm_err']:.2e}")
                print(f"      Diag max rel: {ev_val['max_diag_relerr']:.2e}")
                print(f"      {'✓ PASS' if ev_val['PASS'] else '✗ FAIL'}")

                recursive_results[(n_nodes, depth, run)] = {
                    'time':      t_total,
                    'speedup':   speedup,
                    'stats':     stats,
                    'memory_mb': decomp.memory_bytes() / 1e6,
                    'actual_n':  actual_n,
                }

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY — NORMALIZED LAPLACIAN")
    print("=" * 70)
    print(f"{'n_target':>8s} {'n_actual':>8s} {'depth':>6s} {'run':>4s} "
          f"{'time(s)':>8s} {'speedup':>8s} {'mem(MB)':>8s} "
          f"{'leaves':>7s} {'max_leaf':>9s} {'cut_edges':>10s}")
    print("-" * 85)
    for (n, d, r), res in sorted(recursive_results.items()):
        s     = res['stats']
        actual = res.get('actual_n', n)
        sp    = f"{res['speedup']:>7.1f}x" if not np.isnan(res['speedup']) else f"{'N/A':>8s}"
        print(f"{n:>8d} {actual:>8d} {d:>6d} {r:>4d} {res['time']:>8.2f} "
              f"{sp} {res['memory_mb']:>8.1f} {s['n_leaves']:>7d} "
              f"{s['peak_dense_size']:>9d} {s['total_cut_edges']:>10d}")