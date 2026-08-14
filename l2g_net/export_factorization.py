#!/usr/bin/env python3
"""
Compute a depth-1 Cauchy factorization with the `cauchy_factorization`
package and export the spectral tensors consumed by train.py:

  idx_S, idx_T          node indices of the two subgraphs
  U_S, U_T              local GFT bases (leaf eigenvectors)
  eig_S, eig_T          local eigenvalues
  eigvals               global eigenvalues (of the effective Laplacian)
  basis_cauchy          N x N transition matrix, block basis -> global basis
                        (the product of the Cauchy factors)

basis_cauchy is materialized from the factorization alone: leaf transforms
map leaf eigenvectors to identity columns, so Q^T U_block equals the Cauchy
chain applied to the identity (decomp._apply_parent_chain_batch).

Usage:
  python export_factorization.py --npz data/minesweeper/raw/minesweeper.npz \
      --target-cut 5 --laplacian norm --out minesweeper_factorization.npz
"""
import argparse
import os
import sys
import time

import numpy as np
import networkx as nx

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cauchy_factorization import recursive_cauchy_eigen, eigen_normalized


def load_graph_npz(path):
    data = np.load(path, allow_pickle=True)
    edges = np.asarray(data["edges"], dtype=np.int64)
    n = int(max(edges.max() + 1,
                data["node_features"].shape[0] if "node_features" in data else 0))
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for u, v in edges:
        if u != v:
            G.add_edge(int(u), int(v), weight=1.0)
    return G


def export(G, target_cut, laplacian, seed_note=""):
    n = G.number_of_nodes()
    print(f"graph: n={n} m={G.number_of_edges()} {seed_note}")

    t0 = time.perf_counter()
    if laplacian == "norm":
        out = eigen_normalized(G, depth=1, target_cut=target_cut, tol=1e-10)
    else:
        out = recursive_cauchy_eigen(G, depth=1, target_cut=target_cut,
                                     tol=1e-10, use_zhat=True)
    t_factor = time.perf_counter() - t0
    eigvals, decomp = np.asarray(out[0]), out[1]
    print(f"factorization: {t_factor:.2f}s "
          f"({len(decomp.cauchy_factors)} Cauchy factors)")

    left, right = decomp.left, decomp.right
    if left is None or not left.is_leaf or not right.is_leaf:
        raise RuntimeError("expected a depth-1 decomposition with two leaves")

    idx_S = np.asarray(left.leaf_nodes, dtype=np.int64)
    idx_T = np.asarray(right.leaf_nodes, dtype=np.int64)

    t0 = time.perf_counter()
    basis_cauchy = materialize_chain_dense(decomp, n)
    print(f"basis_cauchy materialized: {time.perf_counter() - t0:.2f}s")

    return {
        "idx_S": idx_S, "idx_T": idx_T,
        "U_S": left.leaf_eigvecs, "U_T": right.leaf_eigvecs,
        "eig_S": np.asarray(left.eigvals), "eig_T": np.asarray(right.eigvals),
        "eigvals": eigvals,
        "basis_cauchy": basis_cauchy,
        "laplacian": np.array(laplacian),
        "target_cut": np.array(target_cut),
    }, decomp


def materialize_chain_dense(decomp, n):
    """Dense product of the Cauchy chain via BLAS (fast path).

    Equivalent to decomp._apply_parent_chain_batch(np.eye(n)) but applies
    each factor's k x k Cauchy block with one dgemm instead of the
    per-vector numba kernels.
    """
    B = np.eye(n)[decomp.sort_perm, :]
    for cf_idx, cf in enumerate(decomp.cauchy_factors):
        for (i_g, j_g, c_g, s_g) in cf.givens:
            bi = B[i_g, :].copy()
            bj = B[j_g, :].copy()
            B[i_g, :] = c_g * bi + s_g * bj
            B[j_g, :] = -s_g * bi + c_g * bj
        if len(cf.d) > 0:
            d, roots = cf.d, cf.roots
            delta = d[None, :] - roots[:, None]
            tiny = np.abs(delta) < 1e-300
            delta[tiny] = np.where(delta[tiny] >= 0, 1e-300, -1e-300)
            if cf.use_zhat and cf.z_hat is not None:
                C = cf.z_hat[None, :] / delta
            else:
                inv_norms = np.where(np.abs(cf.norms) > 1e-300,
                                     1.0 / cf.norms, 0.0)
                C = (cf.z[None, :] / delta) * inv_norms[:, None]
            active = cf.active_idx
            B[active, :] = C @ B[active, :]
        if decomp.resort_perms is not None and cf_idx < len(decomp.resort_perms):
            rp = decomp.resort_perms[cf_idx]
            if rp is not None:
                B = B[rp, :]
    return B


def validate(tensors, decomp, n_probes=5, seed=0):
    """Check basis_cauchy against Q^T applied directly, and orthogonality."""
    n = tensors["basis_cauchy"].shape[0]
    U_block = np.zeros((n, n))
    nS = len(tensors["idx_S"])
    U_block[tensors["idx_S"][:, None], np.arange(nS)] = tensors["U_S"]
    U_block[tensors["idx_T"][:, None], np.arange(len(tensors["idx_T"])) + nS] = \
        tensors["U_T"]
    rng = np.random.RandomState(seed)
    errs, orth = [], []
    for _ in range(n_probes):
        x = rng.randn(n)
        ref = decomp.qt_dot(x)
        via = tensors["basis_cauchy"] @ (U_block.T @ x)
        errs.append(np.linalg.norm(ref - via) / np.linalg.norm(ref))
        orth.append(abs(np.linalg.norm(via) - np.linalg.norm(x))
                    / np.linalg.norm(x))
    print(f"validate: basis_cauchy vs qt_dot rel err max={max(errs):.3e}, "
          f"orthogonality max={max(orth):.3e}")
    return max(errs)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz", required=True,
                   help="dataset npz with an 'edges' array (Platonov format)")
    p.add_argument("--target-cut", type=int, default=5)
    p.add_argument("--laplacian", choices=("norm", "comb"), default="norm")
    p.add_argument("--out", required=True)
    p.add_argument("--validate", action="store_true",
                   help="probe-check basis_cauchy against the factorization")
    args = p.parse_args()

    G = load_graph_npz(args.npz)
    tensors, decomp = export(G, args.target_cut, args.laplacian,
                             seed_note=f"from {args.npz}")
    if args.validate:
        err = validate(tensors, decomp)
        if err > 1e-6:
            raise SystemExit(f"validation failed: rel err {err:.3e}")
    np.savez_compressed(args.out, **tensors)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
