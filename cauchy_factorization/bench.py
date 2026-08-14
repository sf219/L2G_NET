#!/usr/bin/env python3
"""
Benchmark: Cauchy factorization vs dense eigendecomposition.

Builds a graph, runs the recursive Cauchy factorization (combinatorial or
symmetric-normalized Laplacian), and reports:

  * t_factor      : wall time of the factorization
  * t_eigh        : wall time of numpy.linalg.eigh on the dense Laplacian
                    (skippable with --no-dense for large n)
  * speedup       : t_eigh / t_factor
  * memory        : implicit factorization bytes vs dense n^2 doubles
  * validation    : eigenvalue error vs eigh of the EFFECTIVE (sparsified)
                    Laplacian, plus orthogonality probes of Q^T
                    (skippable with --no-validate)

The factorization result is exact for the effective graph obtained after
cut sparsification; --target-cut controls how many cut edges are kept per
partition level and hence the approximation/speed trade-off vs the input
graph.

Defaults reproduce the L2G-Net paper's synthetic setup (Sec. 6.1):
Barabasi-Albert graphs, depth 2 (4 subgraphs), target_cut 5, median of 3.

Examples
--------
# runtime vs n (paper Fig. 3)
OMP_NUM_THREADS=8 python -m cauchy_factorization.bench \
    --n-grid 4000 6000 8000 10000 14000 20000

# runtime vs cut size at fixed n (paper Fig. 4)
OMP_NUM_THREADS=8 python -m cauchy_factorization.bench \
    --n 8000 --k-grid 2 5 10 15 20 25 30 --no-dense
"""
import argparse
import os
import time

import threadpoolctl
_n_threads = int(os.environ.get("OMP_NUM_THREADS", "1"))
threadpoolctl.threadpool_limits(limits=_n_threads)

import numpy as np

try:
    import numba
    numba.set_num_threads(max(_n_threads, 1))
except Exception:
    pass

from .factorization import recursive_cauchy_eigen
from .normalized import (compute_global_degrees, eigen_normalized,
                         normalized_full_laplacian_dense)
from .graphs import (combinatorial_laplacian_dense, make_graph,
                     normalized_laplacian_dense)


def parse():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group("graph")
    g.add_argument("--kind", default="ba",
                   choices=("sbm", "ba", "er", "grid", "file"),
                   help="default 'ba' matches the L2G-Net paper (Sec. 6.1)")
    g.add_argument("--n", type=int, default=8000)
    g.add_argument("--n-grid", type=int, nargs="*", default=None,
                   help="run a scaling table over these sizes instead of --n")
    g.add_argument("--k-grid", type=int, nargs="*", default=None,
                   help="sweep target_cut at fixed n (paper Fig. 4)")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--blocks", type=int, default=4)
    g.add_argument("--p_in", type=float, default=0.15)
    g.add_argument("--p_out", type=float, default=0.005)
    g.add_argument("--m-edges", type=int, default=3)
    g.add_argument("--path", default=None, help="edge list for --kind file")

    f = p.add_argument_group("factorization")
    f.add_argument("--laplacian", choices=("comb", "norm"), default="comb",
                   help="combinatorial or symmetric-normalized Laplacian")
    f.add_argument("--depth", type=int, default=2,
                   help="recursion depth; 2 = 4 subgraphs as in the paper")
    f.add_argument("--target-cut", type=int, default=5,
                   help="cut edges kept per partition after sparsification")
    f.add_argument("--tol", type=float, default=1e-10,
                   help="secular root tolerance")
    f.add_argument("--no-zhat", action="store_false", dest="use_zhat",
                   help="disable Gu-Eisenstat z_hat stabilization")
    f.add_argument("--sparsifier", choices=("ss", "degree"), default="ss",
                   help="cut sampler: 'ss' = effective-resistance sketches "
                        "(default, CG solves per merge); 'degree' = "
                        "deterministic 1/deg importance, no linear solves")
    f.add_argument("--n-sketches", type=int, default=20,
                   help="CG sketches for the 'ss' sparsifier")

    b = p.add_argument_group("benchmark")
    b.add_argument("--no-dense", action="store_false", dest="dense",
                   help="skip the dense eigh reference (large n)")
    b.add_argument("--no-validate", action="store_false", dest="validate",
                   help="skip validation against the effective Laplacian")
    b.add_argument("--n-probes", type=int, default=5,
                   help="random probes for orthogonality validation")
    b.add_argument("--repeats", type=int, default=3,
                   help="repeat the factorization; report the median "
                        "(3 in the paper)")
    return p.parse_args()


def dense_reference(spec, laplacian, G_eff=None, global_deg=None):
    """Dense Laplacian of the ORIGINAL graph (for the eigh timing) and of
    the effective graph (for validation).

    For laplacian='norm' the effective operator is normalized by the
    EFFECTIVE graph's degrees (`global_deg` as returned by eigen_normalized),
    matching the pipeline's own convention.
    """
    if laplacian == "comb":
        L = combinatorial_laplacian_dense(spec.G)
        L_eff = combinatorial_laplacian_dense(G_eff) if G_eff is not None else None
    else:
        L = normalized_laplacian_dense(spec.G, compute_global_degrees(spec.G))
        L_eff = (normalized_full_laplacian_dense(G_eff, global_deg)
                 if G_eff is not None else None)
    return L, L_eff


def run_once(args, n, target_cut=None):
    spec = make_graph(args.kind, n=n, seed=args.seed, blocks=args.blocks,
                      p_in=args.p_in, p_out=args.p_out, m_edges=args.m_edges,
                      path=args.path)
    print(f"graph: {spec.name} n={spec.n} m={spec.m} "
          f"avg_deg={spec.avg_degree:.2f}")

    k_cut = target_cut if target_cut is not None else args.target_cut
    factor_fn = (recursive_cauchy_eigen if args.laplacian == "comb"
                 else eigen_normalized)

    # select the cut sampler (resolved dynamically inside the recursion)
    from . import factorization as _f
    if not hasattr(_f, "_orig_sparsify_ss"):
        _f._orig_sparsify_ss = _f.sparsify_cut_edges_ss
    if args.sparsifier == "degree":
        _f.sparsify_cut_edges_ss = (
            lambda G, p0, p1, target_cut=5, **kw:
            _f.sparsify_cut_edges(G, p0, p1, target_cut=target_cut))
    else:
        ns = args.n_sketches
        _f.sparsify_cut_edges_ss = (
            lambda G, p0, p1, target_cut=5, n_sketches=20, **kw:
            _f._orig_sparsify_ss(G, p0, p1, target_cut=target_cut,
                                 n_sketches=ns, **kw))

    # eigen_normalized fixes use_zhat internally; only the combinatorial
    # entry point exposes the switch
    kw = {"use_zhat": args.use_zhat} if args.laplacian == "comb" else {}

    # warm-up numba on a tiny instance so JIT is excluded from timings
    tiny = make_graph("er", n=64, seed=1)
    factor_fn(tiny.G, depth=1, target_cut=3, tol=args.tol, **kw)

    times = []
    for _ in range(max(args.repeats, 1)):
        t0 = time.perf_counter()
        out = factor_fn(spec.G, depth=args.depth, target_cut=k_cut,
                        tol=args.tol, **kw)
        times.append(time.perf_counter() - t0)
    t_factor = float(np.median(times))
    eigvals, decomp = np.asarray(out[0]), out[1]
    stats = out[2] if len(out) > 2 else {}
    G_eff = out[3] if len(out) > 3 else None
    global_deg = out[4] if (args.laplacian == "norm" and len(out) > 4) else None

    mem_factor = decomp.memory_bytes() if hasattr(decomp, "memory_bytes") else None
    mem_dense = 8.0 * spec.n * spec.n
    print(f"t_factor: {t_factor:.3f}s   "
          f"(depth={args.depth} target_cut={k_cut} "
          f"laplacian={args.laplacian} threads={_n_threads})")
    if stats:
        print(f"  stats: leaves={stats.get('n_leaves')} "
              f"merges={stats.get('n_merges')} "
              f"cut_edges={stats.get('total_cut_edges')} "
              f"root_residual_max={stats.get('root_residual_max', 0):.3e}")
        parts = [(name, stats.get(key, 0.0)) for name, key in
                 [("partition", "time_cuts"), ("sparsify", "time_sparsify"),
                  ("base_eig", "time_base_eig"), ("secular", "time_secular"),
                  ("z_compute", "time_z_compute")]]
        acc = sum(t for _, t in parts)
        breakdown = "  ".join(f"{nm}={t:.3f}s" for nm, t in parts)
        print(f"  runtime breakdown: {breakdown}  (untracked="
              f"{max(t_factor - acc, 0.0):.3f}s)")
    if mem_factor is not None:
        print(f"  memory: factorization {mem_factor / 1e6:.1f} MB vs "
              f"dense eigenvectors {mem_dense / 1e6:.1f} MB "
              f"({mem_dense / max(mem_factor, 1):.1f}x)")

    row = {"n": spec.n, "m": spec.m, "k": k_cut, "t_factor": t_factor,
           "t_eigh": None, "speedup": None, "err_s_eff": None}

    if args.dense or args.validate:
        L, L_eff = dense_reference(spec, args.laplacian,
                                   G_eff if args.validate else None,
                                   global_deg)

    if args.dense:
        t0 = time.perf_counter()
        eigvals_full, _ = np.linalg.eigh(L)
        t_eigh = time.perf_counter() - t0
        row["t_eigh"] = t_eigh
        row["speedup"] = t_eigh / t_factor
        print(f"t_eigh  : {t_eigh:.3f}s   speedup (eigh/factor): "
              f"{t_eigh / t_factor:.2f}x")

    if args.validate and L_eff is not None:
        evals_eff = np.linalg.eigvalsh(L_eff)
        err_s = (np.linalg.norm(np.sort(eigvals) - evals_eff)
                 / max(np.linalg.norm(evals_eff), 1e-30))
        row["err_s_eff"] = err_s
        rng = np.random.RandomState(123)
        orth = []
        res = []
        for _ in range(args.n_probes):
            x = rng.randn(spec.n)
            c = decomp.qt_dot(x)
            orth.append(abs(np.linalg.norm(c) - np.linalg.norm(x))
                        / np.linalg.norm(x))
            res.append(np.linalg.norm(decomp.qt_dot(L_eff @ x) - eigvals * c)
                       / max(np.linalg.norm(L_eff @ x), 1e-30))
        print(f"validation vs EFFECTIVE Laplacian: "
              f"err_s={err_s:.3e} orth_max={max(orth):.3e} "
              f"eigres_max={max(res):.3e}")
    return row


def main():
    args = parse()
    if args.k_grid:
        rows = [run_once(args, args.n, target_cut=k) for k in args.k_grid]
    else:
        sizes = args.n_grid if args.n_grid else [args.n]
        rows = [run_once(args, n) for n in sizes]
    if len(rows) > 1:
        print("\nresults table:")
        print(f"{'n':>8s} {'m':>10s} {'k':>4s} {'t_factor':>10s} "
              f"{'t_eigh':>10s} {'speedup':>8s} {'err_s_eff':>10s}")
        for r in rows:
            te = f"{r['t_eigh']:.3f}" if r["t_eigh"] else "-"
            sp = f"{r['speedup']:.2f}x" if r["speedup"] else "-"
            es = f"{r['err_s_eff']:.2e}" if r["err_s_eff"] is not None else "-"
            print(f"{r['n']:8d} {r['m']:10d} {r['k']:4d} "
                  f"{r['t_factor']:10.3f} {te:>10s} {sp:>8s} {es:>10s}")


if __name__ == "__main__":
    main()
