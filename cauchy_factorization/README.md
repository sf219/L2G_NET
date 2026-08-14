# Cauchy factorization

Standalone code for the recursive Cauchy factorization of graph Laplacian
eigendecompositions (L2G-Net, arXiv:2602.18837). Factorization only — the
HODLR fast matvec is a separate project and is not included.

Pipeline: bisect with METIS → sparsify cut edges to
`target_cut` → eigendecompose parts recursively → reattach each kept edge as
a rank-one secular update, stored as an implicit orthogonal Cauchy factor.
The result is exact for the sparsified (effective) Laplacian.

## Entry points

```python
from cauchy_factorization import recursive_cauchy_eigen, eigen_normalized

evals, decomp, stats, G_eff, *_ = recursive_cauchy_eigen(G, depth=2, target_cut=5)
c = decomp.qt_dot(x)          # exact Q^T x
decomp.memory_bytes()         # implicit storage
```

`recursive_cauchy_eigen` = combinatorial `L = D - W`;
`eigen_normalized` = symmetric normalized `I - D^{-1/2} W D^{-1/2}`
(returns `global_deg` of the effective graph as 5th element).

## Install

```bash
pip install numpy scipy networkx numba threadpoolctl pymetis
```

pymetis is the default partitioner. Without it the code still runs via a
Fiedler-bisection fallback (with a RuntimeWarning), but partitions can be
badly unbalanced and benchmark numbers are not comparable.

## Benchmark

Defaults reproduce the paper's synthetic setup (Sec. 6.1): BA graphs,
depth 2 (4 subgraphs), k = 5, median of 3 runs, vs `numpy.linalg.eigh`.

```bash
# runtime vs n (Fig. 3)
OMP_NUM_THREADS=8 python -m cauchy_factorization.bench \
    --n-grid 4000 6000 8000 10000 14000 20000

# runtime vs cut size at fixed n (Fig. 4)
OMP_NUM_THREADS=8 python -m cauchy_factorization.bench \
    --n 8000 --k-grid 2 5 10 15 20 25 30 --no-dense

# normalized Laplacian
python -m cauchy_factorization.bench --n 8000 --laplacian norm
```

Flags: `--laplacian {comb,norm}`, `--depth`, `--target-cut`, `--n-grid`,
`--k-grid`, `--no-dense`, `--no-validate`, `--repeats`,
`--sparsifier {ss,degree}` (`degree` avoids the CG solves of the
effective-resistance sketch), `--n-sketches`. `--help` for the rest.

Measured tables are in the top-level README. They were produced on a
Xeon E5-2667 v4; the paper's experiments ran on an i9-9900K, so absolute
times differ from the paper while the scaling exponents match.

Output includes an itemized runtime (partition / sparsify / base_eig /
secular / z_compute), memory vs dense eigenvectors, and validation against
the effective Laplacian (eigenvalue error, orthogonality and eigen-residual
probes). First run compiles numba kernels (cached); time from the second run.

## Files

- `factorization.py` — combinatorial pipeline (`recursive_cauchy_eigen`)
- `normalized.py` — normalized pipeline (`eigen_normalized`)
- `recursive_decomp.py` — `RecursiveDecomp` + numba `Q^T x` kernels
- `cauchy_factor.py` — implicit `CauchyFactor`
- `graphs.py`, `bench.py`, `_metis_compat.py`
