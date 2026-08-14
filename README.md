# L2G-Net: Local to Global GNNs via Cauchy Factorizations

**Paper [ICML-2026 Spotlight]:** [_**L2G-Net: Local to Global GNNs via Cauchy Factorizations**_](https://arxiv.org/abs/2602.18837)

**Project page:** [sf219.github.io/L2G_NET](https://sf219.github.io/L2G_NET/)

## Code

[`cauchy_factorization/`](cauchy_factorization/) — implementation of the
Cauchy factorization (combinatorial and normalized Laplacians), with a
benchmark reproducing the paper's runtime experiments:

```bash
OMP_NUM_THREADS=8 python -m cauchy_factorization.bench \
    --n-grid 4000 6000 8000 10000 14000 20000
```

See [`cauchy_factorization/README.md`](cauchy_factorization/README.md) for
flags, install, and validation details.

[`l2g_net/`](l2g_net/) — train L2G-Net from an exported Cauchy factorization
(notebook: [`L2G_Net_Minesweeper.ipynb`](l2g_net/L2G_Net_Minesweeper.ipynb)).
Full 10-split Minesweeper through this pipeline: test ROC AUC 97.28 +/- 0.37
(paper: 97.50 +/- 0.4).

## Runtime vs graph size (paper Fig. 3 setup)

Cauchy factorization (CF) vs dense eigendecomposition (`numpy.linalg.eigh`).
BA graphs, depth 2, k = 5, median of 3 runs; Xeon E5-2667 v4, 8 threads,
pymetis. Scaling matches the paper (CF ~ n^2.0, ED ~ n^2.9).

| n | m | CF (s) | ED (s) | speedup | err_s |
|---|---|---|---|---|---|
| 4000 | 11991 | 2.49 | 3.63 | 1.46x | 2.3e-11 |
| 6000 | 17991 | 5.01 | 12.09 | 2.41x | 1.3e-11 |
| 8000 | 23991 | 8.50 | 27.58 | 3.24x | 3.0e-11 |
| 10000 | 29991 | 13.25 | 51.97 | 3.92x | 3.2e-10 |
| 14000 | 41991 | 27.78 | 137.93 | 4.97x | 7.3e-11 |
| 20000 | 59991 | 63.25 | 384.70 | 6.08x | 4.1e-11 |

## Runtime vs cut size (paper Fig. 4 setup)

CF runtime against the number of bridge edges k at fixed n = 8000 (BA
graph, depth 2, median of 3 runs; same machine and settings as above).

| k | CF (s) | err_s |
|---|---|---|
| 2 | 5.82 | 5.5e-12 |
| 5 | 8.44 | 3.0e-11 |
| 10 | 13.07 | 1.1e-10 |
| 15 | 20.14 | 8.4e-10 |
| 20 | 29.28 | 2.7e-10 |
| 25 | 41.21 | 3.6e-10 |
| 30 | 54.84 | 3.2e-10 |

