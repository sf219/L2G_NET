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

## Runtime vs graph size (paper Fig. 3 setup)

Cauchy factorization (CF) vs dense eigendecomposition (`numpy.linalg.eigh`).
BA graphs, depth 2, k = 5, median of 3 runs; Xeon E5-2667 v4, 8 threads,
pymetis. Scaling matches the paper (CF ~ n^2.0, ED ~ n^2.9); absolute times
differ from the paper's i9-9900K.

| n | m | CF (s) | ED (s) | speedup | err_s |
|---|---|---|---|---|---|
| 4000 | 11991 | 2.49 | 3.63 | 1.46x | 2.3e-11 |
| 6000 | 17991 | 5.01 | 12.09 | 2.41x | 1.3e-11 |
| 8000 | 23991 | 8.50 | 27.58 | 3.24x | 3.0e-11 |
| 10000 | 29991 | 13.25 | 51.97 | 3.92x | 3.2e-10 |
| 14000 | 41991 | 27.78 | 137.93 | 4.97x | 7.3e-11 |
| 20000 | 59991 | 63.25 | 384.70 | 6.08x | 4.1e-11 |

