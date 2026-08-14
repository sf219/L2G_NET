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
flags, measured tables, and validation details.
