# L2G-Net

Project page for **L2G-Net** (see `index.html` / GitHub Pages).

## Code

[`cauchy_factorization/`](cauchy_factorization/) — standalone implementation
of the recursive Cauchy factorization of graph Laplacian eigendecompositions
(combinatorial and symmetric-normalized), with a benchmark demonstrating the
speed-up over a dense eigendecomposition:

```bash
OMP_NUM_THREADS=8 python -m cauchy_factorization.bench --kind sbm --n 4096
```

See [`cauchy_factorization/README.md`](cauchy_factorization/README.md) for
the method summary, flags, and validation details.
