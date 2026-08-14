# L2G-Net on the heterophilous benchmarks

Notebook walkthrough: [`L2G_Net_Minesweeper.ipynb`](L2G_Net_Minesweeper.ipynb).

Training code that consumes a Cauchy factorization produced by
[`cauchy_factorization/`](../cauchy_factorization/). Two stages:

1. `export_factorization.py` — factorize the graph (depth 1: two subgraphs
   plus Cauchy factors for the sparsified cut) and export the spectral
   tensors. Needs only numpy/scipy/networkx/numba/pymetis.
2. `train.py` — train L2G-Net on those tensors. Needs
   torch (CUDA), dgl, pyyaml, tqdm, scikit-learn, matplotlib.

## Step-by-step (Minesweeper)

**1. Get the dataset** (Platonov et al., 2023 format):

```bash
mkdir -p data/minesweeper/raw
wget -O data/minesweeper/raw/minesweeper.npz \
  https://github.com/yandex-research/heterophilous-graphs/raw/main/data/minesweeper.npz
```

**2. Export the factorization** (the dense `basis_cauchy` materialization dominates; ~10-50 min on CPU at n=10000, scaling with target-cut):

```bash
python export_factorization.py \
    --npz data/minesweeper/raw/minesweeper.npz \
    --target-cut 1 --laplacian norm \
    --out minesweeper_k1.npz --validate
```

`--validate` probe-checks the exported `basis_cauchy` against the
factorization's exact transform (expect rel err ~1e-15).

**3. Train** (10 data splits, ~10 min each on a single GPU):

```bash
python train.py --dataset minesweeper --factorization minesweeper_k1.npz \
    --model SGWT --device cuda:0
```

The final lines report test ROC AUC mean ± std over the 10 splits;
the paper (Table 2) reports 97.50 ± 0.4 on Minesweeper.

## Other datasets

`roman-empire`, `amazon-ratings`, `tolokers` follow the same pattern:
download the npz into `data/<name>/raw/`, export, train with
`--dataset <name>`. See `train.py --help` for hyperparameters.

## What the export contains

`idx_S/idx_T` (subgraph node indices), `U_S/U_T` (local GFT bases),
`eig_S/eig_T`, `eigvals` (global spectrum of the effective Laplacian),
and `basis_cauchy` — the N x N product of the Cauchy factors mapping
block-local spectral coefficients to global ones (Eq. 10 of the paper).
`train.py` never recomputes any spectral quantity.
