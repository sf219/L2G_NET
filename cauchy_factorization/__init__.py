"""
Standalone Cauchy factorization of graph Laplacian eigendecompositions.

This package contains ONLY the recursive divide-and-conquer factorization
(L2G-Net companion code): partition -> sparsify cut edges -> factorize the
disconnected parts -> stitch with rank-one secular updates, storing each
update as an implicit orthogonal Cauchy factor. It does NOT contain the
HODLR fast-matvec machinery.

Entry points
------------
recursive_cauchy_eigen(G, depth, target_cut, ...)   combinatorial Laplacian
eigen_normalized(G, depth, target_cut, ...)         sym. normalized Laplacian

Both take a networkx graph and return (eigvals, decomp, stats, G_eff, ...),
where `decomp.qt_dot(x)` applies the exact transform Q^T x.

Run `python -m cauchy_factorization.bench --help` for the benchmark that
demonstrates the factorization speed-up over a dense eigendecomposition.
"""
from .cauchy_factor import CauchyFactor
from .recursive_decomp import RecursiveDecomp
from .factorization import recursive_cauchy_eigen
from .normalized import eigen_normalized

__all__ = ["CauchyFactor", "RecursiveDecomp",
           "recursive_cauchy_eigen", "eigen_normalized"]
