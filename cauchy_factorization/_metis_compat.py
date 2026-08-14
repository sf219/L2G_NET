"""
pymetis compatibility layer.

If the real pymetis is installed it is used verbatim. Otherwise a pure
scipy fallback provides 2-way spectral (Fiedler-vector) bisection with the
same `part_graph(nparts, adjacency=...)` interface. Partitions differ from
METIS, so cut sizes (and therefore exact timings) are only reproducible
across machines when pymetis is installed.
"""
import types

import numpy as np
import scipy.sparse as sp

try:
    import pymetis  # noqa: F401
except ImportError:
    def _part_graph(nparts, adjacency=None, **kwargs):
        n = len(adjacency)
        if adjacency is None or n == 0:
            return 0, [0] * n

        def count_cuts(membership):
            cuts = 0
            for i, nbrs in enumerate(adjacency):
                for j in nbrs:
                    if membership[i] != membership[j]:
                        cuts += 1
            return cuts // 2

        if int(nparts) != 2:
            membership = [i % int(nparts) for i in range(n)]
            return count_cuts(membership), membership

        try:
            rows, cols = [], []
            for i, nbrs in enumerate(adjacency):
                for j in nbrs:
                    rows.append(i)
                    cols.append(j)
            data = np.ones(len(rows), dtype=np.float64)
            A = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
            A = ((A + A.T) > 0).astype(float)

            from scipy.sparse.linalg import eigsh
            deg = np.array(A.sum(axis=1)).ravel()
            L = sp.diags(deg) - A
            vals, vecs = eigsh(L, k=2, which='SM', tol=1e-2)
            fiedler = vecs[:, 1]
            membership = [1 if v > 0 else 0 for v in fiedler]
            if sum(membership) in (0, n):
                membership = [i % 2 for i in range(n)]
            return count_cuts(membership), membership
        except Exception:
            membership = [i % 2 for i in range(n)]
            return count_cuts(membership), membership

    pymetis = types.SimpleNamespace(part_graph=_part_graph)
