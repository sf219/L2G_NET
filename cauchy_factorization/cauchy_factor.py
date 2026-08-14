import numpy as np

# ============================================================================
# CAUCHY MATRIX — implicit representation
# ============================================================================

class CauchyFactor:
    __slots__ = ['d', 'roots', 'z', 'z_hat', 'norms', 'active_idx',
                 'givens', 'perm', 'n_global', 'rho', 'use_zhat']

    def __init__(self, d, roots, z, norms, active_idx, givens, perm, n_global,
                 rho=1.0, z_hat=None, use_zhat=False):
        self.d = np.asarray(d, dtype=np.float64).copy()
        self.roots = np.asarray(roots, dtype=np.float64).copy()
        self.z = np.asarray(z, dtype=np.float64).copy()
        self.norms = np.asarray(norms, dtype=np.float64).copy()
        self.active_idx = np.asarray(active_idx, dtype=np.int64).copy()
        self.givens = list(givens) if givens else []
        self.perm = perm
        self.n_global = n_global
        self.rho = rho
        self.use_zhat = bool(use_zhat)
        self.z_hat = np.asarray(z_hat, dtype=np.float64).copy() if z_hat is not None else None

    def memory_bytes(self):
        mem = 0
        mem += self.d.nbytes
        mem += self.roots.nbytes
        mem += self.z.nbytes
        mem += self.norms.nbytes
        mem += self.active_idx.nbytes
        if self.z_hat is not None:
            mem += self.z_hat.nbytes
        mem += len(self.givens) * 4 * 8
        return mem
