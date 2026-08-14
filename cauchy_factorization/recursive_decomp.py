import numpy as np
import numba as nb

# ============================================================================
# RECURSIVE NODE-TO-GLOBAL INDEX MAPPING
# ============================================================================

@nb.njit(cache=True)
def _apply_cauchy_factor_to_vector_zhat(d, roots, z_hat, givens, active_idx, x):
    """Apply Q^T to vector x using z_hat (Gu-Eisenstat)."""
    k = len(d)
    x_out = x.copy()
    
    n_givens = givens.shape[0]
    for g in range(n_givens):
        i_g = int(givens[g, 0])
        j_g = int(givens[g, 1])
        c_g = givens[g, 2]
        s_g = givens[g, 3]
        xi = x_out[i_g]
        xj = x_out[j_g]
        x_out[i_g] = c_g * xi + s_g * xj
        x_out[j_g] = -s_g * xi + c_g * xj
    
    x_active = np.empty(k, dtype=np.float64)
    for i in range(k):
        x_active[i] = x_out[active_idx[i]]
    
    zx = np.empty(k, dtype=np.float64)
    for i in range(k):
        zx[i] = z_hat[i] * x_active[i]
    
    for j in range(k):
        s = 0.0
        comp = 0.0
        for i in range(k):
            delta = d[i] - roots[j]
            if abs(delta) < 1e-300:
                if delta >= 0:
                    delta = 1e-300
                else:
                    delta = -1e-300
            term = zx[i] / delta
            y = term - comp
            t = s + y
            comp = (t - s) - y
            s = t
        x_out[active_idx[j]] = s
    
    return x_out


@nb.njit(cache=True)
def _apply_cauchy_factor_to_vector_znorms(d, roots, z, norms, givens, active_idx, x):
    """Apply Q^T to vector x using z/norms (classical formula)."""
    k = len(d)
    x_out = x.copy()
    
    n_givens = givens.shape[0]
    for g in range(n_givens):
        i_g = int(givens[g, 0])
        j_g = int(givens[g, 1])
        c_g = givens[g, 2]
        s_g = givens[g, 3]
        xi = x_out[i_g]
        xj = x_out[j_g]
        x_out[i_g] = c_g * xi + s_g * xj
        x_out[j_g] = -s_g * xi + c_g * xj
    
    x_active = np.empty(k, dtype=np.float64)
    for i in range(k):
        x_active[i] = x_out[active_idx[i]]
    
    for j in range(k):
        inv_norm = 1.0 / norms[j] if abs(norms[j]) > 1e-300 else 0.0
        s = 0.0
        comp = 0.0
        for i in range(k):
            delta = d[i] - roots[j]
            if abs(delta) < 1e-300:
                if delta >= 0:
                    delta = 1e-300
                else:
                    delta = -1e-300
            term = z[i] * x_active[i] / delta
            y = term - comp
            t = s + y
            comp = (t - s) - y
            s = t
        x_out[active_idx[j]] = s * inv_norm
    
    return x_out

def _apply_cf(cf, givens_arr, c_sorted):
    """Apply a CauchyFactor's Q^T to c_sorted."""
    if cf.use_zhat and (cf.z_hat is not None):
        return _apply_cauchy_factor_to_vector_zhat(
            cf.d, cf.roots, cf.z_hat, givens_arr, cf.active_idx, c_sorted
        )
    return _apply_cauchy_factor_to_vector_znorms(
        cf.d, cf.roots, cf.z, cf.norms, givens_arr, cf.active_idx, c_sorted
    )

def _build_givens_arr(givens_list):
    if not givens_list:
        return np.empty((0, 4), dtype=np.float64)
    arr = np.empty((len(givens_list), 4), dtype=np.float64)
    for gi, (i_g, j_g, c_g, s_g) in enumerate(givens_list):
        arr[gi] = [i_g, j_g, c_g, s_g]
    return arr

@nb.njit(cache=True)
def _apply_givens_batch(givens_data, future_z_mat):
    """Apply Givens rotations (rows of givens_data) to every column of future_z_mat.
    future_z_mat shape: (n_total, n_future).  Modified in-place."""
    n_givens = givens_data.shape[0]
    n_future = future_z_mat.shape[1]
    for g in range(n_givens):
        i_g = int(givens_data[g, 0])
        j_g = int(givens_data[g, 1])
        c_g = givens_data[g, 2]
        s_g = givens_data[g, 3]
        for fe in range(n_future):
            zi = future_z_mat[i_g, fe]
            zj = future_z_mat[j_g, fe]
            future_z_mat[i_g, fe] =  c_g * zi + s_g * zj
            future_z_mat[j_g, fe] = -s_g * zi + c_g * zj

@nb.njit(cache=True)
def _cauchy_matvec_batch_stable(z_hat, d, roots, X, out):
    """Apply Q^T to X using Gu-Eisenstat z_hat (no norms needed)."""
    k = d.size
    m = X.shape[1]
    for j in range(k):
        for col in range(m):
            acc = 0.0
            comp = 0.0
            for i in range(k):
                delta = d[i] - roots[j]
                if abs(delta) < 1e-300:
                    if delta >= 0:
                        delta = 1e-300
                    else:
                        delta = -1e-300
                term = z_hat[i] / delta * X[i, col]
                y = term - comp
                t = acc + y
                comp = (t - acc) - y
                acc = t
            out[j, col] = acc


@nb.njit(cache=True, parallel=True)
def _cauchy_matvec_batch_parallel_stable(z_hat, d, roots, X, out):
    """Parallel version of stable Q^T X using z_hat."""
    k = d.size
    m = X.shape[1]
    for j in nb.prange(k):
        for col in range(m):
            acc = 0.0
            comp = 0.0
            for i in range(k):
                delta = d[i] - roots[j]
                if abs(delta) < 1e-300:
                    if delta >= 0:
                        delta = 1e-300
                    else:
                        delta = -1e-300
                term = z_hat[i] / delta * X[i, col]
                y = term - comp
                t = acc + y
                comp = (t - acc) - y
                acc = t
            out[j, col] = acc

@nb.njit(cache=True)
def _cauchy_matvec_batch(z, d, roots, norms, X, out):
    k = d.size
    m = X.shape[1]
    for j in range(k):
        inv_norm = 1.0 / norms[j] if abs(norms[j]) > 1e-300 else 0.0
        for col in range(m):
            acc = 0.0
            comp = 0.0
            for i in range(k):
                delta = d[i] - roots[j]
                if abs(delta) < 1e-300:
                    delta = 1e-300 if delta >= 0 else -1e-300
                term = z[i] / delta * X[i, col]
                y = term - comp
                t = acc + y
                comp = (t - acc) - y
                acc = t
            out[j, col] = acc * inv_norm


@nb.njit(cache=True, parallel=True)
def _cauchy_matvec_batch_parallel(z, d, roots, norms, X, out):
    k = d.size
    m = X.shape[1]
    for j in nb.prange(k):
        inv_norm = 1.0 / norms[j] if abs(norms[j]) > 1e-300 else 0.0
        for col in range(m):
            acc = 0.0
            comp = 0.0
            for i in range(k):
                delta = d[i] - roots[j]
                if abs(delta) < 1e-300:
                    delta = 1e-300 if delta >= 0 else -1e-300
                term = z[i] / delta * X[i, col]
                y = term - comp
                t = acc + y
                comp = (t - acc) - y
                acc = t
            out[j, col] = acc * inv_norm

class RecursiveDecomp:
    """
    Stores a recursive Cauchy decomposition tree without materializing
    any eigenvector matrices (except at the leaves).
    """
    __slots__ = ['n', 'eigvals', 'is_leaf',
                 'leaf_eigvecs', 'leaf_nodes', 'leaf_node_map',
                 'left', 'right',
                 'part_0', 'part_1', 'part_0_set', 'part_1_set',
                 'cut_edges',
                 'cauchy_factors', 'sort_perm',
                 'global_to_local',
                 'resort_perms']
    
    def __init__(self):
        self.n = 0
        self.eigvals = None
        self.is_leaf = False
        self.leaf_eigvecs = None
        self.leaf_nodes = None
        self.leaf_node_map = None
        self.left = None
        self.right = None
        self.part_0 = None
        self.part_1 = None
        self.part_0_set = None
        self.part_1_set = None
        self.cut_edges = None
        self.cauchy_factors = None
        self.sort_perm = None
        self.global_to_local = None
        self.resort_perms = None

    def _apply_parent_chain_batch(self, C_batch):
        """
        C_batch: (n_total, n_nodes) — each column is one precomp vector.
        Returns (n_total, n_nodes) after applying sort_perm + all CauchyFactors.
        """
        # Apply sort_perm to all columns at once
        C = C_batch[self.sort_perm, :]   # one fancy-index op, not n_nodes separate ops
        
        for cf_idx, cf in enumerate(self.cauchy_factors):
            if len(cf.d) > 0:
                # Build givens array once
                givens_arr = _build_givens_arr(cf.givens)
                # Apply Givens to all columns — already batched via _apply_givens_batch
                if givens_arr.shape[0] > 0:
                    _apply_givens_batch(givens_arr, C)   # in-place, all cols
                # Cauchy matvec: replace active rows with Q^T * active_rows
                active = cf.active_idx
                X_active = C[active, :].copy()          # (k, n_nodes)
                out = np.empty_like(X_active)
                if cf.use_zhat and cf.z_hat is not None:
                    if len(active) > 200:
                        _cauchy_matvec_batch_parallel_stable(cf.z_hat, cf.d, cf.roots, X_active, out)
                    else:
                        _cauchy_matvec_batch_stable(cf.z_hat, cf.d, cf.roots, X_active, out)
                else:
                    if len(active) > 200:
                        _cauchy_matvec_batch_parallel(cf.z, cf.d, cf.roots, cf.norms, X_active, out)
                    else:
                        _cauchy_matvec_batch(cf.z, cf.d, cf.roots, cf.norms, X_active, out)
                C[active, :] = out
            else:
                # Deflated factor — apply Givens only
                for (i_g, j_g, c_g, s_g) in cf.givens:
                    ci = C[i_g, :].copy()
                    cj = C[j_g, :].copy()
                    C[i_g, :] = c_g * ci + s_g * cj
                    C[j_g, :] = -s_g * ci + c_g * cj
            
            if self.resort_perms is not None and cf_idx < len(self.resort_perms):
                rp = self.resort_perms[cf_idx]
                if rp is not None:
                    C = C[rp, :]
        
        return C

    def _apply_parent_chain(self, c_in_child_basis):
        """
        Apply only THIS node's Cauchy chain (sort_perm + cauchy_factors + resort_perms)
        to a vector already in the merged child eigenbasis [Q_left | Q_right].

        Avoids re-traversing child subtrees — caller provides the vector directly
        in the merged child eigenbasis (e.g., from precomp_child rows).

        Parameters
        ----------
        c_in_child_basis : ndarray, shape (n0 + n1,)
            Vector in the concatenated child eigenbasis order (left then right),
            NOT yet sorted.

        Returns
        -------
        ndarray, shape (n,)
            Vector in this node's sorted eigenbasis after all Cauchy factors.
        """
        c_sorted = c_in_child_basis[self.sort_perm].copy()

        for cf_idx, cf in enumerate(self.cauchy_factors):
            if len(cf.d) > 0:
                n_givens = len(cf.givens)
                if n_givens > 0:
                    givens_arr = np.empty((n_givens, 4), dtype=np.float64)
                    for gi, (i_g, j_g, c_g, s_g) in enumerate(cf.givens):
                        givens_arr[gi, 0] = i_g
                        givens_arr[gi, 1] = j_g
                        givens_arr[gi, 2] = c_g
                        givens_arr[gi, 3] = s_g
                else:
                    givens_arr = np.empty((0, 4), dtype=np.float64)
                c_sorted = _apply_cf(cf, givens_arr, c_sorted)
            else:
                for (i_g, j_g, c_g, s_g) in cf.givens:
                    ci = c_sorted[i_g]
                    cj = c_sorted[j_g]
                    c_sorted[i_g] = c_g * ci + s_g * cj
                    c_sorted[j_g] = -s_g * ci + c_g * cj

            if self.resort_perms is not None and cf_idx < len(self.resort_perms):
                rp = self.resort_perms[cf_idx]
                if rp is not None:
                    c_sorted = c_sorted[rp]

        return c_sorted

    def qt_dot_sparse(self, node_vals):
        """
        Compute Q^T v where v is sparse, given as a dict {node_id: value}.
        """
        if self.is_leaf:
            v_local = np.zeros(self.n, dtype=np.float64)
            for nd, val in node_vals.items():
                if nd in self.leaf_node_map:
                    v_local[self.leaf_node_map[nd]] = val
            return self.leaf_eigvecs.T @ v_local
        
        vals_left = {}
        vals_right = {}
        for nd, val in node_vals.items():
            if nd in self.part_0_set:
                vals_left[nd] = val
            elif nd in self.part_1_set:
                vals_right[nd] = val
        
        n0 = self.left.n
        n1 = self.right.n
        
        if vals_left:
            c0 = self.left.qt_dot_sparse(vals_left)
        else:
            c0 = np.zeros(n0, dtype=np.float64)
        
        if vals_right:
            c1 = self.right.qt_dot_sparse(vals_right)
        else:
            c1 = np.zeros(n1, dtype=np.float64)
        
        c_combined = np.empty(n0 + n1, dtype=np.float64)
        c_combined[:n0] = c0
        c_combined[n0:] = c1
        return self._apply_parent_chain(c_combined)
    
    def qt_dot(self, x):
        """
        Compute Q^T x where x is a dense array indexed by node ID (0..n-1).
        """
        if self.is_leaf:
            v_local = np.empty(self.n, dtype=np.float64)
            for i, nd in enumerate(self.leaf_nodes):
                v_local[i] = x[nd]
            return self.leaf_eigvecs.T @ v_local
        
        c0 = self.left.qt_dot(x)
        c1 = self.right.qt_dot(x)
        
        n0 = self.left.n
        c_combined = np.empty(n0 + self.right.n, dtype=np.float64)
        c_combined[:n0] = c0
        c_combined[n0:] = c1
        return self._apply_parent_chain(c_combined)
        
    def memory_bytes(self):
        if self.is_leaf:
            return self.leaf_eigvecs.nbytes + self.leaf_nodes.nbytes
        mem = 0
        mem += self.left.memory_bytes()
        mem += self.right.memory_bytes()
        for cf in self.cauchy_factors:
            mem += cf.memory_bytes()
        mem += self.sort_perm.nbytes
        mem += self.part_0.nbytes + self.part_1.nbytes
        if self.resort_perms is not None:
            for rp in self.resort_perms:
                if rp is not None:
                    mem += rp.nbytes
        return mem


