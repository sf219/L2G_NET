"""Synthetic graph generators and Laplacian helpers for the benchmark."""
from dataclasses import dataclass

import networkx as nx
import numpy as np
import scipy.sparse as sp


@dataclass
class GraphSpec:
    name: str
    G: nx.Graph

    @property
    def n(self):
        return self.G.number_of_nodes()

    @property
    def m(self):
        return self.G.number_of_edges()

    @property
    def avg_degree(self):
        return 2.0 * self.m / max(self.n, 1)


def _ensure_connected(G):
    if nx.is_connected(G):
        return G
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    G = G.subgraph(comps[0]).copy()
    return nx.convert_node_labels_to_integers(G, ordering="sorted")


def make_graph(kind, *, n=1000, seed=0, blocks=4, p_in=0.15, p_out=0.005,
               m_edges=3, p_er=None, path=None):
    """kind: sbm | ba | er | grid | file (weighted edge list u v [w])."""
    if kind == "sbm":
        sizes = [n // blocks] * blocks
        sizes[0] += n - sum(sizes)
        P = np.full((blocks, blocks), p_out)
        np.fill_diagonal(P, p_in)
        G = nx.stochastic_block_model(sizes, P.tolist(), seed=seed)
        G = nx.Graph(G)
        name = f"sbm_b{blocks}_pin{p_in}_pout{p_out}"
    elif kind == "ba":
        G = nx.barabasi_albert_graph(n, m_edges, seed=seed)
        name = f"ba_m{m_edges}"
    elif kind == "er":
        p = p_er if p_er is not None else 2.0 * np.log(n) / n
        G = nx.gnp_random_graph(n, p, seed=seed)
        name = f"er_p{p:.4g}"
    elif kind == "grid":
        side = int(round(np.sqrt(n)))
        G = nx.grid_2d_graph(side, side)
        G = nx.convert_node_labels_to_integers(G)
        name = f"grid_{side}x{side}"
    elif kind == "file":
        if path is None:
            raise ValueError("--path is required for kind=file")
        G = nx.read_weighted_edgelist(path, nodetype=int)
        name = f"file_{path}"
    else:
        raise ValueError(f"unknown graph kind {kind!r}")

    G = _ensure_connected(G)
    for _, _, d in G.edges(data=True):
        d.setdefault("weight", 1.0)
    return GraphSpec(name=name, G=G)


def combinatorial_laplacian_dense(G):
    n = G.number_of_nodes()
    return np.asarray(
        nx.laplacian_matrix(G, nodelist=range(n)).todense(), dtype=np.float64)


def normalized_laplacian_dense(G, global_deg):
    """I - D^{-1/2} W D^{-1/2} with the provided (global) degrees."""
    n = G.number_of_nodes()
    W = nx.adjacency_matrix(G, nodelist=range(n)).astype(np.float64)
    d = np.asarray(global_deg, dtype=np.float64)
    d_inv_sqrt = 1.0 / np.sqrt(np.maximum(d, 1e-300))
    Dm = sp.diags(d_inv_sqrt)
    return np.asarray((sp.eye(n) - Dm @ W @ Dm).todense())
