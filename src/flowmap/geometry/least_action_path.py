from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from typing import Dict


class LagrangianPathOptimizer:
    """
    Minimum-action path computation for FlowMap embeddings.

    This class computes transition trajectories under the action

    .. math::

        S[x] = \\frac{1}{2D} \\int ||\\dot{x} - v(x)||^2 dt
               + \\lambda \\int ||\\dot{x}||^2 dt

    where:

    - :math:`v(x)` is the vector field in embedding space
    - :math:`D` is the diffusion constant
    - :math:`\\lambda` is an arc-length regularization weight

    The optimizer proceeds in two stages:

    1. Initialize a path via Dijkstra on a kNN graph
    2. Refine the path by gradient descent on the action functional

    Notes
    -----
    The graph used for initialization can be built in either:

    - embedding space: ``(X_emb, V_emb)``
    - original space: ``(X, V)``

    However, the final action optimization is always performed in
    embedding space, since ``spline_vf`` is defined there.
    """

    def __init__(self, embedding, D: float = 1.0, lam: float = 0.0):
        """
        Parameters
        ----------
        embedding : VectorFieldEmbedder
            Must expose:

            - X_emb : ndarray of shape (N, d_emb)
            - V_emb : ndarray of shape (N, d_emb)
            - X : ndarray of shape (N, d_orig)
            - V : ndarray of shape (N, d_orig)
            - spline_vf : spline-like vector field on embedding space
        D : float, default=1.0
            Diffusion constant.
        lam : float, default=0.0
            Arc-length regularization weight.
        """
        self.embedding = embedding

        self.X_emb = embedding.X_emb
        self.V_emb = embedding.V_emb
        self.X_orig = embedding.X
        self.V_orig = embedding.V
        self.spline_vf = embedding.spline_vf

        self.D = D
        self.lam = lam

    # ------------------------------------------------------------------
    # Action functional in embedding space
    # ------------------------------------------------------------------

    @staticmethod
    def _dt_uniform(n_segments: int) -> float:
        return 1.0 / max(1, n_segments)

    def _action(self, path: np.ndarray, dt: float) -> float:
        segs = path[1:] - path[:-1]
        mids = 0.5 * (path[1:] + path[:-1])

        vf = self.spline_vf.predict(mids)
        residual = segs / dt - vf

        action = 0.5 / self.D * np.sum(np.sum(residual**2, axis=1) * dt)

        if self.lam > 0:
            action += self.lam * np.sum(np.sum(segs**2, axis=1))

        return float(action)

    def _grad_action(self, path: np.ndarray, dt: float) -> np.ndarray:
        n = len(path) - 1
        grad = np.zeros_like(path)

        segs = path[1:] - path[:-1]
        mids = 0.5 * (path[1:] + path[:-1])

        vf = self.spline_vf.predict(mids)
        jac = self.spline_vf.compute_jacobians(mids)

        residual = segs / dt - vf

        for m in range(1, n):
            term_main = residual[m - 1] - residual[m]
            term_jac = jac[m - 1].T @ residual[m - 1] + jac[m].T @ residual[m]

            grad[m] = (term_main - 0.5 * dt * term_jac) / self.D

            if self.lam > 0:
                grad[m] += 2 * self.lam * (
                    (path[m] - path[m - 1]) - (path[m + 1] - path[m])
                )

        return grad

    # ------------------------------------------------------------------
    # Optimization in embedding space
    # ------------------------------------------------------------------

    def _optimize(self, path: np.ndarray, lr: float, iters: int):
        path = path.copy()
        dt = self._dt_uniform(len(path) - 1)

        for _ in range(iters):
            grad = self._grad_action(path, dt)
            path[1:-1] -= lr * grad[1:-1]

        return path, dt

    # ------------------------------------------------------------------
    # Graph-based initialization
    # ------------------------------------------------------------------

    @staticmethod
    def _resample(points: np.ndarray, n_segments: int) -> np.ndarray:
        if len(points) == 1:
            return np.repeat(points, n_segments + 1, axis=0)

        segs = points[1:] - points[:-1]
        lens = np.linalg.norm(segs, axis=1)
        cum = np.r_[0.0, np.cumsum(lens)]
        total = cum[-1]

        if total == 0:
            return np.repeat(points[:1], n_segments + 1, axis=0)

        targets = np.linspace(0.0, total, n_segments + 1)
        out = np.zeros((n_segments + 1, points.shape[1]), dtype=points.dtype)

        j = 0
        for i, s in enumerate(targets):
            while j < len(cum) - 2 and cum[j + 1] < s:
                j += 1

            denom = cum[j + 1] - cum[j]
            t = 0.0 if denom == 0 else (s - cum[j]) / denom
            out[i] = points[j] + t * (points[j + 1] - points[j])

        return out

    def _get_graph_data(self, distance_mode: str):
        if distance_mode == "embed":
            return self.X_emb, self.V_emb
        if distance_mode == "orig":
            return self.X_orig, self.V_orig
        raise ValueError(
            f"Unknown distance_mode={distance_mode!r}. Expected 'embed' or 'orig'."
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_path(
        self,
        start: np.ndarray,
        end: np.ndarray,
        *,
        distance_mode: str = "embed",
        subsample_n: int = 4000,
        k: int = 20,
        alpha: float = 0.0,
        n_segments: int = 100,
        lr: float = 5e-3,
        iters: int = 300,
    ) -> Dict:
        """
        Compute a minimum-action transition path between two states.

        Parameters
        ----------
        start : ndarray of shape (d_emb,)
            Start coordinate in embedding space.
        end : ndarray of shape (d_emb,)
            End coordinate in embedding space.
        distance_mode : {"embed", "orig"}, default="embed"
            Space used to construct the kNN graph for path initialization.

            - ``"embed"`` uses ``(X_emb, V_emb)``
            - ``"orig"`` uses ``(X, V)``

            In both cases, the refined path is optimized in embedding space.
        subsample_n : int, default=4000
            Number of cells used to build the graph.
        k : int, default=20
            Number of nearest neighbors in the graph.
        alpha : float, default=0.0
            Velocity-alignment penalty weight used during graph construction.
            Larger values penalize edges that disagree with local velocity.
        n_segments : int, default=100
            Number of segments in the resampled path.
        lr : float, default=5e-3
            Gradient descent learning rate for path refinement.
        iters : int, default=300
            Number of optimization iterations.

        Returns
        -------
        dict
            Dictionary with keys:

            - ``path_init`` : ndarray of shape (n_segments + 1, d_emb)
              Initial path in embedding space.
            - ``path_refined`` : ndarray of shape (n_segments + 1, d_emb)
              Refined minimum-action path in embedding space.
            - ``dt`` : float
              Uniform discretization step.
        """
        X_graph, V_graph = self._get_graph_data(distance_mode)

        n_cells = X_graph.shape[0]
        idx_sub = np.random.choice(n_cells, min(subsample_n, n_cells), replace=False)

        # start/end are given in embedding space, so find anchor cell indices there
        start_idx = int(np.argmin(np.linalg.norm(self.X_emb - start, axis=1)))
        end_idx = int(np.argmin(np.linalg.norm(self.X_emb - end, axis=1)))

        # avoid duplicates if start/end are already in the subsample
        idx_sub = idx_sub[(idx_sub != start_idx) & (idx_sub != end_idx)]

        # graph-space coordinates / velocities, dimension-matched
        graph_indices = np.r_[start_idx, end_idx, idx_sub]
        X_sub = X_graph[graph_indices]
        V_sub = V_graph[graph_indices]

        # kNN graph
        n_nodes = X_sub.shape[0]
        if n_nodes < 2:
            raise ValueError("Not enough nodes to construct a graph.")

        n_neighbors = min(k + 1, n_nodes)
        nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(X_sub)
        dists, idxs = nbrs.kneighbors(X_sub)

        rows, cols, weights = [], [], []

        for i in range(n_nodes):
            for j, dist in zip(idxs[i, 1:], dists[i, 1:]):
                weight = dist

                if alpha > 0:
                    v = V_sub[i]
                    direction = X_sub[j] - X_sub[i]

                    v_norm = np.linalg.norm(v)
                    d_norm = np.linalg.norm(direction)

                    if v_norm > 0 and d_norm > 0:
                        cos_sim = np.dot(v, direction) / (v_norm * d_norm)
                    else:
                        cos_sim = 0.0

                    weight *= 1.0 + alpha * (1.0 - cos_sim)

                rows.append(i)
                cols.append(j)
                weights.append(weight)

        W = csr_matrix((weights, (rows, cols)), shape=(n_nodes, n_nodes))

        _, predecessors = shortest_path(W, directed=False, return_predecessors=True)

        # recover path from node 0 (start) to node 1 (end)
        path_node_ids = []
        cur = 1
        while cur != 0 and cur != -9999:
            path_node_ids.append(cur)
            cur = predecessors[0, cur]

        if cur == -9999:
            raise ValueError(
                "No path found between start and end in the kNN graph. "
                "Try increasing subsample_n or k."
            )

        path_node_ids.append(0)
        path_node_ids = path_node_ids[::-1]

        # map graph path back to embedding coordinates for refinement
        path_cell_indices = graph_indices[path_node_ids]
        path_coords_emb = self.X_emb[path_cell_indices]

        path_init = self._resample(path_coords_emb, n_segments)
        path_refined, dt = self._optimize(path_init, lr, iters)

        return {
            "path_init": path_init,
            "path_refined": path_refined,
            "dt": dt,
        }
        