from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd


@dataclass
class FlowMapDataset:
    """
    Lightweight container for FlowMap data, metadata, and an optional embedder.

    This object is intentionally AnnData-like but much smaller. It exists to
    keep matrices, cell metadata, gene metadata, and a fitted FlowMap embedder
    together without changing the VectorFieldEmbedder API.
    """

    X: np.ndarray
    V: np.ndarray
    embedder: Optional[Any] = None
    obs: Optional[pd.DataFrame] = None
    var: Optional[pd.DataFrame] = None
    layers: dict[str, np.ndarray] = field(default_factory=dict)
    obsm: dict[str, np.ndarray] = field(default_factory=dict)
    varm: dict[str, np.ndarray] = field(default_factory=dict)
    uns: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.X = _as_array(self.X)
        self.V = _as_array(self.V)

        if self.X.shape[0] != self.V.shape[0]:
            raise ValueError(
                "X and V must have the same number of cells. "
                f"Got X={self.X.shape}, V={self.V.shape}."
            )

        if self.obs is None:
            self.obs = pd.DataFrame(index=[f"cell_{i}" for i in range(self.n_obs)])
        else:
            self.obs = self.obs.copy()

        if self.var is None:
            self.var = pd.DataFrame(index=[f"feature_{i}" for i in range(self.n_features)])
        else:
            self.var = self.var.copy()

        if len(self.obs) != self.n_obs:
            raise ValueError(
                "obs must have one row per cell. "
                f"Got obs={len(self.obs)}, n_obs={self.n_obs}."
            )

        layers = {} if self.layers is None else self.layers
        obsm = {} if self.obsm is None else self.obsm
        varm = {} if self.varm is None else self.varm
        uns = {} if self.uns is None else self.uns
        metadata = {} if self.metadata is None else self.metadata

        self.layers = {key: _as_array(value) for key, value in layers.items()}
        self.obsm = {key: _as_array(value) for key, value in obsm.items()}
        self.varm = {key: _as_array(value) for key, value in varm.items()}
        self.uns = dict(uns)
        self.metadata = dict(metadata)

    @property
    def n_obs(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_vars(self) -> int:
        return int(len(self.var))

    @property
    def n_features(self) -> int:
        return int(self.X.shape[1])

    @property
    def obs_names(self) -> np.ndarray:
        return self.obs.index.to_numpy()

    @property
    def var_names(self) -> np.ndarray:
        return self.var.index.to_numpy()

    @property
    def gene_names(self) -> np.ndarray:
        return self.var_names

    @property
    def X_emb(self) -> Optional[np.ndarray]:
        if self.embedder is not None and hasattr(self.embedder, "X_emb"):
            return self.embedder.X_emb
        return self.obsm.get("X_flowmap")

    @property
    def V_emb(self) -> Optional[np.ndarray]:
        if self.embedder is not None and hasattr(self.embedder, "V_emb"):
            return self.embedder.V_emb
        return self.obsm.get("V_flowmap")

    def __repr__(self) -> str:
        lines = [
            (
                "FlowMapDataset object with "
                f"n_obs x n_features = {self.n_obs} x {self.n_features}"
            ),
            f"    X: {self.X.shape}, V: {self.V.shape}",
        ]

        if self.n_vars != self.n_features:
            lines.append(f"    var: {self.n_vars} variables")

        if self.embedder is not None:
            lines.append(f"    embedder: {type(self.embedder).__name__}")

        for label, mapping in [
            ("obs", self.obs),
            ("var", self.var),
            ("layers", self.layers),
            ("obsm", self.obsm),
            ("varm", self.varm),
            ("uns", self.uns),
            ("metadata", self.metadata),
        ]:
            keys = _format_keys(mapping)
            if keys:
                lines.append(f"    {label}: {keys}")

        return "\n".join(lines)

    def __str__(self) -> str:
        return self.__repr__()

    def attach_embedder(self, embedder: Any) -> "FlowMapDataset":
        self.embedder = embedder
        if hasattr(embedder, "gene_names"):
            embedder.gene_names = self.var_names
        return self


def _format_keys(value: Any) -> str:
    if isinstance(value, pd.DataFrame):
        keys = list(value.columns)
    elif isinstance(value, dict):
        keys = list(value.keys())
    else:
        return ""

    if not keys:
        return ""
    return ", ".join(repr(str(key)) for key in keys)


def _as_array(value: Any) -> np.ndarray:
    if hasattr(value, "toarray"):
        value = value.toarray()
    return np.asarray(value)
