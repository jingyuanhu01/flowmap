from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional

import joblib
import numpy as np
import pandas as pd

from .dataset import FlowMapDataset


def save_dataset(data: FlowMapDataset, path: str | Path, **joblib_kwargs) -> None:
    """
    Save a FlowMapDataset as an ordinary joblib file.
    """

    joblib.dump(data, path, **joblib_kwargs)


def load_dataset(path: str | Path) -> FlowMapDataset:
    """
    Load a FlowMapDataset saved with save_dataset.
    """

    data = joblib.load(path)
    if not isinstance(data, FlowMapDataset):
        raise TypeError(
            f"Expected FlowMapDataset in {path!s}; got {type(data).__name__}."
        )
    return data


def save_embedder(embedder: Any, path: str | Path, **joblib_kwargs) -> None:
    """
    Save a fitted VectorFieldEmbedder or compatible object with joblib.
    """

    joblib.dump(embedder, path, **joblib_kwargs)


def load_embedder(path: str | Path) -> Any:
    """
    Load a fitted VectorFieldEmbedder or compatible object saved with joblib.
    """

    return joblib.load(path)


def from_anndata(
    adata,
    *,
    X_layer: Optional[str] = None,
    V_layer: str = "velocity",
    X_obsm: Optional[str] = None,
    V_obsm: Optional[str] = None,
    X_emb_obsm: Optional[str] = None,
    V_emb_obsm: Optional[str] = None,
    embedder=None,
    copy_layers: Iterable[str] | str | bool | None = None,
    copy_obsm: Iterable[str] | str | bool | None = None,
    copy_varm: Iterable[str] | str | bool | None = None,
    copy_uns: bool = False,
    metadata: Optional[dict[str, Any]] = None,
) -> FlowMapDataset:
    """
    Build a FlowMapDataset from an AnnData-like object.
    """

    X = _matrix_from_anndata(adata, layer=X_layer, obsm=X_obsm, name="X")
    V = _matrix_from_anndata(adata, layer=V_layer, obsm=V_obsm, name="V")

    obs = adata.obs.copy()
    var = adata.var.copy()

    layers = {
        key: _as_array(adata.layers[key])
        for key in _resolve_keys(copy_layers, adata.layers.keys())
    }
    obsm = {
        key: _as_array(adata.obsm[key])
        for key in _resolve_keys(copy_obsm, adata.obsm.keys())
    }
    varm = {
        key: _as_array(adata.varm[key])
        for key in _resolve_keys(copy_varm, adata.varm.keys())
    }

    if X_emb_obsm is not None:
        obsm["X_flowmap"] = _as_array(adata.obsm[X_emb_obsm])
    if V_emb_obsm is not None:
        obsm["V_flowmap"] = _as_array(adata.obsm[V_emb_obsm])

    uns = dict(adata.uns) if copy_uns else {}
    meta = {
        "schema_version": 1,
        "source_format": "anndata",
    }
    if metadata:
        meta.update(metadata)

    data = FlowMapDataset(
        X=X,
        V=V,
        embedder=embedder,
        obs=obs,
        var=var,
        layers=layers,
        obsm=obsm,
        varm=varm,
        uns=uns,
        metadata=meta,
    )

    if data.embedder is not None and hasattr(data.embedder, "gene_names"):
        data.embedder.gene_names = data.var_names

    return data


def from_embedder(
    embedder,
    *,
    adata=None,
    copy_layers: Iterable[str] | str | bool | None = None,
    copy_obsm: Iterable[str] | str | bool | None = None,
    copy_varm: Iterable[str] | str | bool | None = None,
    copy_uns: bool = False,
    metadata: Optional[dict[str, Any]] = None,
) -> FlowMapDataset:
    """
    Build a FlowMapDataset around an existing fitted VectorFieldEmbedder.
    """

    if not hasattr(embedder, "X") or not hasattr(embedder, "V"):
        raise ValueError("embedder must expose X and V arrays.")

    obs = None
    var = None
    layers = {}
    obsm = {}
    varm = {}
    uns = {}
    meta = {
        "schema_version": 1,
        "source_format": "embedder",
    }

    if adata is not None:
        obs = adata.obs.copy()
        var = adata.var.copy()
        layers = {
            key: _as_array(adata.layers[key])
            for key in _resolve_keys(copy_layers, adata.layers.keys())
        }
        obsm = {
            key: _as_array(adata.obsm[key])
            for key in _resolve_keys(copy_obsm, adata.obsm.keys())
        }
        varm = {
            key: _as_array(adata.varm[key])
            for key in _resolve_keys(copy_varm, adata.varm.keys())
        }
        uns = dict(adata.uns) if copy_uns else {}
        meta["source_format"] = "embedder+anndata"

    if hasattr(embedder, "X_emb") and embedder.X_emb is not None:
        obsm.setdefault("X_flowmap", embedder.X_emb)
    if hasattr(embedder, "V_emb") and embedder.V_emb is not None:
        obsm.setdefault("V_flowmap", embedder.V_emb)

    if metadata:
        meta.update(metadata)

    data = FlowMapDataset(
        X=embedder.X,
        V=embedder.V,
        embedder=embedder,
        obs=obs,
        var=var,
        layers=layers,
        obsm=obsm,
        varm=varm,
        uns=uns,
        metadata=meta,
    )

    if hasattr(data.embedder, "gene_names"):
        data.embedder.gene_names = data.var_names

    return data


def to_anndata(data: FlowMapDataset):
    """
    Convert a FlowMapDataset to AnnData.
    """

    try:
        import anndata as ad
    except ImportError as exc:
        raise ImportError("to_anndata requires the anndata package.") from exc

    flowmap_metadata = dict(data.metadata)
    if len(data.var) == data.n_features:
        var = data.var.copy()
    else:
        var = pd.DataFrame(index=[f"feature_{i}" for i in range(data.n_features)])
        flowmap_metadata["var_names"] = data.var.index.astype(str).to_numpy()
        flowmap_metadata["var"] = data.var.copy()

    adata = ad.AnnData(X=data.X, obs=data.obs.copy(), var=var)

    moved_layers = {}
    for key, value in data.layers.items():
        if value.shape == data.X.shape:
            adata.layers[key] = value
        elif value.shape[0] == data.n_obs:
            obsm_key = f"flowmap_layer_{key}"
            adata.obsm[obsm_key] = value
            moved_layers[key] = obsm_key
        else:
            flowmap_metadata.setdefault("layers", {})[key] = value

    for key, value in data.obsm.items():
        adata.obsm[key] = value

    moved_varm = {}
    for key, value in data.varm.items():
        if value.shape[0] == data.n_features:
            adata.varm[key] = value
        else:
            flowmap_metadata.setdefault("varm", {})[key] = value
            moved_varm[key] = "flowmap.varm"

    if moved_layers:
        flowmap_metadata["layers_stored_in_obsm"] = moved_layers
    if moved_varm:
        flowmap_metadata["varm_stored_in_uns"] = moved_varm

    if data.X_emb is not None:
        adata.obsm.setdefault("X_flowmap", data.X_emb)
    if data.V_emb is not None:
        adata.obsm.setdefault("V_flowmap", data.V_emb)

    adata.uns.update(data.uns)
    adata.uns["flowmap"] = flowmap_metadata

    return adata


def _matrix_from_anndata(
    adata,
    *,
    layer: Optional[str],
    obsm: Optional[str],
    name: str,
) -> np.ndarray:
    if layer is not None and obsm is not None:
        raise ValueError(f"Specify only one of {name}_layer or {name}_obsm.")
    if layer is not None:
        return _as_array(adata.layers[layer])
    if obsm is not None:
        return _as_array(adata.obsm[obsm])
    if name == "X":
        return _as_array(adata.X)
    raise ValueError(f"Specify a layer or obsm key for {name}.")


def _resolve_keys(request, available) -> list[str]:
    available = list(available)
    if request is None or request is False:
        return []
    if request is True:
        return available
    if isinstance(request, str):
        return [request]
    return list(request)


def _as_array(value: Any) -> np.ndarray:
    if hasattr(value, "toarray"):
        value = value.toarray()
    return np.asarray(value)
