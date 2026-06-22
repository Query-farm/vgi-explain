"""Consume the self-contained model BLOB produced by vgi-sklearn / vgi-xgboost.

vgi-explain does not train models; it interprets models that other workers fit.
Those workers serialize a fitted estimator together with its metadata into one
self-describing BLOB so the model can flow through SQL as a single value:

    SET VARIABLE m = (SELECT model FROM sklearn.fit(...));
    SELECT * FROM explain.shap_values((SELECT ...), model := getvariable('m'));

This module replicates that exact BLOB format (it is intentionally byte-for-byte
compatible with ``vgi_sklearn.registry``) so a BLOB packed by vgi-sklearn or
vgi-xgboost unpacks here unchanged.

BLOB layout: 4-byte big-endian metadata-JSON length || metadata JSON || skops
bytes. Estimators are serialized with skops (not pickle): loading reconstructs
only a known set of types instead of executing arbitrary code, and we further
restrict the trusted set to the scikit-learn / numpy / scipy / xgboost
namespaces so a crafted artifact cannot smuggle in an arbitrary callable.
"""

from __future__ import annotations

import json
import struct
from dataclasses import asdict, dataclass, field
from typing import Any

import skops.io as sio

# Trusted namespaces for skops safe-load. vgi-sklearn restricts to
# sklearn/numpy/scipy; we additionally trust xgboost so a model BLOB packed by
# vgi-xgboost (which embeds an ``xgboost.sklearn`` estimator) also unpacks.
_TRUSTED_PREFIXES = (
    "sklearn.",
    "numpy.",
    "numpy",
    "scipy.",
    "scipy",
    "xgboost.",
    "xgboost",
)


class UntrustedModelError(ValueError):
    """Raised when a serialized model contains types outside the trusted namespaces."""


class InvalidModelBlobError(ValueError):
    """Raised when a value is not a well-formed model BLOB (too short / truncated / garbage)."""


def _skops_dumps(estimator: Any) -> bytes:
    return sio.dumps(estimator)


def _skops_loads(data: bytes) -> Any:
    """Safely load a skops-serialized estimator, trusting only known ML namespaces."""
    try:
        untrusted = sio.get_untrusted_types(data=data)
    except Exception as exc:  # noqa: BLE001 - skops raises a variety of types on garbage
        raise InvalidModelBlobError(f"not a valid model BLOB: {exc}") from exc
    disallowed = [t for t in untrusted if not t.startswith(_TRUSTED_PREFIXES)]
    if disallowed:
        raise UntrustedModelError(
            f"refusing to load model containing untrusted type(s): {', '.join(disallowed)}"
        )
    try:
        return sio.loads(data, trusted=untrusted)
    except Exception as exc:  # noqa: BLE001
        raise InvalidModelBlobError(f"not a valid model BLOB: {exc}") from exc


@dataclass(kw_only=True)
class ModelMetadata:
    """Everything needed to score / interpret a model. Mirrors vgi-sklearn's metadata."""

    name: str
    estimator: str
    task: str  # "classification" | "regression"
    target: str
    feature_names: list[str]
    params: dict[str, Any] = field(default_factory=dict)
    classes: list[Any] | None = None
    n_samples: int = 0
    n_features: int = 0
    train_score: float | None = None
    sklearn_version: str = ""
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ModelMetadata:
        known = {f for f in cls.__dataclass_fields__}  # noqa: C416
        return cls(**{k: v for k, v in d.items() if k in known})


def pack_model(estimator: Any, meta: ModelMetadata) -> bytes:
    """Serialize ``(estimator, metadata)`` into one self-describing BLOB.

    Byte-for-byte compatible with ``vgi_sklearn.registry.pack_model``; used by the
    test fixtures here to stand in for a BLOB produced by vgi-sklearn.
    """
    est_bytes = _skops_dumps(estimator)
    meta_bytes = json.dumps(meta.to_dict(), default=str).encode("utf-8")
    return struct.pack(">I", len(meta_bytes)) + meta_bytes + est_bytes


def _split_blob(blob: bytes) -> tuple[bytes, bytes]:
    if not isinstance(blob, (bytes, bytearray)) or len(blob) < 4:
        raise InvalidModelBlobError("not a valid model BLOB (too short or not bytes)")
    (n,) = struct.unpack(">I", blob[:4])
    if len(blob) < 4 + n:
        raise InvalidModelBlobError("not a valid model BLOB (truncated metadata)")
    return blob[4 : 4 + n], blob[4 + n :]


def unpack_meta(blob: bytes) -> ModelMetadata:
    """Read just the metadata from a model BLOB (cheap; no estimator load)."""
    meta_bytes, _ = _split_blob(blob)
    try:
        return ModelMetadata.from_dict(json.loads(meta_bytes))
    except (json.JSONDecodeError, TypeError, UnicodeDecodeError) as exc:
        raise InvalidModelBlobError(f"not a valid model BLOB (bad metadata: {exc})") from exc


def unpack_model(blob: bytes) -> tuple[Any, ModelMetadata]:
    """Read both estimator and metadata from a model BLOB (skops safe-load)."""
    meta = unpack_meta(blob)
    _, est_bytes = _split_blob(blob)
    return _skops_loads(est_bytes), meta
