"""SHAP model explanations as DuckDB/SQL functions, for models fitted by sibling
VGI workers (vgi-sklearn, vgi-xgboost).

vgi-explain does not train models; it *interprets* the self-contained model BLOB
those workers produce. The implementation is split so each module stays focused:

- ``registry``  -- unpack the model BLOB (skops safe-load; trusted namespaces)
- ``explain``   -- pure SHAP logic (explainer selection, additivity, importance)
- ``features``  -- align an input relation to a model's fitted columns by name
- ``functions`` -- the SQL function surface (shap_values / shap_base_value /
  feature_importance)
- ``worker``    -- assembles the ``explain`` catalog and the process entry points

``explain_worker.py`` at the repo root assembles these into the ``explain``
catalog and runs the worker.
"""

from __future__ import annotations

__version__ = "0.1.0"
