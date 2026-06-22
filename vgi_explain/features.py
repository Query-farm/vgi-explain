"""Feature-matrix assembly: align an input relation to a model's fitted columns.

Same convention as vgi-sklearn ``predict``: features are selected *by name* (so
input column order does not matter and shuffled columns still work), an optional
``id`` column is excluded from features and carried through, and a clear error is
raised for a missing or non-numeric feature column rather than an opaque
pyarrow / numpy failure.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa


def matrix(table: pa.Table, feature_names: list[str], *, what: str = "feature") -> np.ndarray:
    """Assemble the named columns (in the given order) into a 2D float64 array.

    Selects ``feature_names`` by name, so input column order does not matter and
    extra columns are ignored. Raises a clear error when a column is missing or
    not numeric.
    """
    present = set(table.schema.names)
    missing = [n for n in feature_names if n not in present]
    if missing:
        raise ValueError(
            f"missing required {what} column(s): {', '.join(missing)}; "
            f"input has columns: {', '.join(table.schema.names)}"
        )
    non_numeric = [
        n
        for n in feature_names
        if not pa.types.is_floating(table.schema.field(n).type)
        and not pa.types.is_integer(table.schema.field(n).type)
        and not pa.types.is_boolean(table.schema.field(n).type)
    ]
    if non_numeric:
        raise ValueError(
            f"{what} column(s) must be numeric, but these are not: "
            + ", ".join(f"{n} ({table.schema.field(n).type})" for n in non_numeric)
            + ". Select only numeric columns, or encode/scale them first."
        )
    cols = [np.asarray(table.column(name).to_numpy(zero_copy_only=False), dtype=float) for name in feature_names]
    if not cols:
        return np.empty((table.num_rows, 0), dtype=float)
    return np.column_stack(cols)
