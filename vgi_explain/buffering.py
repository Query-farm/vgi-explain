"""Shared plumbing for the buffering table function (feature_importance).

``feature_importance`` computes a global statistic -- mean(|shap|) over every
input row -- so it must see the whole relation before emitting. The sink phase
serializes each input batch to execution-scoped storage; finalize reassembles
the full table. Lifted from vgi-sklearn's buffering helper.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pyarrow as pa
from vgi.table_buffering_function import TableBufferingFunction, TableBufferingParams
from vgi_rpc import ArrowSerializableDataclass

_DATA_KEY = b"input_batches"


@dataclass(kw_only=True)
class DrainState(ArrowSerializableDataclass):
    """Per-finalize-stream cursor: emit the result once, then finish."""

    done: bool = False


def serialize_batch(batch: pa.RecordBatch) -> bytes:
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, batch.schema) as writer:
        writer.write_batch(batch)
    return sink.getvalue().to_pybytes()


def deserialize_batches(value: bytes) -> list[pa.RecordBatch]:
    reader = pa.ipc.open_stream(pa.BufferReader(value))
    return reader.read_all().to_batches()


class SinkBuffer[TArgs, TState](TableBufferingFunction[TArgs, TState]):
    """Single-bucket sink/combine: buffer every input batch under one key."""

    @classmethod
    def process(cls, batch: pa.RecordBatch, params: TableBufferingParams[TArgs]) -> bytes:
        if batch.num_rows:
            params.storage.state_append(_DATA_KEY, b"", serialize_batch(batch))
        return params.execution_id

    @classmethod
    def combine(cls, state_ids: list[bytes], params: TableBufferingParams[TArgs]) -> list[bytes]:
        return [params.execution_id]

    @classmethod
    def buffered_table(cls, params: TableBufferingParams[TArgs], input_schema: pa.Schema) -> pa.Table | None:
        batches: list[pa.RecordBatch] = []
        for _sid, value in params.storage.state_log_scan(_DATA_KEY, b""):
            batches.extend(deserialize_batches(value))
        if not batches:
            return None
        return pa.Table.from_batches(batches, schema=input_schema)


def input_schema_of(params: Any) -> pa.Schema:
    """Input schema from a process/finalize params object."""
    schema = params.init_call.bind_call.input_schema
    assert schema is not None
    return schema
