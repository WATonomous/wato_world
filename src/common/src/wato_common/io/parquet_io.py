"""Thin pyarrow wrappers used by every component. All paths are URIs."""

from __future__ import annotations

import os
from typing import Iterable

import pyarrow as pa
import pyarrow.parquet as pq

from wato_common.artifact_store import ensure_local_dir, local_path


def write_table(rows: Iterable[dict], schema: pa.Schema, uri: str) -> None:
    """Write a list of dict rows to a Parquet file at `uri`.

    The schema is required so empty inputs still produce a valid file (with
    columns).  Bool/string columns missing from a row are filled with None.
    """
    rows = list(rows)
    if rows:
        # Normalize: every row must have every schema column.
        cols = {f.name: [] for f in schema}
        for r in rows:
            for f in schema:
                cols[f.name].append(r.get(f.name))
        table = pa.Table.from_pydict(cols, schema=schema)
    else:
        table = schema.empty_table()

    parent = os.path.dirname(uri)
    if parent:
        ensure_local_dir(parent)
    pq.write_table(table, local_path(uri))


def read_table(uri: str) -> pa.Table:
    return pq.read_table(local_path(uri))


def read_rows(uri: str) -> list[dict]:
    return read_table(uri).to_pylist()
