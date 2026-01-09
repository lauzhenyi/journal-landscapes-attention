#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build an edge list (core → x / x → core) and peripheral‑article metadata
from an OpenAlex snapshot.

* First pass:  scan all .gz files to collect the set of IDs **cited by** any
  core article (cached to disk so you only do it once).
* Second pass: write two Parquet files
      1. core_edge_list.parquet      –  (source_id, target_id, is_core)
      2. peripheral_metadata.parquet –  selected metadata for non‑core papers
         that either cite a core paper or are cited by a core paper.
"""

from __future__ import annotations

import gzip
import json
import os
import pickle
from multiprocessing import get_context, set_start_method, cpu_count
from pathlib import Path

import dask.dataframe as dd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
import private_info


# ──────────────────────────────────────────────────────────────
# Fields we want to keep for peripheral articles
# ──────────────────────────────────────────────────────────────
WANTED_FIELDS = [
    "id",
    "title",
    "publication_year",
    "publication_date",
    "language",
    "type",
    "is_retracted",
    "cited_by_count",
    "counts_by_year",
    "countries_distinct_count",
    "institutions_distinct_count",
    "domain",
    "primary_topic",
    "fwci",
]


# ──────────────────────────────────────────────────────────────
# Globals for the *second* pass worker (edges + peripheral meta)
# ──────────────────────────────────────────────────────────────
_shared_core_ids: set[str] = set()
_shared_cited_by_core_ids: set[str] = set()


def init_worker(core_set: set[str], cited_set: set[str]) -> None:
    """Copies the big sets into process‑wide globals (second‑pass pool)."""
    global _shared_core_ids, _shared_cited_by_core_ids
    _shared_core_ids = core_set
    _shared_cited_by_core_ids = cited_set


def process_one_file(file_path_str: str):
    """
    Worker for the second pass.

    Returns
    -------
    tuple[list[tuple[str, str, bool]], list[dict]]
        (edge_rows, metadata_rows)
    """
    if not hasattr(process_one_file, "_dbg"):
        # One‑time debug print per worker
        print(
            f"[PID {os.getpid()}] core_ids={len(_shared_core_ids):,}   "
            f"cited_by_core_ids={len(_shared_cited_by_core_ids):,}"
        )
        process_one_file._dbg = True

    path = Path(file_path_str)
    edge_rows, meta_rows = [], []

    try:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                    rec_id: str | None = rec.get("id")
                    refs: list[str] = rec.get("referenced_works", [])

                    # ----- edges (core → x  /  x → core) ----------------------
                    if rec_id in _shared_core_ids:
                        for ref in refs:
                            edge_rows.append((rec_id, ref, True))
                    else:
                        for ref in refs:
                            if ref in _shared_core_ids:
                                edge_rows.append((rec_id, ref, False))

                    # ----- peripheral metadata --------------------------------
                    cites_core = any(r in _shared_core_ids for r in refs)
                    is_cited_by_core = rec_id in _shared_cited_by_core_ids
                    is_peripheral = (
                        (rec_id not in _shared_core_ids)
                        and (cites_core or is_cited_by_core)
                    )

                    if is_peripheral:
                        entry = {
                            k: (
                                json.dumps(rec.get(k))             # lists / dicts
                                if isinstance(rec.get(k), (list, dict))
                                else str(rec.get(k))               # everything else
                            )
                            for k in WANTED_FIELDS
                        }
                        perc = rec.get("citation_normalized_percentile", {})
                        entry["is_in_top_1_percent"] = bool(
                            perc.get("is_in_top_1_percent")
                        )
                        entry["is_in_top_10_percent"] = bool(
                            perc.get("is_in_top_10_percent")
                        )
                        meta_rows.append(entry)

                except Exception:
                    # broken JSON line → skip
                    continue

    except Exception as e:
        print(f"File error in {path.name}: {e}")

    return edge_rows, meta_rows


# ──────────────────────────────────────────────────────────────
# Globals & workers for the *first* pass (collect IDs cited by core)
# ──────────────────────────────────────────────────────────────
_CORE_IDS: set[str] = set()


def _collect_init(core_ids: set[str]) -> None:
    """Initializer – copies core_ids into a global for the worker."""
    global _CORE_IDS
    _CORE_IDS = core_ids


def _collect_cited_worker(file_path_str: str) -> set[str]:
    """Return IDs referenced by *core* articles in one .gz file."""
    cited = set()
    try:
        with gzip.open(file_path_str, "rt", encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                if rec.get("id") in _CORE_IDS:
                    cited.update(rec.get("referenced_works", []))
    except Exception:
        pass
    return cited


def collect_cited_by_core_ids(gz_files: list[str], core_ids: set[str]) -> set[str]:
    """
    First pass over the dump – find every ID that *any* core paper cites.
    Results are cached to cited_by_core_ids.pkl.
    """
    cache_path = Path("cited_by_core_ids.pkl")
    if cache_path.exists():
        print("✅  Loaded cached cited_by_core_ids")
        return pickle.loads(cache_path.read_bytes())

    all_cited: set[str] = set()
    with get_context("spawn").Pool(
        processes=max(cpu_count() - 1, 1),
        initializer=_collect_init,
        initargs=(core_ids,),
    ) as pool, tqdm(total=len(gz_files)) as bar:
        for cited in pool.imap_unordered(_collect_cited_worker, gz_files):
            all_cited.update(cited)
            bar.update()

    cache_path.write_bytes(pickle.dumps(all_cited))
    print("✅  Saved cited_by_core_ids to cache")
    return all_cited


# ──────────────────────────────────────────────────────────────
# Main driver
# ──────────────────────────────────────────────────────────────
def main() -> None:
    print("🔎  Loading core articles …")
    # Read only the 'id' column; convert to str for cheaper comparisons
    ddf = dd.read_parquet("checkpoint_datacleaning2.parquet", columns=["id"])
    core_ids: set[str] = {str(x) for x in ddf["id"].compute()}
    print(f"   core_ids            = {len(core_ids):,}")

    src_dir = Path(private_info.open_alex_data_dump_dir)
    gz_files = sorted(src_dir.rglob("*.gz"))
    print(f"   .gz files available = {len(gz_files):,}")

    # ---------- first pass (may load from cache) ------------------------------
    cited_by_core_ids = collect_cited_by_core_ids(gz_files, core_ids)
    print(f"   cited_by_core_ids    = {len(cited_by_core_ids):,}")

    # ---------- prepare Parquet writers ---------------------------------------
    edge_schema = pa.schema(
        [
            ("source_id", pa.string()),
            ("target_id", pa.string()),
            ("is_core", pa.bool_()),
        ]
    )
    meta_schema = pa.schema(
        [(k, pa.string()) for k in WANTED_FIELDS]
        + [
            ("is_in_top_1_percent", pa.bool_()),
            ("is_in_top_10_percent", pa.bool_()),
        ]
    )
    edge_writer = meta_writer = None

    # ---------- second pass (build edges + metadata) --------------------------
    print("🔁  Building edge list + peripheral metadata …")
    with get_context("spawn").Pool(
        processes=max(cpu_count() - 1, 1),
        initializer=init_worker,
        initargs=(core_ids, cited_by_core_ids),
    ) as pool, tqdm(total=len(gz_files)) as bar:

        for edge_rows, meta_rows in pool.imap_unordered(process_one_file, gz_files):
            bar.update()

            if edge_rows:
                if edge_writer is None:
                    edge_writer = pq.ParquetWriter(
                        "core_edge_list.parquet", edge_schema, compression="snappy"
                    )
                edge_writer.write_table(
                    pa.Table.from_pylist(
                        [
                            {
                                "source_id": s,
                                "target_id": t,
                                "is_core": c,
                            }
                            for s, t, c in edge_rows
                        ],
                        schema=edge_schema,
                    )
                )

            if meta_rows:
                if meta_writer is None:
                    meta_writer = pq.ParquetWriter(
                        "peripheral_metadata.parquet",
                        meta_schema,
                        compression="snappy",
                    )
                meta_writer.write_table(
                    pa.Table.from_pylist(meta_rows, schema=meta_schema)
                )

    # ---------- tidy up -------------------------------------------------------
    if edge_writer:
        edge_writer.close()
    if meta_writer:
        meta_writer.close()
    print("✅  Finished – Parquet files written.")


if __name__ == "__main__":
    # On macOS the default is already 'spawn', but be explicit & safe.
    set_start_method("spawn", force=True)
    main()
