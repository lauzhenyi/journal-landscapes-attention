#!/usr/bin/env python3
"""
build_parquet_from_dump.py
--------------------------

Scan a local OpenAlex works dump (*.gz) and build papers.parquet, applying the
same filters and extra columns you used with the API version.

Changes relative to v1
----------------------
* `ParquetSink` now fixes the schema to **all-string** once and for all, then
  casts every incoming batch (`pa.Table.cast`) to that schema.  This guarantees
  column-type consistency, even when a later batch has only nulls.
* Everything else (filters, one-time debug print per worker, streaming write)
  behaves exactly as before.

Run
---
python build_parquet_from_dump.py
"""

import gzip
import json
import sys
from multiprocessing import Pool, cpu_count, set_start_method
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

import private_info  

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Journal metadata → lookup
# ─────────────────────────────────────────────────────────────────────────────
df_journals = pd.read_csv("progress.csv")

df_journals["canonical_issn"] = (
    df_journals["EISSN"]
    .fillna(df_journals["ISSN"])
    .astype(str)
    .str.lower()
    .str.strip()
)
df_journals.dropna(subset=["canonical_issn"], inplace=True)

ISSN_SET  = set(df_journals["canonical_issn"])
TITLE_MAP = dict(zip(df_journals["canonical_issn"], df_journals["title"]))
OA_MAP    = dict(zip(df_journals["canonical_issn"],
                     df_journals.get("is_oa", pd.Series(False, index=df_journals.index))))

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Parquet writer with fixed string schema
# ─────────────────────────────────────────────────────────────────────────────
class ParquetSink:
    """Incrementally appends batches to `path`, coercing every column to pa.string()"""

    def __init__(self, path: str):
        self.path = path
        self.writer = None       # pq.ParquetWriter
        self.schema = None       # pa.schema
        self.master_cols = None  # list[str]

    def _init_writer(self, df_first: pd.DataFrame):
        """Create writer & fixed all-string schema based on first batch columns."""
        self.master_cols = sorted(df_first.columns.tolist())        # stable order
        self.schema = pa.schema([(c, pa.string()) for c in self.master_cols])

        table = pa.Table.from_pandas(
            df_first[self.master_cols].astype(str),   # enforce pandas string
            preserve_index=False
        ).cast(self.schema)

        self.writer = pq.ParquetWriter(self.path, self.schema)
        self.writer.write_table(table)

    def write(self, df_batch: pd.DataFrame):
        if df_batch.empty:
            return

        if self.writer is None:
            self._init_writer(df_batch)
            return

        # add any missing cols (unlikely) and ensure full set / order
        for col in self.master_cols:
            if col not in df_batch.columns:
                df_batch[col] = ""
        df_batch = df_batch[self.master_cols]

        # pyarrow table: cast → guaranteed schema match
        table = pa.Table.from_pandas(
            df_batch.astype(str),
            preserve_index=False
        ).cast(self.schema)

        self.writer.write_table(table)

    def close(self):
        if self.writer is not None:
            self.writer.close()

# ─────────────────────────────────────────────────────────────────────────────
# 3.  Filter helper
# ─────────────────────────────────────────────────────────────────────────────
def record_matches(work_json: dict) -> tuple[bool, str]:
    """Return (True, issn) if work passes filters; else (False, "")."""
    try:
        if work_json.get("publication_date", "") < "2015-01-01":
            return False, ""
        issn = work_json["primary_location"]["source"]["issn_l"].lower().strip()
        return (issn in ISSN_SET), issn
    except Exception:
        return False, ""

# ─────────────────────────────────────────────────────────────────────────────
# 4.  Worker function
# ─────────────────────────────────────────────────────────────────────────────
def process_gz_file(gz_path: str) -> list[dict]:
    """
    Read one .gz, return list of matching works.
    Prints a debug line once per process when the first match is found.
    """
    _debug_printed = False
    matches = []

    with gzip.open(gz_path, "rt", encoding="utf-8") as fh:
        for line in fh:
            try:
                work = json.loads(line)
            except json.JSONDecodeError:
                continue

            keep, issn = record_matches(work)
            if not keep:
                continue

            if not _debug_printed:
                _debug_printed = True
                print(f"[DEBUG {Path(gz_path).name}] first match → id={work.get('id')}",
                      file=sys.stderr, flush=True)

            work["journal_issn"]  = issn
            work["journal_title"] = TITLE_MAP.get(issn, "")
            work["is_oa_flag"]    = OA_MAP.get(issn, False)
            matches.append(work)

    return matches

# ─────────────────────────────────────────────────────────────────────────────
# 5.  Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    dump_root = Path(private_info.open_alex_data_dump_dir)
    gz_files  = sorted(dump_root.rglob("*.gz"))
    if not gz_files:
        sys.exit(f"No .gz files found under {dump_root}")

    sink = ParquetSink("papers.parquet")
    n_workers = max(cpu_count() - 1, 1)

    print(f"🔍 Scanning {len(gz_files):,} files with {n_workers} worker(s)…")
    with Pool(processes=n_workers) as pool:
        for batch in tqdm(pool.imap_unordered(process_gz_file, map(str, gz_files)),
                          total=len(gz_files), desc="Files", unit="file"):
            if batch:
                sink.write(pd.DataFrame(batch))

    sink.close()
    print("✅  Finished!  papers.parquet is ready.")

# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    set_start_method("spawn", force=True)   # portable across OSes
    main()
