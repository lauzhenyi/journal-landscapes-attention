#!/usr/bin/env python3
"""
build_parquet_from_dump_topics.py
---------------------------------

Scan a local OpenAlex works dump (*.gz) and build robustness_check_papers.parquet.

Filtering:
- language == "en"
- type == "article"
- topics[*].field.display_name in allowed list (case-insensitive)

Keep only these top-level keys:
- title
- topics
- referenced_works_count
- primary_location
- is_retracted

Plus derived columns:
- field  : unique topics[*].field.display_name joined by "; "
- domain : unique topics[*].domain.display_name joined by "; "

Run:
python build_parquet_from_dump_topics.py
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

import private_info  # must define open_alex_data_dump_dir


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Allowed fields (case-insensitive)
# ─────────────────────────────────────────────────────────────────────────────
ALLOWED_FIELDS = {
    "chemistry",
    "computer science",
    "mathematics",
    "economics, econometrics and finance",
    "psychology",
}


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Parquet writer with fixed string schema
# ─────────────────────────────────────────────────────────────────────────────
class ParquetSink:
    """Incrementally appends batches to `path`, coercing every column to pa.string()."""

    def __init__(self, path: str):
        self.path = path
        self.writer = None
        self.schema = None
        self.master_cols = None

    def _init_writer(self, df_first: pd.DataFrame):
        self.master_cols = sorted(df_first.columns.tolist())
        self.schema = pa.schema([(c, pa.string()) for c in self.master_cols])

        table = pa.Table.from_pandas(
            df_first[self.master_cols].astype(str),
            preserve_index=False,
        ).cast(self.schema)

        self.writer = pq.ParquetWriter(self.path, self.schema)
        self.writer.write_table(table)

    def write(self, df_batch: pd.DataFrame):
        if df_batch.empty:
            return

        if self.writer is None:
            self._init_writer(df_batch)
            return

        for col in self.master_cols:
            if col not in df_batch.columns:
                df_batch[col] = ""
        df_batch = df_batch[self.master_cols]

        table = pa.Table.from_pandas(
            df_batch.astype(str),
            preserve_index=False,
        ).cast(self.schema)

        self.writer.write_table(table)

    def close(self):
        if self.writer is not None:
            self.writer.close()


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Topic helpers + filter
# ─────────────────────────────────────────────────────────────────────────────
def extract_field_and_domain(topics: list[dict]) -> tuple[list[str], list[str]]:
    fields = []
    domains = []
    for t in topics or []:
        f = (t.get("field") or {}).get("display_name")
        d = (t.get("domain") or {}).get("display_name")
        if f:
            fields.append(str(f).strip())
        if d:
            domains.append(str(d).strip())
    return fields, domains


def record_matches(work_json: dict) -> tuple[bool, str, str]:
    """
    Return (keep, field_str, domain_str).
    keep iff:
      - language == "en"
      - type == "article"
      - any topics[*].field.display_name is in ALLOWED_FIELDS (case-insensitive)
    """
    if work_json.get("language") != "en":
        return False, "", ""
    if work_json.get("type") != "article":
        return False, "", ""

    topics = work_json.get("topics") or []
    fields, domains = extract_field_and_domain(topics)

    keep = any((f.lower() in ALLOWED_FIELDS) for f in fields)
    if not keep:
        return False, "", ""

    field_str = "; ".join(sorted(set(fields)))
    domain_str = "; ".join(sorted(set(domains)))
    return True, field_str, domain_str


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Worker
# ─────────────────────────────────────────────────────────────────────────────
def process_gz_file(gz_path: str) -> list[dict]:
    """
    Read one .gz, return list of matching works with reduced columns.
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

            keep, field_str, domain_str = record_matches(work)
            if not keep:
                continue

            if not _debug_printed:
                _debug_printed = True
                print(
                    f"[DEBUG {Path(gz_path).name}] first match → id={work.get('id')}",
                    file=sys.stderr,
                    flush=True,
                )

            out = {
                "title": work.get("title", ""),
                "topics": json.dumps(work.get("topics") or [], ensure_ascii=False),
                "referenced_works_count": work.get("referenced_works_count", ""),
                "primary_location": json.dumps(work.get("primary_location") or {}, ensure_ascii=False),
                "is_retracted": work.get("is_retracted", ""),
                "field": field_str,
                "domain": domain_str,
            }
            matches.append(out)

    return matches


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    dump_root = Path(private_info.open_alex_data_dump_dir)
    gz_files = sorted(dump_root.rglob("*.gz"))
    if not gz_files:
        sys.exit(f"No .gz files found under {dump_root}")

    out_path = "robustness_check_papers.parquet"
    sink = ParquetSink(out_path)
    n_workers = max(cpu_count() - 1, 1)

    print(f"🔍 Scanning {len(gz_files):,} files with {n_workers} worker(s)…")
    with Pool(processes=n_workers) as pool:
        for batch in tqdm(
            pool.imap_unordered(process_gz_file, map(str, gz_files)),
            total=len(gz_files),
            desc="Files",
            unit="file",
        ):
            if batch:
                sink.write(pd.DataFrame(batch))

    sink.close()
    print(f"✅ Finished! {out_path} is ready.")


if __name__ == "__main__":
    set_start_method("spawn", force=True)
    main()
