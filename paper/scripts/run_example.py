#!/usr/bin/env python3
"""
Headless reproducible example script for DocuScope CA.

Purpose
-------
Provide a minimal, deterministic, non-Streamlit workflow that:
1. Loads the sample corpus in `paper/data/test_corpus/`
2. Parses it with the DocuScope-enhanced spaCy model (local or installed)
3. Generates core analytical artifacts (tokens + frequency & tag tables)
4. Writes outputs to `paper/data/example_output/`
5. Records a manifest (JSON) capturing environment + reproducibility metadata

Intended Use
------------
Run inside the project root (or within the Docker container) with:
    python paper/scripts/run_example.py

The produced artifacts can be referenced in the JOSS paper and regenerated
by reviewers to validate functionality without using the GUI.

Notes
-----
- No network calls are made intentionally; AI/LLM features are not invoked.
- Script avoids Streamlit session/state; uses direct library calls only.
- If the spaCy model isn't installed, it attempts to load from `webapp/_models/`.
- If parsing fails due to unexpected DataFrame schema, adjust the schema section.

Outputs
-------
`paper/data/example_output/`
    tokens.parquet            -> Parsed token-level annotations
    frequency_pos.parquet     -> POS frequency table
    frequency_ds.parquet      -> DocuScope tag frequency table
    tags_pos.parquet          -> POS tags table
    tags_ds.parquet           -> DocuScope tags table
    manifest.json             -> Reproducibility metadata

Exit Codes
----------
0 success
1 missing corpus directory
2 no text files found
3 model load failure
4 parse failure
5 artifact write failure

"""
from __future__ import annotations

import json
import sys
import time
import hashlib
import subprocess
from pathlib import Path
from typing import List, Dict, Any

# Deterministic seeds
import random
random.seed(42)
try:
    import numpy as np
    np.random.seed(42)  # type: ignore
except Exception:  # pragma: no cover
    np = None  # noqa: N816

try:
    import polars as pl
except ImportError as e:
    print(
        "[ERROR] polars is required. Install project dependencies first.",
        file=sys.stderr,
    )
    raise SystemExit(1) from e

try:
    import spacy
except ImportError as e:
    print("[ERROR] spaCy is required. Install project dependencies first.", file=sys.stderr)
    raise SystemExit(1) from e

try:
    import docuscospacy as ds
except ImportError as e:
    print(
        "[ERROR] docuscospacy is required. Install project dependencies first.",
        file=sys.stderr,
    )
    raise SystemExit(1) from e

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CORPUS_DIR = PROJECT_ROOT / "paper" / "data" / "test_corpus"
OUTPUT_DIR = PROJECT_ROOT / "paper" / "data" / "example_output"
PYPROJECT_TOML = PROJECT_ROOT / "pyproject.toml"
MODEL_CANDIDATES: List[str] = [
    "en_docusco_spacy",  # installed package path
]
LOCAL_MODEL_DIRS: List[Path] = [
    PROJECT_ROOT / "webapp" / "_models" / "en_docusco_spacy",
    PROJECT_ROOT / "webapp" / "_models" / "en_docusco_spacy_cd",
]


def read_project_version() -> str:
    try:
        import tomllib  # Python 3.11+
        with PYPROJECT_TOML.open("rb") as f:
            data = tomllib.load(f)
        return data.get("project", {}).get("version", "unknown")
    except Exception:
        return "unknown"


def current_git_commit() -> str | None:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT)
            .decode()
            .strip()
        )
    except Exception:
        return None


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def load_sample_corpus(corpus_dir: Path) -> pl.DataFrame:
    if not corpus_dir.exists():
        print(f"[ERROR] Corpus directory not found: {corpus_dir}", file=sys.stderr)
        raise SystemExit(1)
    txt_files = sorted(p for p in corpus_dir.glob("*.txt"))
    if not txt_files:
        print(f"[ERROR] No .txt files found in {corpus_dir}", file=sys.stderr)
        raise SystemExit(2)

    records: List[Dict[str, Any]] = []
    for idx, path in enumerate(txt_files, start=1):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(encoding="latin-1", errors="ignore")
        records.append({
            "doc_id": idx,
            "doc_name": path.stem,
            "text": text,
            "n_chars": len(text),
            "sha256": sha256_of_file(path),
        })
    return pl.DataFrame(records)


def _prepare_for_parsing(df: pl.DataFrame) -> pl.DataFrame:
    """Return a DataFrame with only required columns (doc_id, text) and
    enforce the expected types for docuscospacy.docuscope_parse.

    docuscospacy expects:
        - doc_id: UTF8 / string
        - text:   UTF8 / string
    Any other columns cause a schema validation error.
    """
    base = df.select([pl.col("doc_id").cast(pl.Utf8), pl.col("text")])
    return base


def load_model() -> Any:
    # Try named installed models first
    for name in MODEL_CANDIDATES:
        try:
            return spacy.load(name)
        except Exception:
            continue
    # Try local directories
    for d in LOCAL_MODEL_DIRS:
        if d.exists():
            try:
                return spacy.load(str(d))
            except Exception:
                continue
    print(
        "[ERROR] Unable to load a DocuScope spaCy model from candidates.",
        file=sys.stderr,
    )
    raise SystemExit(3)


def parse_corpus(df: pl.DataFrame, nlp) -> pl.DataFrame:
    """Invoke docuscospacy pipeline.

    Expected input schema is flexible; we provide both doc_name and text.
    """
    # Strip to required schema
    minimal_df = _prepare_for_parsing(df)
    try:
        parsed = ds.docuscope_parse(corp=minimal_df, nlp_model=nlp)
    except Exception as e:  # pragma: no cover
        print(f"[ERROR] Parsing failed: {e}", file=sys.stderr)
        raise SystemExit(4) from e
    return parsed


def compute_artifacts(tokens: pl.DataFrame) -> Dict[str, pl.DataFrame]:
    try:
        ft_pos, ft_ds = ds.frequency_table(tokens, count_by="both")
        tt_pos, tt_ds = ds.tags_table(tokens, count_by="both")
    except Exception as e:
        print(f"[ERROR] Failed to compute frequency/tag tables: {e}", file=sys.stderr)
        raise SystemExit(4) from e
    return {
        "frequency_pos": ft_pos,
        "frequency_ds": ft_ds,
        "tags_pos": tt_pos,
        "tags_ds": tt_ds,
    }


def write_parquet(df: pl.DataFrame, path: Path) -> None:
    try:
        df.write_parquet(path)
    except Exception as e:
        print(f"[ERROR] Failed to write {path.name}: {e}", file=sys.stderr)
        raise SystemExit(5) from e


def build_manifest(
    tokens: pl.DataFrame,
    artifacts: Dict[str, pl.DataFrame],
    model_name: str,
    original_docs: pl.DataFrame,
) -> Dict[str, Any]:
    total_tokens = int(tokens.height)
    n_docs = (
        int(tokens.get_column("doc_id").unique().len())
        if "doc_id" in tokens.columns
        else None
    )
    version = read_project_version()
    commit = current_git_commit()
    return {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "project_version": version,
        "git_commit": commit,
        "model_name": model_name,
        "artifacts": {
            k: {"rows": int(v.height), "columns": v.columns}
            for k, v in artifacts.items()
        },
        "corpus": {
            "n_documents": n_docs,
            "total_tokens": total_tokens,
            "documents": [
                {
                    "doc_id": str(row[0]),
                    "doc_name": row[1],
                    "n_chars": int(row[2]),
                    "sha256": row[3],
                }
                for row in original_docs.select(
                    ["doc_id", "doc_name", "n_chars", "sha256"]
                ).to_numpy()
            ],
        },
        "script": "run_example.py",
        "repro_instructions": "python paper/scripts/run_example.py",
    }


def main() -> int:
    print("[INFO] Loading sample corpus ...")
    raw_df = load_sample_corpus(CORPUS_DIR)
    print(f"[INFO] Loaded {raw_df.height} documents (rows) from test corpus.")

    print("[INFO] Loading spaCy DocuScope model ...")
    nlp = load_model()
    model_name = getattr(nlp, "meta", {}).get("name", "unknown_model")
    print(f"[INFO] Model loaded: {model_name}")

    print("[INFO] Parsing corpus (DocuScope + POS tagging) ...")
    tokens = parse_corpus(raw_df, nlp)
    print(
        f"[INFO] Parsed tokens DataFrame shape: {tokens.height} rows x "
        f"{len(tokens.columns)} cols"
    )

    print("[INFO] Computing frequency & tag tables ...")
    artifacts = compute_artifacts(tokens)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Writing artifacts to {OUTPUT_DIR.relative_to(PROJECT_ROOT)}")

    write_parquet(tokens, OUTPUT_DIR / "tokens.parquet")
    for name, df in artifacts.items():
        write_parquet(df, OUTPUT_DIR / f"{name}.parquet")

    manifest = build_manifest(tokens, artifacts, model_name, raw_df)
    with (OUTPUT_DIR / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("[INFO] Done. Generated files:")
    for p in sorted(OUTPUT_DIR.glob("*")):
        print(f"  - {p.name}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
