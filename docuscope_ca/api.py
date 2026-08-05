"""Headless processing API for DocuScope CA.

Design goals:
* Zero dependency on Streamlit session/state.
* Deterministic, reproducible outputs (seed optional; no randomness expected here).
* Clear error taxonomy for caller handling.
* Minimal, well-typed surface returning rich results instead of forcing file export.

This module intentionally duplicates a *small* subset of logic from the example
script to avoid importing UI layers. Refactors in internal processing code should
avoid breaking this contract; extend with **new** optional parameters instead of
changing or removing existing ones.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Dict, Any, Optional, Literal
import hashlib
import time

import polars as pl

try:  # soft import; raise custom error later if missing
    import spacy  # type: ignore
except Exception:  # pragma: no cover
    spacy = None  # type: ignore

try:
    import docuscospacy as ds  # type: ignore
except Exception:  # pragma: no cover
    ds = None  # type: ignore

Metric = Literal["freq", "tags", "dtm"]
SUPPORTED_METRICS: tuple[Metric, ...] = ("freq", "tags", "dtm")


# ------------------------- Error classes ---------------------------------- #

class CorpusProcessingError(RuntimeError):
    """Base class for processing layer errors."""


class CorpusLoadError(CorpusProcessingError):
    """Raised when input corpus cannot be read / is empty."""


class ModelLoadError(CorpusProcessingError):
    """Raised when the spaCy DocuScope model cannot be loaded."""


class ArtifactWriteError(CorpusProcessingError):
    """Raised when exporting artifacts fails."""


# ------------------------- Data structures -------------------------------- #

@dataclass(slots=True)
class CorpusResult:
    """Container for processed corpus artifacts.

    Attributes
    ----------
    tokens: pl.DataFrame
        Token-level annotations (DocuScope + POS + structural fields).
    freq_pos / freq_ds: Optional frequency tables (word-level & rhetorical).
    tags_pos / tags_ds: Optional tag tables (per-document tag counts).
    dtm_pos / dtm_ds: Optional document-term matrices.
    manifest: Dict[str, Any]
        Provenance + reproducibility metadata.
    timings: Dict[str, float]
        Simple timing breakdown (seconds) for major phases.
    """

    tokens: pl.DataFrame
    freq_pos: Optional[pl.DataFrame] = None
    freq_ds: Optional[pl.DataFrame] = None
    tags_pos: Optional[pl.DataFrame] = None
    tags_ds: Optional[pl.DataFrame] = None
    dtm_pos: Optional[pl.DataFrame] = None
    dtm_ds: Optional[pl.DataFrame] = None
    manifest: Dict[str, Any] = None  # type: ignore[assignment]
    timings: Dict[str, float] = None  # type: ignore[assignment]


# ------------------------- Helpers ---------------------------------------- #

def _sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def _ingest_sources(sources: Sequence[str] | str) -> pl.DataFrame:
    """Build initial corpus DataFrame.

    Accepts:
    * Directory path (string) -> scans for *.txt
    * Single file path or list of file paths (.txt)
        * List of raw text strings (if not all elements resolve to files).
            If *all* supplied entries are existing file paths, they are treated as files.
    """
    paths: List[Path] = []
    records: List[Dict[str, Any]] = []

    if isinstance(sources, str):
        p = Path(sources)
        if p.is_dir():
            paths = sorted(p.glob("*.txt"))
        elif p.is_file():
            paths = [p]
        else:
            # treat as single raw text
            text = sources
            records.append({
                "doc_id": "1",
                "doc_name": "doc_1",
                "text": text,
                "n_chars": len(text),
                "sha256": _sha256_bytes(text.encode("utf-8")),
            })
    else:
        # list / sequence
        candidate_paths = [Path(s) for s in sources]
        if all(p.exists() and p.is_file() for p in candidate_paths):
            paths = candidate_paths  # all real files
        else:
            # treat everything as raw text sequence
            for i, text in enumerate(sources, start=1):
                records.append({
                    "doc_id": str(i),
                    "doc_name": f"doc_{i}",
                    "text": text,
                    "n_chars": len(text),
                    "sha256": _sha256_bytes(text.encode("utf-8")),
                })

    for idx, path in enumerate(paths, start=1):
        try:
            b = path.read_bytes()
            text = b.decode("utf-8")
        except UnicodeDecodeError:
            text = b.decode("latin-1", errors="ignore")  # type: ignore[name-defined]
        records.append({
            "doc_id": str(idx),
            "doc_name": path.stem,
            "text": text,
            "n_chars": len(text),
            "sha256": _sha256_bytes(b),  # type: ignore[name-defined]
        })

    if not records:
        raise CorpusLoadError("No documents found in provided sources")

    return pl.DataFrame(records)


def _load_model(model: str) -> Any:
    if spacy is None:  # pragma: no cover
        raise ModelLoadError("spaCy not available; install project dependencies")
    try:
        return spacy.load(model)
    except Exception as e:
        # attempt local model fallback (common repository path)
        local_dir = Path(__file__).resolve().parents[1] / "webapp" / "_models" / model
        if local_dir.exists():  # pragma: no branch
            try:
                return spacy.load(str(local_dir))
            except Exception:
                pass
        raise ModelLoadError(f"Unable to load model '{model}': {e}") from e


def _parse(tokens_df: pl.DataFrame, nlp) -> pl.DataFrame:
    if ds is None:  # pragma: no cover
        raise CorpusProcessingError("docuscospacy not available; install dependencies")
    base = tokens_df.select([
        pl.col("doc_id").cast(pl.Utf8),
        pl.col("text")
    ])
    try:
        parsed = ds.docuscope_parse(corp=base, nlp_model=nlp)
    except Exception as e:
        raise CorpusProcessingError(f"Parsing failed: {e}") from e
    return parsed


def _normalize_metrics(metrics: Iterable[Metric]) -> tuple[Metric, ...]:
    """Materialize, validate, and deduplicate requested metrics in order."""

    requested = tuple(metrics)
    if not requested:
        raise CorpusProcessingError(
            "At least one metric is required; choose from freq, tags, and dtm."
        )

    unsupported = sorted(
        {str(metric) for metric in requested if metric not in SUPPORTED_METRICS}
    )
    if unsupported:
        raise CorpusProcessingError(
            "Unsupported metric selection: "
            f"{', '.join(unsupported)}. Choose from freq, tags, and dtm."
        )

    return tuple(dict.fromkeys(requested))


def _compute_metrics(tokens: pl.DataFrame, metrics: Sequence[Metric]):
    freq_pos = freq_ds = tags_pos = tags_ds = dtm_pos = dtm_ds = None
    if ds is None:  # pragma: no cover
        raise CorpusProcessingError("docuscospacy not available; install dependencies")
    metric_set = set(metrics)
    if metric_set & {"freq", "tags"}:
        # frequency_table(count_by="both") returns (pos, ds)
        if "freq" in metric_set:
            freq_pos, freq_ds = ds.frequency_table(tokens, count_by="both")
        if "tags" in metric_set:
            tags_pos, tags_ds = ds.tags_table(tokens, count_by="both")
    if "dtm" in metric_set:
        dtm_pos, dtm_ds = ds.tags_dtm(tokens, count_by="both")
    return freq_pos, freq_ds, tags_pos, tags_ds, dtm_pos, dtm_ds


def _build_manifest(
    tokens: pl.DataFrame,
    docs_df: pl.DataFrame,
    metrics: Sequence[Metric],
    model_name: str,
) -> Dict[str, Any]:
    total_tokens = int(tokens.height)
    n_docs = (
        int(tokens.get_column("doc_id").unique().len())
        if "doc_id" in tokens.columns
        else None
    )
    try:
        from importlib.metadata import version
        pkg_version = version("docuscope-ca-online")
    except Exception:  # pragma: no cover
        pkg_version = "0.0.0+local"
    return {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "package_version": pkg_version,
        "model_name": model_name,
        "metrics": list(metrics),
        "corpus": {
            "n_documents": n_docs,
            "total_tokens": total_tokens,
            "documents": [
                {
                    "doc_id": row[0],
                    "doc_name": row[1],
                    "n_chars": int(row[2]),
                    "sha256": row[3],
                }
                for row in docs_df.select(
                    ["doc_id", "doc_name", "n_chars", "sha256"]
                ).to_numpy()
            ],
        },
    }


# ------------------------- Public function -------------------------------- #

def process_corpus(
    sources: Sequence[str] | str,
    model: str = "en_docusco_spacy",
    metrics: Iterable[Metric] = ("freq", "tags"),
    export_dir: Optional[str | Path] = None,
) -> CorpusResult:
    """Process a corpus headlessly and (optionally) export artifacts.

    Parameters
    ----------
    sources:
        Directory path, file path, list of file paths, or list of raw string documents.
    model:
        spaCy model name or path (fallback to local ``webapp/_models/<model>``).
    metrics:
        Non-empty iterable of metric identifiers among {"freq", "tags", "dtm"}.
        Duplicate identifiers are removed while preserving their first-seen order.
    export_dir:
        Optional directory to write artifacts (Parquet + manifest JSON). Created if missing.

    Returns
    -------
    CorpusResult
        Container with tokens and any requested metric tables.
    """
    normalized_metrics = _normalize_metrics(metrics)
    t0 = time.perf_counter()
    docs_df = _ingest_sources(sources)
    t_ingest = time.perf_counter()
    nlp = _load_model(model)
    t_model = time.perf_counter()
    tokens = _parse(docs_df, nlp)
    t_parse = time.perf_counter()
    freq_pos, freq_ds, tags_pos, tags_ds, dtm_pos, dtm_ds = _compute_metrics(
        tokens, normalized_metrics
    )
    t_metrics = time.perf_counter()

    # Resolve model name robustly (meta may differ if loaded from path)
    meta = getattr(nlp, "meta", {})
    model_name = (
        meta.get("name")
        or getattr(meta, "get", lambda *_: None)("name")  # type: ignore[attr-defined]
        or model
    )
    manifest = _build_manifest(tokens, docs_df, normalized_metrics, model_name)

    result = CorpusResult(
        tokens=tokens,
        freq_pos=freq_pos,
        freq_ds=freq_ds,
        tags_pos=tags_pos,
        tags_ds=tags_ds,
        dtm_pos=dtm_pos,
        dtm_ds=dtm_ds,
        manifest=manifest,
        timings={
            "ingest": t_ingest - t0,
            "model_load": t_model - t_ingest,
            "parse": t_parse - t_model,
            "metrics": t_metrics - t_parse,
            "total": t_metrics - t0,
        },
    )

    if export_dir is not None:
        _export_results(result, export_dir)

    return result


def _export_results(result: CorpusResult, export_dir: str | Path) -> None:
    out = Path(export_dir)
    out.mkdir(parents=True, exist_ok=True)
    try:
        result.tokens.write_parquet(out / "tokens.parquet")
        if result.freq_pos is not None:
            result.freq_pos.write_parquet(out / "frequency_pos.parquet")
        if result.freq_ds is not None:
            result.freq_ds.write_parquet(out / "frequency_ds.parquet")
        if result.tags_pos is not None:
            result.tags_pos.write_parquet(out / "tags_pos.parquet")
        if result.tags_ds is not None:
            result.tags_ds.write_parquet(out / "tags_ds.parquet")
        if result.dtm_pos is not None:
            result.dtm_pos.write_parquet(out / "dtm_pos.parquet")
        if result.dtm_ds is not None:
            result.dtm_ds.write_parquet(out / "dtm_ds.parquet")
        import json
        with (out / "manifest.json").open("w", encoding="utf-8") as f:
            json.dump(result.manifest, f, indent=2)
    except Exception as e:  # pragma: no cover
        raise ArtifactWriteError(f"Failed to write artifacts: {e}") from e
