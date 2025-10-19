"""Command line interface for DocuScope CA headless processing.

Examples:

  docuscope-ca process --input paper/data/test_corpus --metrics freq,tags --out out_dir

Exit Codes:
  0 success
  2 corpus load error (no docs)
  3 model load error
  4 processing error (parse / metrics)
  5 export error
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .api import (
    process_corpus,
    CorpusLoadError,
    ModelLoadError,
    CorpusProcessingError,
    ArtifactWriteError,
)


def _add_process(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("process", help="Process a corpus headlessly and export artifacts")
    p.add_argument(
        "--input",
        required=True,
        help="Directory, file, or comma-separated list of files / raw texts",
    )
    p.add_argument(
        "--model",
        default="en_docusco_spacy",
        help="spaCy DocuScope model name or path",
    )
    p.add_argument(
        "--metrics",
        default="freq,tags",
        help="Comma-separated metrics subset of freq,tags,dtm",
    )
    p.add_argument(
        "--out",
        default="docuscope_output",
        help="Output directory (created if missing)",
    )
    p.add_argument(
        "--no-export",
        action="store_true",
        help="Do not write files; print summary JSON only",
    )
    p.add_argument(
        "--manifest",
        action="store_true",
        help="Print full manifest JSON to stdout",
    )
    p.set_defaults(func=_cmd_process)


def _cmd_process(args: argparse.Namespace) -> int:
    metrics = tuple(m.strip() for m in args.metrics.split(",") if m.strip())
    sources: Sequence[str] | str
    if "," in args.input and not Path(args.input).exists():
        sources = [s.strip() for s in args.input.split(",") if s.strip()]
    else:
        sources = args.input
    try:
        result = process_corpus(
            sources=sources,
            model=args.model,
            metrics=metrics,
            export_dir=None if args.no_export else args.out,
        )
    except CorpusLoadError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 2
    except ModelLoadError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 3
    except ArtifactWriteError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 5
    except CorpusProcessingError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 4

    if args.manifest:
        print(json.dumps(result.manifest, indent=2))
    else:
        print(
            json.dumps(
                {
                    "documents": result.manifest["corpus"]["n_documents"],
                    "total_tokens": result.manifest["corpus"]["total_tokens"],
                    "metrics": result.manifest["metrics"],
                    "timings": result.timings,
                },
                indent=2,
            )
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="docuscope-ca", description="DocuScope CA headless tools"
    )
    sub = parser.add_subparsers(dest="command")
    _add_process(sub)
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
