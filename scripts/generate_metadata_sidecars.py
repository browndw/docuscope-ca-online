from __future__ import annotations

import argparse
import gzip
import json
import pickle
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def generate_sidecar(corpus_dir: Path, overwrite: bool) -> bool:
    """Generate a metadata sidecar for one corpus directory if ds_tokens.gz exists."""

    from webapp.utilities.session.session_core import (
        METADATA_DESCRIPTOR_FILE,
        write_metadata_descriptor_sidecar,
    )

    ds_tokens_path = corpus_dir / "ds_tokens.gz"
    sidecar_path = corpus_dir / METADATA_DESCRIPTOR_FILE

    if not ds_tokens_path.exists():
        return False
    if sidecar_path.exists() and not overwrite:
        return False

    with gzip.open(ds_tokens_path, "rb") as file_handle:
        ds_tokens = pickle.load(file_handle)

    write_metadata_descriptor_sidecar(corpus_dir, df=ds_tokens)

    return True


def main() -> int:
    from webapp.utilities.session.session_core import METADATA_DESCRIPTOR_FILE

    parser = argparse.ArgumentParser(
        description="Generate metadata descriptor sidecars for corpus directories."
    )
    parser.add_argument(
        "root",
        nargs="?",
        default="webapp/_corpora",
        help="Root directory containing corpus subdirectories.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate sidecars even if they already exist.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    generated = 0

    for corpus_dir in sorted(path for path in root.glob("*/*") if path.is_dir()):
        if generate_sidecar(corpus_dir, overwrite=args.overwrite):
            generated += 1
            print(f"generated {corpus_dir / METADATA_DESCRIPTOR_FILE}")

    print(json.dumps({"generated": generated, "root": str(root)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
