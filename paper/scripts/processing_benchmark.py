#!/usr/bin/env python3
"""
DocuScope CA performance benchmark using
the same pipeline as the application.
This replicates the real process_new() function workflow.
"""

import time
import sys
import psutil
import polars as pl
import unidecode
import os
from pathlib import Path

# Add the webapp directory to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def load_test_corpus_real_format():
    """Load test corpus in the exact format the application expects."""
    corpus_path = Path("paper/data/test_corpus")

    records = []
    exceptions = []

    print("Loading corpus files in application format...")
    for txt_file in sorted(corpus_path.glob("*.txt")):
        try:
            # Exact same process as corpus_from_widget()
            with open(txt_file, 'r', encoding='utf-8') as f:
                doc_txt = f.read()

            # Apply same transformations as the app
            doc_txt = unidecode.unidecode(doc_txt)
            doc_id = str(os.path.splitext(txt_file.name.replace(" ", ""))[0])
            records.append({"doc_id": doc_id, "text": doc_txt})

        except Exception:
            exceptions.append(txt_file.name)

    if records:
        # Exact same DataFrame construction as the app
        df = pl.DataFrame(records)
        df = (
            df.with_columns(
                pl.col("text").str.strip_chars()
            )
            .sort("doc_id")
        )
    else:
        df = pl.DataFrame({"doc_id": [], "text": []})

    print(f"✓ Loaded {len(df)} documents")
    print(f"✓ Total words: {sum(len(text.split()) for text in df['text'])}")
    print(f"✓ Exceptions: {len(exceptions)}")
    print()

    return df, exceptions


def benchmark_complete_pipeline():
    """Benchmark the complete DocuScope processing pipeline."""
    print("=" * 60)
    print("Authentic DocuScope CA Pipeline Benchmark")
    print("=" * 60)
    print()

    # System info
    print("System Information:")
    print(f"✓ CPU cores: {psutil.cpu_count()}")
    print(f"✓ Available memory: {psutil.virtual_memory().total / 1024 / 1024 / 1024:.1f} GB")  # noqa: E501
    print(f"✓ Python version: {sys.version.split()[0]}")
    print()

    # Step 1: Load corpus in application format
    start_time = time.time()
    corp_df, exceptions = load_test_corpus_real_format()
    load_time = time.time() - start_time

    if corp_df.is_empty():
        print("❌ No corpus data loaded")
        return

    # Step 2: Load spaCy model (exactly like the app)
    print("Loading spaCy model...")
    model_start_memory = get_memory_usage()
    model_start_time = time.time()

    try:
        import spacy
        # Load the actual model the app uses
        nlp = spacy.load("webapp/_models/en_docusco_spacy")
        model_load_time = time.time() - model_start_time
        model_memory = get_memory_usage() - model_start_memory

        print(f"✓ Model loaded: {model_load_time:.2f} seconds")
        print(f"✓ Model memory: {model_memory:.1f} MB")
        print()

    except Exception as e:
        print(f"❌ Could not load DocuScope model: {e}")
        return

    # Step 3: Process with DocuScope (exactly like process_new())
    print("Processing corpus with DocuScope...")
    process_start_memory = get_memory_usage()
    process_start_time = time.time()

    try:
        import docuscospacy as ds

        # This is the EXACT function call from process_new()
        ds_tokens = ds.docuscope_parse(corp=corp_df, nlp_model=nlp)

        process_time = time.time() - process_start_time
        process_memory = get_memory_usage() - process_start_memory

        # Analyze results
        total_tokens = len(ds_tokens) if not ds_tokens.is_empty() else 0
        total_words = sum(len(text.split()) for text in corp_df['text'])

        print(f"✓ Processing completed: {process_time:.2f} seconds")
        print(f"✓ Generated tokens: {total_tokens:,}")
        print(f"✓ Input words: {total_words:,}")
        print(f"✓ Processing memory: {process_memory:.1f} MB")
        print()

        # Performance metrics
        docs_per_second = len(corp_df) / process_time
        tokens_per_second = total_tokens / process_time
        words_per_minute = (total_words / process_time) * 60

        # Total pipeline time
        total_time = load_time + model_load_time + process_time

        print("=" * 60)
        print("AUTHENTIC PERFORMANCE METRICS")
        print("=" * 60)
        print(f"Corpus loading: {load_time:.2f} seconds")
        print(f"Model loading: {model_load_time:.1f} seconds (one-time startup)")
        print(f"DocuScope processing: {process_time:.1f} seconds")
        print(f"Total pipeline: {total_time:.1f} seconds")
        print()
        print(f"Document processing: {docs_per_second:.1f} docs/second")
        print(f"Token annotation: {tokens_per_second:,.0f} tokens/second")
        print(f"Word processing: {words_per_minute:,.0f} words/minute")
        print()
        print(f"Memory usage - Model: {model_memory:.0f} MB")
        print(f"Memory usage - Processing: {process_memory:.0f} MB")
        print(f"Total corpus: {len(corp_df)} documents, {total_words:,} words")
        print()

        # Validate the 1-minute-per-million-words claim
        million_word_time = 1000000 / words_per_minute
        print(f"Time to process 1 million words: {million_word_time:.1f} minutes")
        print("(Application claims ~1 minute/million words)")

    except Exception as e:
        print(f"❌ DocuScope processing failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run the complete authentic benchmark."""
    benchmark_complete_pipeline()


if __name__ == "__main__":
    main()
