"""
Corpus discovery functions for finding saved corpora.

This module contains functions for discovering and locating saved corpus
files and reference corpora.
"""

import os
import pathlib

# Ensure project root is in sys.path
project_root = pathlib.Path(__file__).parent.parents[2].resolve()

# Set corpus directory
CORPUS_DIR = project_root.joinpath("webapp/_corpora")


def find_saved(model_type: str) -> dict:
    """
    Find saved corpora for a given model type.

    Parameters
    ----------
    model_type : str
        The model type directory to search (e.g., 'cd', 'ld').

    Returns
    -------
    dict
        Dictionary mapping corpus names to their paths.
    """
    SUB_DIR = CORPUS_DIR.joinpath(model_type)
    if not SUB_DIR.exists():
        return {}

    saved_paths = [f.path for f in os.scandir(SUB_DIR) if f.is_dir()]
    saved_names = [f.name for f in os.scandir(SUB_DIR) if f.is_dir()]
    saved_corpora = {
        saved_names[i]: saved_paths[i] for i in range(len(saved_names))
        }
    return saved_corpora


def find_saved_reference(target_model: str, target_path: str) -> tuple[dict, dict]:
    """
    Find saved reference corpora that can be compared to the target.

    Parameters
    ----------
    target_model : str
        The target model name.
    target_path : str
        The path to the target corpus.
        
    Returns
    -------
    tuple[dict, dict]
        A tuple containing:
        - Dictionary of all saved corpora for the model type
        - Dictionary of reference corpora (excluding target corpus type)
    """
    # Only allow comparisons of ELSEVIER to MICUSP
    target_base = os.path.splitext(
        os.path.basename(pathlib.Path(target_path))
        )[0]
    if "MICUSP" in target_base:
        corpus = "MICUSP"
    else:
        corpus = "ELSEVIER"
    model_type = ''.join(word[0] for word in target_model.lower().split())
    SUB_DIR = CORPUS_DIR.joinpath(model_type)
    
    if not SUB_DIR.exists():
        return {}, {}
        
    saved_paths = [f.path for f in os.scandir(SUB_DIR) if f.is_dir()]
    saved_names = [f.name for f in os.scandir(SUB_DIR) if f.is_dir()]
    saved_corpora = {
        saved_names[i]: saved_paths[i] for i in range(len(saved_names))
        }
    saved_ref = {
        key: val for key, val in saved_corpora.items() if corpus not in key
        }

    return saved_corpora, saved_ref
