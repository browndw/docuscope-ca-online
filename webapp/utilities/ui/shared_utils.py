"""
Shared utilities for UI components.

This module contains utility functions that are used by multiple UI modules
to avoid circular imports.
"""

import os
import re
import pandas as pd
import streamlit as st

from webapp.utilities.session.session_core import safe_session_get


def add_category_description(
        cat_counts: dict,
        session: dict = None,
        corpus_type: str = "target"  # "target" or "reference"
        ) -> pd.DataFrame:
    """
    Adds a 'Category Description' column to cat_df
    if the corpus is internal and a mapping exists.
    Also displays a documentation link button for
    internal corpora.
    """
    cat_df = pd.DataFrame(cat_counts.items(), columns=["Category", "Count"]).sort_values("Category")  # noqa: E501

    # Determine which session key to use
    db_key = f"{corpus_type}_db"
    target_db = safe_session_get(session, db_key, '')

    if not target_db:
        return cat_df

    corpus_name = os.path.basename(target_db)

    # Documentation links for each corpus family
    doc_links = {
        "MICUSP": "https://browndw.github.io/docuscope-docs/datasets/micusp.html",
        "BAWE": "https://browndw.github.io/docuscope-docs/datasets/bawe.html",
        "ELSEVIER": "https://browndw.github.io/docuscope-docs/datasets/elsevier.html",
        "HAPE": "https://browndw.github.io/docuscope-docs/datasets/hape.html",
    }

    # Map corpus name to doc link by checking which family it belongs to
    doc_link = None
    for key in doc_links:
        if key in corpus_name:
            doc_link = doc_links[key]
            break

    mappings = {
        "A_MICUSP_mini": {
            "BIO": "Biology", "CEE": "Civil and Environmental Engineering", "CLS": "Classical Studies",  # noqa: E501
            "ECO": "Economics", "EDU": "Education", "ENG": "English", "HIS": "History",
            "IOE": "Industrial and Operational Engineering", "LIN": "Linguistics", "MEC": "Mechanical Engineering",  # noqa: E501
            "NRE": "Natural Resources", "NUR": "Nursing", "PHI": "Philosophy", "PHY": "Physics",  # noqa: E501
            "POL": "Political Science", "PSY": "Psychology", "SOC": "Sociology"
            },
        "B_MICUSP": {
            "BIO": "Biology", "CEE": "Civil and Environmental Engineering", "CLS": "Classical Studies",  # noqa: E501
            "ECO": "Economics", "EDU": "Education", "ENG": "English", "HIS": "History",
            "IOE": "Industrial and Operational Engineering", "LIN": "Linguistics", "MEC": "Mechanical Engineering",  # noqa: E501
            "NRE": "Natural Resources", "NUR": "Nursing", "PHI": "Philosophy", "PHY": "Physics",  # noqa: E501
            "POL": "Political Science", "PSY": "Psychology", "SOC": "Sociology"
            },
        "C_BAWE_mini": {
            "AH": "Arts and Humanities", "LS": "Life Sciences", "PS": "Physical Sciences", "SS": "Social Sciences"  # noqa: E501
            },
        "D_BAWE": {
            "AH": "Arts and Humanities", "LS": "Life Sciences", "PS": "Physical Sciences", "SS": "Social Sciences"  # noqa: E501
            },
        "E_ELSEVIER": {
            "ARTS": "Arts and Humanities", "BIOC": "Biochemistry, Genetics and Molecular Biology",  # noqa: E501
            "BUSI": "Business, Management and Accounting", "CENG": "Chemical Engineering", "CHEM": "Chemistry",  # noqa: E501
            "COMP": "Computer Science", "DECI": "Decision Sciences", "ECON": "Economics, Econometrics and Finance",  # noqa: E501
            "ENGI": "Engineering", "ENVI": "Environmental Science", "HEAL": "Health Professions",  # noqa: E501
            "IMMU": "Immunology and Microbiology", "MATE": "Material Science", "MATH": "Mathematics",  # noqa: E501
            "MEDI": "Medicine", "NEUR": "Neuroscience", "NURS": "Nursing", "PHYS": "Physics and Astronomy",  # noqa: E501
            "PSYC": "Psychology", "SOCI": "Social Sciences"
            },
        "G_MICUSP_by_level": {
            "G0": "Final Year Undergraduate", "G1": "First Year Graduate",
            "G2": "Second Year Graduate", "G3": "Third Year Graduate"
            },
        "F_MICUSP_by_paper": {
            "CreativeWriting": "Narrative writing, poetry, drama scripts",
            "Critique": "Evaluation of business practices, problem–solution, literary critique, operations report",  # noqa: E501
            "Essay": "Argumentative essay, persuasive essay, literary analysis essay",
            "Proposal": "Research proposal, numeric model proposal, effective business/management design",  # noqa: E501
            "Report": "Lab report, literature review, article review, annotated bibliography, compare/contrast paper ",  # noqa: E501
            "ResearchPaper": "Research paper, replication study",
            "ResponsePaper": "Solution to a homework problem, personal response to a text ",
            },
        "H_HAPE_mini": {"gpt4": "Texts authored by ChatGPT-4o",
                        "human": "Texts authored by human writers",
                        "llama": "Texts authored by Llama 8B Instruct",
                        },
    }

    # Check if we have a mapping for this corpus
    if corpus_name in mappings:
        pattern = r'_([A-Z]+)(?:_|$)'
        corpus_base = re.search(pattern, corpus_name)
        if corpus_base:
            corpus_base = corpus_base.group(1)
        else:
            corpus_base = corpus_name
        # Display documentation link for internal corpora
        if doc_link:
            st.link_button(
                label=f"**About**: {corpus_base}",
                url=doc_link,
                icon=":material/info:"
            )
        # Add description column
        cat_df["Category Description"] = cat_df["Category"].map(mappings[corpus_name])
        # Move the Description column to be second
        cols = cat_df.columns.tolist()
        cols = [cols[0], cols[-1], cols[1]]
        cat_df = cat_df[cols]

    return cat_df
