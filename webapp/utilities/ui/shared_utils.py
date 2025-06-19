"""
Shared utilities for UI components.

This module contains utility functions that are used by multiple UI modules
to avoid circular imports.
"""

import os
import pandas as pd
import streamlit as st


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
    target_db = session.get(db_key, [''])[0]
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
            "AGRI": "Agriculture", "ANTH": "Anthropology", "ARCH": "Architecture", "BIOL": "Biology",  # noqa: E501
            "BUSC": "Business", "CHEM": "Chemistry", "CLAS": "Classical Studies", "COMP": "Computer Science",  # noqa: E501
            "ECON": "Economics", "ENGI": "Engineering", "ENGL": "English Literature", "FOOD": "Food Science",  # noqa: E501
            "GEOG": "Geography", "HIST": "History", "HOSP": "Hospitality", "LING": "Linguistics",  # noqa: E501
            "MATH": "Mathematics", "MEDI": "Medicine", "PHYS": "Physics", "POLI": "Politics",  # noqa: E501
            "PSYC": "Psychology", "PUBL": "Publishing", "SOCL": "Sociology",
            "STAT": "Statistics"
        },
        "D_BAWE": {
            "AGRI": "Agriculture", "ANTH": "Anthropology", "ARCH": "Architecture", "BIOL": "Biology",  # noqa: E501
            "BUSC": "Business", "CHEM": "Chemistry", "CLAS": "Classical Studies", "COMP": "Computer Science",  # noqa: E501
            "ECON": "Economics", "ENGI": "Engineering", "ENGL": "English Literature", "FOOD": "Food Science",  # noqa: E501
            "GEOG": "Geography", "HIST": "History", "HOSP": "Hospitality", "LING": "Linguistics",  # noqa: E501
            "MATH": "Mathematics", "MEDI": "Medicine", "PHYS": "Physics", "POLI": "Politics",  # noqa: E501
            "PSYC": "Psychology", "PUBL": "Publishing", "SOCL": "Sociology",
            "STAT": "Statistics"
        },
        "E_ELSEVIER": {
            "EART": "Earth and Planetary Sciences", "ENGI": "Engineering", "MATE": "Materials Science",  # noqa: E501
            "COMP": "Computer Science", "ENER": "Energy", "ENVI": "Environmental Science", "IMMU": "Immunology"  # noqa: E501
        },
        "F_MICUSP_by_paper": {
            "Academic_Argument": "Academic Argument", "Creative_Writing": "Creative Writing",  # noqa: E501
            "Critique_Evaluation": "Critique/Evaluation", "Experimental_Report": "Experimental Report",  # noqa: E501
            "Proposal": "Proposal", "Research_Paper": "Research Paper", "Response_Paper": "Response Paper"  # noqa: E501
        },
        "G_MICUSP_by_level": {
            "First_Year": "First Year", "Second_Year": "Second Year", "Third_Year": "Third Year",  # noqa: E501
            "Fourth_Year": "Fourth Year", "Graduate": "Graduate"
        },
        "H_HAPE_mini": {
            "HIS": "History", "LIT": "Literature", "PHI": "Philosophy", "POL": "Politics"
        }
    }

    # Check if we have a mapping for this corpus
    if corpus_name in mappings:
        # Display documentation link for internal corpora
        if doc_link:
            st.markdown(f"**Corpus documentation**: [{corpus_name} Info]({doc_link})")

        # Add description column
        cat_df["Category Description"] = cat_df["Category"].map(mappings[corpus_name])
        # Move the Description column to be second
        cols = cat_df.columns.tolist()
        cols = [cols[0], cols[-1], cols[1]]
        cat_df = cat_df[cols]

    return cat_df
