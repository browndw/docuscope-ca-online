# Copyright (C) 2025 David West Brown

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import pathlib
import sys

import gzip
import glob
import pickle
import polars as pl
import random
import re
import unidecode
import docuscospacy as ds
import streamlit as st
from collections import OrderedDict
from lingua import Language, LanguageDetectorBuilder

# Ensure project root is in sys.path for both desktop and online
project_root = pathlib.Path(__file__).parent.parents[1].resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from webapp.utilities import handlers   # noqa: E402

CORPUS_DIR = project_root.joinpath("webapp/_corpora")

WARNING_CORRUPT_TARGET = 10
WARNING_CORRUPT_REFERENCE = 11
WARNING_DUPLICATE_REFERENCE = 21
WARNING_EXCLUDED_TARGET = 40
WARNING_EXCLUDED_REFERENCE = 41


@st.cache_resource(show_spinner=False)
def load_detector():
    detector = LanguageDetectorBuilder.from_all_languages().with_low_accuracy_mode().build()  # noqa: E501
    return detector


def load_corpus_internal(db_path: str,
                         session_id: str,
                         corpus_type='target'):
    """
    Load a corpus from the specified database path into the session state.

    Parameters
    ----------
    db_path : str
        The path to the database containing the corpus files.
    session_id : str
        The session ID for which the corpus is to be loaded.
    corpus_type : str, optional
        The type of corpus to be loaded (default is 'target').

    Returns
    -------
    None
    """
    if corpus_type not in st.session_state[session_id]:
        st.session_state[session_id][corpus_type] = {}
    files_list = glob.glob(os.path.join(db_path, '*.gz'))
    random.shuffle(files_list)
    data = {}
    for file in files_list:
        try:
            with gzip.open(file, 'rb') as f:
                data[
                    str(os.path.basename(file)).removesuffix(".gz")
                    ] = pickle.load(f)
        except Exception:
            pass
    if len(data) != 7:
        random.shuffle(files_list)
        data = {}
        for file in files_list:
            try:
                with gzip.open(file, 'rb') as f:
                    data[
                        str(os.path.basename(file)).removesuffix(".gz")
                        ] = pickle.load(f)
            except Exception:
                pass
    else:
        for key, value in data.items():
            if key not in st.session_state[session_id][corpus_type]:
                st.session_state[session_id][corpus_type][key] = {}
            st.session_state[session_id][corpus_type][key] = value


def load_corpus_new(ds_tokens: pl.DataFrame,
                    dtm_ds: pl.DataFrame,
                    dtm_pos: pl.DataFrame,
                    ft_ds: pl.DataFrame,
                    ft_pos: pl.DataFrame,
                    tt_ds: pl.DataFrame,
                    tt_pos: pl.DataFrame,
                    session_id: str,
                    corpus_type='target') -> None:
    """
    Load new corpus dataframes into the session state
    for a given session ID and corpus type.

    Parameters
    ----------
    ds_tokens : pl.DataFrame
        The dataframe containing token-level data for the corpus.
    dtm_ds : pl.DataFrame
        The document-term matrix for dictionary tags.
    dtm_pos : pl.DataFrame
        The document-term matrix for POS tags.
    ft_ds : pl.DataFrame
        The frequency table for dictionary tags.
    ft_pos : pl.DataFrame
        The frequency table for POS tags.
    tt_ds : pl.DataFrame
        The tags table for dictionary tags.
    tt_pos : pl.DataFrame
        The tags table for POS tags.
    session_id : str
        The session ID for which the corpus is to be loaded.
    corpus_type : str, optional
        The type of corpus to be loaded (default is 'target').

    Returns
    -------
    None
    """
    if corpus_type not in st.session_state[session_id]:
        st.session_state[session_id][corpus_type] = {}

    if "ds_tokens" not in st.session_state[session_id][corpus_type]:
        st.session_state[session_id][corpus_type]["ds_tokens"] = {}
    st.session_state[session_id][corpus_type]["ds_tokens"] = ds_tokens

    if "dtm_ds" not in st.session_state[session_id][corpus_type]:
        st.session_state[session_id][corpus_type]["dtm_ds"] = {}
    st.session_state[session_id][corpus_type]["dtm_ds"] = dtm_ds

    if "dtm_pos" not in st.session_state[session_id][corpus_type]:
        st.session_state[session_id][corpus_type]["dtm_pos"] = {}
    st.session_state[session_id][corpus_type]["dtm_pos"] = dtm_pos

    if "ft_ds" not in st.session_state[session_id][corpus_type]:
        st.session_state[session_id][corpus_type]["ft_ds"] = {}
    st.session_state[session_id][corpus_type]["ft_ds"] = ft_ds

    if "ft_pos" not in st.session_state[session_id][corpus_type]:
        st.session_state[session_id][corpus_type]["ft_pos"] = {}
    st.session_state[session_id][corpus_type]["ft_pos"] = ft_pos

    if "tt_ds" not in st.session_state[session_id][corpus_type]:
        st.session_state[session_id][corpus_type]["tt_ds"] = {}
    st.session_state[session_id][corpus_type]["tt_ds"] = tt_ds

    if "tt_pos" not in st.session_state[session_id][corpus_type]:
        st.session_state[session_id][corpus_type]["tt_pos"] = {}
    st.session_state[session_id][corpus_type]["tt_pos"] = tt_pos


def find_saved(model_type: str):
    SUB_DIR = CORPUS_DIR.joinpath(model_type)
    saved_paths = [f.path for f in os.scandir(SUB_DIR) if f.is_dir()]
    saved_names = [f.name for f in os.scandir(SUB_DIR) if f.is_dir()]
    saved_corpora = {
        saved_names[i]: saved_paths[i] for i in range(len(saved_names))
        }
    return saved_corpora


def find_saved_reference(target_model,
                         target_path):
    # only allow comparisions of ELSEVIER to MICUSP
    target_base = os.path.splitext(
        os.path.basename(pathlib.Path(target_path))
        )[0]
    if "MICUSP" in target_base:
        corpus = "MICUSP"
    else:
        corpus = "ELSEVIER"
    model_type = ''.join(word[0] for word in target_model.lower().split())
    SUB_DIR = CORPUS_DIR.joinpath(model_type)
    saved_paths = [f.path for f in os.scandir(SUB_DIR) if f.is_dir()]
    saved_names = [f.name for f in os.scandir(SUB_DIR) if f.is_dir()]
    saved_corpora = {
        saved_names[i]: saved_paths[i] for i in range(len(saved_names))
        }
    saved_ref = {
        key: val for key, val in saved_corpora.items() if corpus not in key
        }

    return saved_corpora, saved_ref


def corpus_from_widget(
        docs
        ) -> tuple[pl.DataFrame, list]:
    """
    Process uploaded files from a widget and return
    a Polars DataFrame and a list of exceptions.

    Parameters
    ----------
    docs : iterable
        Iterable of file-like objects with .name and .getvalue() methods.

    Returns
    -------
    tuple
        (Polars DataFrame with columns 'doc_id' and 'text',
        list of filenames that failed to decode)
    """
    exceptions = []
    records = []
    for doc in docs:
        try:
            doc_txt = doc.getvalue().decode('utf-8')
            doc_txt = unidecode.unidecode(doc_txt)
            doc_id = str(os.path.splitext(doc.name.replace(" ", ""))[0])
            records.append({"doc_id": doc_id, "text": doc_txt})
        except Exception:
            exceptions.append(doc.name)

    if records:
        df = pl.DataFrame(records)
        df = (
            df.with_columns(
                pl.col("text").str.strip_chars()
            )
            .sort("doc_id")
        )
    else:
        df = pl.DataFrame({"doc_id": [], "text": []})

    return df, exceptions


def check_language(
        text_str: str
        ) -> bool:
    """
    Check if the given text is in English based on language detection model.

    Parameters
    ----------
    text_str : str
        The text string to be checked.
    detect_model : LanguageDetectorBuilder
        The language detection model used to compute language confidence.
    detect_language : Language
        The target language to detect.

    Returns
    -------
    bool
        True if the text is detected as English with high confidence.
    """
    detect_model = load_detector()
    detect_language = Language.ENGLISH
    # Clean up the text string
    doc_len = len(text_str)
    predictions = []
    if doc_len > 5000:
        idx_a = random.randint(0, doc_len - 1500)
        idx_b = random.randint(0, doc_len - 1500)
        idx_c = random.randint(0, doc_len - 1500)
        sample_a = text_str[idx_a:idx_a + 1000]
        sample_a = " ".join(sample_a.split())
        sample_b = text_str[idx_b:idx_b + 1000]
        sample_b = " ".join(sample_b.split())
        sample_c = text_str[idx_c:idx_c + 1000]
        sample_c = " ".join(sample_c.split())
        text_sample = [sample_a, sample_b, sample_c]
        # get prediction for each chunk
        for chunk in text_sample:  # Language predict each sampled chunk
            value = detect_model.compute_language_confidence(
                chunk,
                detect_language
                )
            predictions.append(value)
    else:
        text_str = " ".join(text_str.split())
        value = detect_model.compute_language_confidence(
            text_str,
            detect_language
            )
        predictions.append(value)

    confidence = sum(predictions) / len(predictions)

    # Only want to know if this is English or not.
    return confidence > .9


def check_schema(
        tok_pl: pl.DataFrame
        ) -> bool:
    """
    Validate the schema of a Polars DataFrame.

    Parameters
    ----------
    tok_pl : pl.DataFrame
        A Polars DataFrame containing the corpus data.

    Returns
    -------
    bool
        True if the schema of the DataFrame matches the expected schema,
        False otherwise.
    """
    validation = OrderedDict(
        [
            ('doc_id', pl.String),
            ('token', pl.String),
            ('pos_tag', pl.String),
            ('ds_tag', pl.String),
            ('pos_id', pl.UInt32),
            ('ds_id', pl.UInt32)
            ])
    return tok_pl.schema == validation


def check_corpus_new(
        docs: list,
        check_size=False,
        check_language_flag=False,
        check_ref=False,
        target_docs=None
        ) -> tuple:
    """
    Check the corpus for duplicates, size, reference documents, and language.

    Parameters
    ----------
    docs : list
        A list of document objects to be checked.
    check_size : bool, optional
        If True, calculate the total size of the corpus (default is False).
    check_language_flag : bool, optional
        If True, check the language of the documents (default is False).
    check_ref : bool, optional
        If True, check for reference documents in the corpus (default is False).
    target_docs : list, optional
        A list of target document identifiers to check against (default is None).
    detect_model : LanguageDetectorBuilder, optional
        The language detection model.
    detect_language : Language, optional
        The target language to detect.

    Returns
    -------
    tuple
        A tuple containing:
        - dup_ids (list): A list of duplicate document identifiers.
        - dup_docs (list, optional): A list of documents found in both the
          corpus and target_docs (only if check_ref is True).
        - lang_fail (list, optional): A list of doc_ids that fail the language check
          (only if check_language_flag is True).
        - corpus_size (int, optional): The total size of the corpus in bytes
          (only if check_size is True).
    """  # noqa: E501
    if len(docs) > 0:
        all_files = []
        if check_size:
            for file in docs:
                bytes_data = file.getvalue()
                file_size = len(bytes_data)
                all_files.append(file_size)
            corpus_size = sum(all_files)
        # check for duplicates
        doc_ids = [str(os.path.splitext(doc.name)[0]) for doc in docs]
        doc_ids = [doc.replace(" ", "") for doc in doc_ids]
        if len(doc_ids) > len(set(doc_ids)):
            dup_ids = [x for x in doc_ids if doc_ids.count(x) >= 2]
            dup_ids = list(set(dup_ids))
        else:
            dup_ids = []
        if check_ref and target_docs is not None:
            dup_docs = list(set(target_docs).intersection(doc_ids))
        else:
            dup_docs = []
        if check_language_flag:
            # Check language of each document
            lang_fail = []
            for doc in docs:
                try:
                    doc_txt = doc.getvalue().decode('utf-8')
                except Exception:
                    lang_fail.append(
                        str(os.path.splitext(doc.name.replace(" ", ""))[0])
                        )
                    continue
                if not check_language(doc_txt):
                    lang_fail.append(
                        str(os.path.splitext(doc.name.replace(" ", ""))[0])
                        )
        else:
            lang_fail = []
    else:
        corpus_size = 0
        dup_ids = []
        dup_docs = []
        lang_fail = []
    # Compose return tuple based on which checks are enabled
    result = [dup_ids]
    if check_ref:
        result.append(dup_docs)
    if check_language_flag:
        result.append(lang_fail)
    if check_size:
        result.append(corpus_size)
    if len(result) == 1:
        return result[0]
    return tuple(result)


def check_corpus_external(
        tok_pl: pl.DataFrame,
        check_size=False,
        check_ref=False,
        target_docs=None
        ) -> tuple | bool:
    """
    Check the corpus for schema validation, duplicates, size,
    and reference documents.

    Parameters
    ----------
    tok_pl : pl.DataFrame or None
        A Polars DataFrame containing the corpus data, or None.
    check_size : bool, optional
        If True, calculate the total size of the corpus (default is False).
    check_ref : bool, optional
        If True, check for reference documents in the corpus
        (default is False).
    target_docs : list, optional
        A list of target document identifiers to check against
        (default is None).

    Returns
    -------
    tuple or bool
        Returns a tuple containing:
        - is_valid (bool): Whether the schema of the DataFrame is valid.
        - dup_docs (list, optional): Documents found in both the corpus and target_docs (if check_ref).
        - corpus_size (int, optional): Total size of the corpus in bytes (if check_size).
        If no options are enabled, returns just is_valid.
    """  # noqa: E501
    if tok_pl is None:
        # Return the correct number of outputs based on flags
        if check_ref and check_size:
            return False, [], 0
        elif check_ref:
            return False, []
        elif check_size:
            return False, 0
        else:
            return False

    is_valid = check_schema(tok_pl)
    result = [is_valid]

    if check_ref and target_docs is not None:
        doc_ids = tok_pl.get_column("doc_id").unique().to_list()
        dup_docs = list(set(target_docs).intersection(doc_ids))
        result.append(dup_docs)
    elif check_ref:
        result.append([])

    if check_size:
        corpus_size = tok_pl.estimated_size()
        result.append(corpus_size)

    if len(result) == 1:
        return result[0]
    return tuple(result)


def get_doc_cats(
        doc_ids: list
        ) -> list:
    """
    Extract document categories from a list of document identifiers.

    Parameters
    ----------
    doc_ids : list
        A list of document identifiers.

    Returns
    -------
    list
        A list of document categories extracted from the document identifiers.
    """
    if all(['_' in item for item in doc_ids]):
        doc_cats = [re.sub(
            r"_\S+$", "", item, flags=re.UNICODE
            ) for item in doc_ids]
        if min([len(item) for item in doc_cats]) == 0:
            doc_cats = []
    else:
        doc_cats = []
    return doc_cats


def handle_uploaded_parquet(
        uploaded_file: pl.DataFrame | None,
        check_size: bool,
        max_size: int,
        target_docs=None
        ) -> tuple[pl.DataFrame | None, bool, list]:
    """Read a parquet file and check corpus validity, size,
    and (optionally) duplicates."""
    if uploaded_file is not None:
        try:
            df = pl.read_parquet(uploaded_file)
        except Exception as e:
            st.error(f"Failed to read file: {e}")
            return None, False, []
    else:
        df = None

    check_kwargs = dict(tok_pl=df)
    if check_size:
        check_kwargs['check_size'] = True
    if target_docs is not None:
        check_kwargs['check_ref'] = True
        check_kwargs['target_docs'] = target_docs

    result = check_corpus_external(**check_kwargs)

    # Unpack result based on which checks are enabled
    if check_size and target_docs is not None:
        is_valid, dup_docs, corpus_size = result
    elif check_size:
        is_valid, corpus_size = result
        dup_docs = []
    elif target_docs is not None:
        is_valid, dup_docs = result
        corpus_size = 0
    else:
        is_valid = result
        dup_docs = []
        corpus_size = 0

    # Only show format error if a file was uploaded and is invalid
    if uploaded_file is not None and not is_valid:
        st.error(
            """
            Your pre-processed corpus is not in the correct format.
            You can try selecting a different file or processing your corpus
            from the original text files and saving it again.
            """,
            icon=":material/block:"
        )
    if check_size and corpus_size > max_size:
        st.error(
            """
            Your corpus is too large for online processing.
            The online version of DocuScope Corpus Analysis & Concordancer
            accepts data up to roughly 3 million words.
            If you'd like to process more data, try
            [the desktop version of the tool](https://github.com/browndw/docuscope-cac)
            which available for free.
            """,  # noqa: E501
            icon=":material/disc_full:"
            )
    if target_docs is not None and len(dup_docs) > 0:
        st.error(
            f"""
            The files you selected could not be processed.
            Files with these names were also submitted
            as part of your target corpus:
            ```
            {sorted(dup_docs)}
            ```
            Plese remove files from your reference corpus before processing.
            To clear this warning click the **UPLOAD REFERENCE** button.
            """,
            icon=":material/block:"
        )

    ready = (
        is_valid and
        df is not None and
        df.is_empty() is False and
        (corpus_size <= max_size if check_size else True) and
        (len(dup_docs) == 0 if target_docs is not None else True)
    )

    if ready:
        st.success(
            """Success! Your corpus is ready to be processed.
            Use the **Process** button in the sidebar to continue.
            """,
            icon=":material/celebration:"
        )
    return df, ready


def handle_uploaded_text(
        uploaded_files: list,
        check_size: bool,
        max_size: int,
        check_language_flag=False,
        check_ref=False,
        target_docs=None
        ) -> tuple[pl.DataFrame | None, bool, list]:
    """
    Handle uploaded text files, run check_corpus_new,
    and return (DataFrame, ready, exceptions).
    """
    if not uploaded_files or len(uploaded_files) == 0:
        # No files uploaded
        return None, False, []

    # Prepare kwargs for check_corpus_new
    check_kwargs = dict(docs=uploaded_files)
    if check_size:
        check_kwargs['check_size'] = True
    if check_language_flag:
        check_kwargs['check_language_flag'] = True
    if check_ref:
        check_kwargs['check_ref'] = True
        check_kwargs['target_docs'] = target_docs

    result = check_corpus_new(**check_kwargs)

    # Unpack result based on which options are enabled
    dup_ids, dup_docs, lang_fail, corpus_size = [], [], [], 0
    if check_ref and check_size and check_language_flag:
        dup_ids, dup_docs, lang_fail, corpus_size = result
    elif check_ref and check_size:
        dup_ids, dup_docs, corpus_size = result
        lang_fail = []
    elif check_ref and check_language_flag:
        dup_ids, dup_docs, lang_fail = result
        corpus_size = 0
    elif check_ref:
        dup_ids, dup_docs = result
        lang_fail = []
        corpus_size = 0
    elif check_size and check_language_flag:
        dup_ids, lang_fail, corpus_size = result
        dup_docs = []
    elif check_size:
        dup_ids, corpus_size = result
        dup_docs = []
        lang_fail = []
    elif check_language_flag:
        dup_ids, lang_fail = result
        dup_docs = []
        corpus_size = 0
    else:
        dup_ids = result
        dup_docs = []
        lang_fail = []
        corpus_size = 0

    # Streamlit error handling (for user feedback)
    if len(dup_ids) > 0:
        st.error(
            f"""
            The files you selected could not be processed.
            Your corpus contains these duplicate file names:
            ```
            {sorted(dup_ids)}
            ```
            Plese remove duplicates before processing.
            To clear this warning click the **UPLOAD** button.
            """,
            icon=":material/block:"
        )
    if check_ref and len(dup_docs) > 0:
        st.error(
            f"""
            The files you selected could not be processed.
            Files with these names were also submitted
            as part of your target corpus:
            ```
            {sorted(dup_docs)}
            ```
            Plese remove files from your reference corpus before processing.
            To clear this warning click the **UPLOAD REFERENCE** button.
            """,
            icon=":material/block:"
        )
    if check_language_flag and len(lang_fail) > 0:
        st.error(
            f"""
            The files you selected could not be processed.
            Files with these names are either not in English or
            are incompatible with the reqirement of the model:
            ```
            {sorted(lang_fail)}
            ```
            Plese remove files from your corpus before processing.
            To clear this warning click the **UPLOAD TARGET** button.
            """,
            icon=":material/translate:"
        )
    if check_size and corpus_size > max_size:
        st.error(
            """
            Your corpus is too large for online processing.
            The online version of DocuScope Corpus Analysis & Concordancer
            accepts data up to roughly 3 million words.
            If you'd like to process more data, try
            [the desktop version of the tool](https://github.com/browndw/docuscope-cac)
            which available for free.
            """,  # noqa: E501
            icon=":material/disc_full:"
        )

    # Determine readiness
    ready = (
        len(uploaded_files) > 0 and
        len(dup_ids) == 0 and
        (len(dup_docs) == 0 if check_ref else True) and
        (corpus_size <= max_size if check_size else True) and
        (len(lang_fail) == 0 if check_language_flag else True)
    )

    # Only create DataFrame if ready
    if ready:
        st.success(
            f"""Success!
            **{len(uploaded_files)}** corpus files ready!
            Use the **Process** button in the sidebar to continue.
            """,
            icon=":material/celebration:"
        )
        df, exceptions = corpus_from_widget(uploaded_files)
    else:
        df, exceptions = None, []

    return df, ready, exceptions


def finalize_corpus_load(ds_tokens, user_session_id, corpus_type):
    ft_pos, ft_ds = ds.frequency_table(ds_tokens, count_by="both")
    tt_pos, tt_ds = ds.tags_table(ds_tokens, count_by="both")
    dtm_pos, dtm_ds = ds.tags_dtm(ds_tokens, count_by="both")
    load_corpus_new(
        ds_tokens,
        dtm_ds, dtm_pos,
        ft_ds, ft_pos,
        tt_ds, tt_pos,
        user_session_id, corpus_type
    )
    if corpus_type == 'target':
        handlers.init_metadata_target(user_session_id)
        handlers.update_session('has_target', True, user_session_id)
    else:
        handlers.init_metadata_reference(user_session_id)
        handlers.update_session('has_reference', True, user_session_id)
    st.rerun()


def process_new(
        corp_df,
        nlp,
        user_session_id,
        corpus_type,
        exceptions=None
        ):
    ds_tokens = ds.docuscope_parse(corp=corp_df, nlp_model=nlp)
    if exceptions and ds_tokens.is_empty():
        st.session_state[user_session_id]['warning'] = (
            WARNING_CORRUPT_TARGET if corpus_type == 'target' else WARNING_CORRUPT_REFERENCE  # noqa: E501
        )
        st.rerun()
    elif exceptions:
        st.session_state[user_session_id]['warning'] = (
            WARNING_EXCLUDED_TARGET if corpus_type == 'target' else WARNING_EXCLUDED_REFERENCE  # noqa: E501
        )
        st.session_state[user_session_id]['exceptions'] = exceptions
        finalize_corpus_load(ds_tokens, user_session_id, corpus_type)
    else:
        st.success('Processing complete!')
        st.session_state[user_session_id]['warning'] = 0
        finalize_corpus_load(ds_tokens, user_session_id, corpus_type)


def process_external(
        df,
        user_session_id,
        corpus_type
        ):
    # For external (preprocessed) corpora, no parsing/model needed
    ds_tokens = df
    finalize_corpus_load(ds_tokens, user_session_id, corpus_type)


def process_internal(
        corp_path,
        user_session_id,
        corpus_type
        ):
    load_corpus_internal(
        corp_path,
        user_session_id,
        corpus_type=corpus_type
    )
    if corpus_type == "target":
        handlers.init_metadata_target(user_session_id)
        handlers.update_session(
            'target_db',
            str(corp_path),
            user_session_id
            )
        handlers.update_session(
            'has_target',
            True,
            user_session_id
            )
    else:
        handlers.init_metadata_reference(user_session_id)
        handlers.update_session(
            'reference_db',
            str(corp_path),
            user_session_id
            )
        handlers.update_session(
            'has_reference',
            True,
            user_session_id
            )
    st.rerun()


def sidebar_process_section(
    section_title: str,
    button_label: str,
    button_icon: str,
    process_fn,
    spinner_text: str = "Processing corpus data..."
) -> None:
    """
    Helper to standardize sidebar processing UI.

    Parameters
    ----------
        section_title (str): The sidebar section title.
        button_label (str): The label for the action button.
        process_fn (callable): Function to call when button is pressed.
        spinner_text (str): Text to show in the spinner.
    """
    st.sidebar.markdown(f"### {section_title}")
    st.sidebar.markdown(
        """
        Once you have selected your files,
        use the button to process your corpus.
        """)
    if st.sidebar.button(button_label, icon=button_icon):
        with st.sidebar, st.spinner(spinner_text):
            process_fn()
    st.sidebar.markdown("---")
