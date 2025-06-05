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

# Copyright (C) 2025 David West Brown

import pathlib
import sys

import spacy
import streamlit as st
from collections import Counter

# Ensure project root is in sys.path for both desktop and online
project_root = pathlib.Path(__file__).parent.parents[1].resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import webapp.utilities as _utils   # noqa: E402
from webapp.menu import menu, require_login   # noqa: E402

# Set up paths for models and options
MODEL_LARGE = str(project_root.joinpath("webapp/_models/en_docusco_spacy"))
MODEL_SMALL = str(project_root.joinpath("webapp/_models/en_docusco_spacy_cd"))
OPTIONS = str(project_root.joinpath("webapp/options.toml"))

# Import global options from config file
_options = _utils.handlers.import_options_general(OPTIONS)

# Set global flags and limits
DESKTOP = _options['global']['desktop_mode']
CHECK_SIZE = _options['global']['check_size']
ENABLE_DETECT = _options['global']['check_language']
MAX_TEXT = _options['global']['max_bytes_text']
MAX_POLARS = _options['global']['max_bytes_polars']


TITLE = "Manage Corpus Data"
ICON = ":material/database:"

# --- Constants ---
KEY_READY_TO_PROCESS = 'ready_to_process'
KEY_WARNING = 'warning'
KEY_HAS_META = 'has_meta'
KEY_HAS_TARGET = 'has_target'
KEY_HAS_REFERENCE = 'has_reference'
KEY_EXCEPTIONS = 'exceptions'
KEY_REF_EXCEPTIONS = 'ref_exceptions'
KEY_MODEL = 'model'
KEY_DOCIDS = 'docids'
KEY_DOCCATS = 'doccats'
KEY_TARGET_DB = 'target_db'

CORPUS_TARGET = 'target'
CORPUS_REFERENCE = 'reference'
CORPUS_SOURCES = ["Internal", "External", "New"]

LABEL_PROCESS_TARGET = "Process Target"
LABEL_PROCESS_REFERENCE = "Process Reference"
LABEL_UPLOAD_TARGET = "UPLOAD TARGET"
LABEL_UPLOAD_REFERENCE = "UPLOAD REFERENCE"
LABEL_RESET_CORPUS = "Reset Corpus"

ICON_PROCESS_TARGET = ":material/manufacturing:"
ICON_PROCESS_REFERENCE = ":material/manufacturing:"
ICON_RESET = ":material/refresh:"

MODEL_LARGE_LABEL = "Large Dictionary"
MODEL_SMALL_LABEL = "Common Dictionary"
MODEL_OPTIONS = [MODEL_LARGE_LABEL, MODEL_SMALL_LABEL]

STATES = {
    'metadata_target': {},
    'metadata_reference': {},
    'session': {},
    'data': {},
    'warning': 0,
}

st.set_page_config(
    page_title=TITLE, page_icon=ICON,
    layout="wide"
    )


# Cache spaCy models for efficiency
@st.cache_resource(show_spinner=False)
def load_models():
    large_model = spacy.load(MODEL_LARGE)
    small_model = spacy.load(MODEL_SMALL)
    models = {MODEL_LARGE_LABEL: large_model,
              MODEL_SMALL_LABEL: small_model}
    return models


def main() -> None:
    # Set login requirements for navigaton
    require_login()
    menu()
    st.markdown(f"## {TITLE}")
    # Get or initialize user session
    user_session_id, session = _utils.handlers.get_or_init_user_session()

    if KEY_READY_TO_PROCESS not in st.session_state[user_session_id]:
        st.session_state[user_session_id][KEY_READY_TO_PROCESS] = False

    # If a target corpus is already loaded
    if session.get('has_target')[0] is True:
        # Load target corpus metadata
        # Note that metadata is stored as a Polars DataFrame
        # and converted to a dictionary for easier access
        metadata_target = st.session_state[user_session_id]['metadata_target'].to_dict()  # noqa: E501

        # If a reference corpus is also loaded
        if session.get('has_reference')[0] is True:
            metadata_reference = st.session_state[user_session_id]['metadata_reference'].to_dict()  # noqa: E501

        # Show info about the loaded target corpus
        st.markdown(_utils.content.message_target_info(metadata_target))

        # Expanders for document IDs and metadata
        with st.expander("Documents:"):
            st.write(metadata_target.get(KEY_DOCIDS)[0]['ids'])

        if session.get(KEY_HAS_META)[0] is True:
            st.markdown('##### Target corpus metadata:')
            with st.expander("Counts of document categories:"):
                st.write(Counter(metadata_target.get(KEY_DOCCATS)[0]['cats']))
        else:
            st.sidebar.markdown('### Target corpus metadata:')
            load_cats = st.sidebar.radio(
                "Do you have categories in your file names to process?",
                ("No", "Yes"),
                horizontal=True
            )
            if load_cats == 'Yes':
                if st.sidebar.button("Process Document Metadata"):
                    with st.spinner('Processing metadata...'):
                        doc_cats = _utils.process.get_doc_cats(
                            metadata_target.get(KEY_DOCIDS)[0]['ids']
                        )
                        if (
                            len(set(doc_cats)) > 1 and
                            len(set(doc_cats)) < 21
                        ):
                            _utils.handlers.update_metadata(
                                CORPUS_TARGET,
                                KEY_DOCCATS,
                                doc_cats,
                                user_session_id)
                            _utils.handlers.update_session(
                                KEY_HAS_META,
                                True,
                                user_session_id)
                            st.success('Processing complete!')
                            st.rerun()
                        elif len(doc_cats) != 0:
                            st.sidebar.warning(
                                """
                                Your data should contain at least 2 and
                                no more than 20 categories. You can either proceed
                                without assigning categories, or reset the corpus,
                                fix your file names, and try again.
                                """,
                                icon="material/info"
                            )
                        else:
                            st.sidebar.warning(
                                """
                                Your categories don't seem to be formatted correctly.
                                You can either proceed without assigning categories,
                                or reset the corpus, fix your file names, and try again.
                                """,
                                icon=":material/info:"
                                )

            st.sidebar.markdown("---")

        # If reference corpus is loaded, show info and warnings
        if session.get('has_reference')[0] is True:
            metadata_reference = _utils.handlers.load_metadata(
                CORPUS_REFERENCE,
                user_session_id
                )

            st.markdown(
                _utils.content.message_reference_info(metadata_reference)
                )

            with st.expander("Documents in reference corpus:"):
                st.write(metadata_reference.get(KEY_DOCIDS)[0]['ids'])

        else:
            # Reference corpus not loaded: offer options to load one
            st.markdown("---")
            st.markdown('##### Reference corpus:')
            load_ref = st.radio(
                "Would you like to load a reference corpus?",
                ("No", "Yes"),
                horizontal=True
                )

            st.markdown("---")

            if load_ref == 'Yes':
                # Choose reference corpus source
                ref_corpus_source = st.radio(
                    "What kind of reference corpus would you like to prepare?",
                    CORPUS_SOURCES,
                    captions=[
                        """:material/database:
                        Load a pre-processed corpus from the interface.
                        (Note that only MICUSP and ELSEVIER can be compared.)
                        """,
                        """:material/upload:
                        Upload a pre-processed corpus from your computer.
                        """,
                        """:material/library_books:
                        Process a new corpus from plain text files.
                        """
                        ],
                    horizontal=False,
                    index=None)

                st.markdown("---")

                # Option 1: Load internal reference corpus
                if ref_corpus_source == 'Internal':
                    st.markdown(_utils.content.message_select_reference)
                    st.sidebar.markdown(
                        """Use the button to load
                        a previously processed corpus.
                        """
                        )
                    saved_corpora, saved_ref = _utils.process.find_saved_reference(  # noqa: E501
                        metadata_target.get(KEY_MODEL)[0],
                        session.get(KEY_TARGET_DB)[0]
                        )
                    to_load = st.sidebar.selectbox(
                        'Select a saved corpus to load:',
                        (sorted(saved_ref))
                        )
                    _utils.process.sidebar_process_section(
                        section_title=LABEL_PROCESS_REFERENCE,
                        button_label=LABEL_PROCESS_REFERENCE,
                        button_icon=ICON_PROCESS_REFERENCE,
                        process_fn=lambda: _utils.process.process_internal(
                                saved_corpora.get(to_load),
                                user_session_id,
                                CORPUS_REFERENCE
                                ))

                # Option 2: Upload external reference corpus (parquet)
                if ref_corpus_source == 'External':
                    st.markdown(_utils.content.message_load_target_external)

                    with st.form("ref-file-form", clear_on_submit=True):
                        ref_file = st.file_uploader(
                            "Upload your reference corpus",
                            type=["parquet"],
                            accept_multiple_files=False
                            )
                        submitted = st.form_submit_button(
                            LABEL_UPLOAD_REFERENCE
                            )

                    if submitted:
                        st.session_state[user_session_id][KEY_WARNING] = 0

                    # Use the helper function for upload and validation
                    tok_pl, ready = _utils.process.handle_uploaded_parquet(
                        ref_file, CHECK_SIZE, MAX_POLARS,
                        target_docs=metadata_target.get(KEY_DOCIDS)[0]['ids']
                    )

                    if ready:
                        st.session_state[user_session_id][KEY_READY_TO_PROCESS] = True  # noqa: E501

                    # Sidebar UI for processing reference corpus
                    if st.session_state[user_session_id][KEY_READY_TO_PROCESS]:
                        _utils.process.sidebar_process_section(
                            section_title=LABEL_PROCESS_REFERENCE,
                            button_label=LABEL_UPLOAD_REFERENCE,
                            process_fn=lambda: _utils.process.process_external(
                                tok_pl, user_session_id, CORPUS_REFERENCE
                            ))

                # Option 3: Process new reference corpus from text files
                if ref_corpus_source == 'New':
                    st.markdown(_utils.content.message_load_reference)

                    with st.form("ref-form", clear_on_submit=True):
                        ref_files = st.file_uploader(
                            "Upload your reference corpus",
                            type=["txt"],
                            accept_multiple_files=True,
                            key='reffiles'
                        )
                        submitted = st.form_submit_button(
                            LABEL_UPLOAD_REFERENCE
                            )

                        if submitted:
                            st.session_state[user_session_id][KEY_WARNING] = 0

                        # Check text files to ensure they are valid
                        # and ready for processing
                        corp_df, ready, exceptions = _utils.process.handle_uploaded_text(  # noqa: E501
                            ref_files,
                            CHECK_SIZE,
                            MAX_TEXT,
                            check_language_flag=ENABLE_DETECT,
                            check_ref=True,
                            target_docs=metadata_target.get(KEY_DOCIDS)[0]['ids']  # noqa: E501
                        )

                    if ready:
                        st.session_state[user_session_id][KEY_READY_TO_PROCESS] = True  # noqa: E501

                    # Sidebar UI for processing reference corpus
                    if st.session_state[user_session_id][KEY_READY_TO_PROCESS]:  # noqa: E501
                        models = load_models()
                        selected_dict = metadata_target.get('model')[0]
                        nlp = models[selected_dict]

                        _utils.process.sidebar_process_section(
                            section_title=LABEL_PROCESS_REFERENCE,
                            button_label=LABEL_PROCESS_REFERENCE,
                            button_icon=ICON_PROCESS_REFERENCE,
                            process_fn=lambda: _utils.process.process_new(
                                corp_df,
                                nlp,
                                user_session_id,
                                CORPUS_REFERENCE,
                                exceptions
                            ))

        # Sidebar: Reset all tools and files
        st.sidebar.markdown('### Reset all tools and files:')
        st.sidebar.markdown(""":warning:
                            Using the **reset** button will cause
                            all files, tables, and plots to be cleared.
                            """)
        if st.sidebar.button(label=LABEL_RESET_CORPUS, icon=ICON_RESET):
            st.session_state[user_session_id] = {}
            _utils.handlers.generate_temp(
                STATES.items(),
                user_session_id
                )
            _utils.handlers.init_session(
                user_session_id
                )
            st.rerun()
        st.sidebar.markdown("""---""")

    else:
        # No target corpus loaded: show options and info
        st.markdown("###  :dart: Load or process a target corpus")
        st.markdown(_utils.content.message_load)
        st.markdown("---")

        # Expanders for corpus info
        with st.expander("About internal corpora", icon=":material/database:"):
            st.markdown(_utils.content.message_internal_corpora)
        with st.expander("About external corpora", icon=":material/upload:"):
            st.markdown(_utils.content.message_external_corpora)
        with st.expander("About new corpora", icon=":material/library_books:"):
            st.markdown(_utils.content.message_naming)
        st.markdown("---")
        st.markdown("### Process a corpus:")

        # Choose corpus source
        corpus_source = st.radio(
            "What kind of corpus would you like to prepare?",
            CORPUS_SOURCES,
            captions=[
                """:material/database:
                Load a pre-processed corpus from the interface.
                """,
                """:material/upload:
                Upload a pre-processed corpus from your computer.
                """,
                """:material/library_books:
                Process a new corpus from plain text files.
                """
                ],
            horizontal=False,
            index=None
            )

        # Option 1: Load internal target corpus
        if corpus_source == 'Internal':
            st.markdown("---")
            st.markdown(_utils.content.message_load_target_internal)
            st.sidebar.markdown(
                """
                Use the button to load a previously processed corpus.
                """
                )
            from_model = st.sidebar.radio(
                "Select data tagged with:",
                MODEL_OPTIONS,
                key='corpora_to_load'
                )
            if from_model == 'Large Dictionary':
                saved_corpora = _utils.process.find_saved('ld')
                to_load = st.sidebar.selectbox(
                    'Select a saved corpus to load:',
                    (sorted(saved_corpora))
                    )
            if from_model == 'Common Dictionary':
                saved_corpora = _utils.process.find_saved('cd')
                to_load = st.sidebar.selectbox(
                    'Select a saved corpus to load:',
                    (sorted(saved_corpora))
                    )
            _utils.process.sidebar_process_section(
                section_title=LABEL_PROCESS_TARGET,
                button_label=LABEL_PROCESS_TARGET,
                button_icon=ICON_PROCESS_TARGET,
                process_fn=lambda: _utils.process.process_internal(
                        saved_corpora.get(to_load),
                        user_session_id,
                        CORPUS_TARGET
                        ))

        # Option 2: Upload external target corpus (parquet)
        if corpus_source == 'External':
            st.markdown("---")
            st.markdown(_utils.content.message_load_target_external)

            with st.form("corpus-file-form", clear_on_submit=True):
                corp_file = st.file_uploader(
                    "Upload your target corpus",
                    type=["parquet"],
                    accept_multiple_files=False
                )
                # Submit button for file upload
                submitted = st.form_submit_button(LABEL_UPLOAD_TARGET)

                if submitted:
                    st.session_state[user_session_id][KEY_WARNING] = 0

                # Use the helper function for upload and validation
                tok_pl, ready = _utils.process.handle_uploaded_parquet(
                    corp_file, CHECK_SIZE, MAX_POLARS
                )

            if ready:
                st.session_state[user_session_id][KEY_READY_TO_PROCESS] = True

            # Sidebar UI for processing target corpus
            if st.session_state[user_session_id][KEY_READY_TO_PROCESS]:
                _utils.process.sidebar_process_section(
                    section_title=LABEL_PROCESS_TARGET,
                    button_label=LABEL_PROCESS_TARGET,
                    button_icon=ICON_PROCESS_TARGET,
                    process_fn=lambda: _utils.process.process_external(
                        tok_pl, user_session_id, CORPUS_TARGET
                    ))

        # Option 3: Process new target corpus from text files
        if corpus_source == 'New':
            st.markdown("---")
            st.markdown(_utils.content.message_load_target_new)

            with st.form("corpus-form", clear_on_submit=True):
                corp_files = st.file_uploader(
                    "Upload your target corpus",
                    type=["txt"],
                    accept_multiple_files=True
                )
                submitted = st.form_submit_button(LABEL_UPLOAD_TARGET)

                if submitted:
                    st.session_state[user_session_id][KEY_WARNING] = 0
                if submitted and not corp_files:
                    st.warning(
                        "Please select at least one file to upload.",
                        icon=":material/warning:")

                # Check text files to ensure they are valid
                # and ready for processing
                corp_df, ready, exceptions = _utils.process.handle_uploaded_text(  # noqa: E501
                    corp_files,
                    CHECK_SIZE,
                    MAX_TEXT,
                    check_language_flag=ENABLE_DETECT
                )

            if ready:
                st.session_state[user_session_id][KEY_READY_TO_PROCESS] = True

            # Sidebar UI for model selection and processing
            st.sidebar.markdown("### Models")
            models = load_models()
            selected_dict = st.sidebar.selectbox(
                "Select a DocuScope model:",
                options=MODEL_OPTIONS
                )
            nlp = models[selected_dict]
            st.session_state[user_session_id][KEY_MODEL] = selected_dict

            with st.sidebar.expander("Which model do I choose?"):
                st.markdown(_utils.content.message_models)

            st.sidebar.markdown("---")

            if st.session_state[user_session_id][KEY_READY_TO_PROCESS]:
                _utils.process.sidebar_process_section(
                    section_title=LABEL_PROCESS_TARGET,
                    button_label=LABEL_PROCESS_TARGET,
                    button_icon=ICON_PROCESS_TARGET,
                    process_fn=lambda: _utils.process.process_new(
                        corp_df,
                        nlp,
                        user_session_id,
                        CORPUS_TARGET,
                        exceptions
                    ))


if __name__ == "__main__":
    main()
