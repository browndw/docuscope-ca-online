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

    st.sidebar.link_button(
        label="Help",
        url="https://browndw.github.io/docuscope-docs/guide/load-corpus.html",
        icon=":material/help:"
        )

    # If a target corpus is already loaded
    if session.get('has_target')[0] is True:
        # Load target corpus metadata
        # Note that metadata is stored as a Polars DataFrame
        # and converted to a dictionary for easier access
        metadata_target = st.session_state[user_session_id]['metadata_target'].to_dict()  # noqa: E501

        # Check if reference is loaded
        has_reference = session.get('has_reference')[0] is True
        if has_reference:
            metadata_reference = _utils.handlers.load_metadata(
                CORPUS_REFERENCE,
                user_session_id
            )

        # Create tabs for Target and Reference
        tab_labels = ["Target corpus"]
        if has_reference:
            tab_labels.append("Reference corpus")
        tabs = st.tabs(tab_labels)

        # --- Target Tab ---
        with tabs[0]:
            st.info(_utils.content.message_target_info(metadata_target))
            with st.expander("Documents:"):
                st.write(metadata_target.get(KEY_DOCIDS)[0]['ids'])
            if session.get(KEY_HAS_META)[0] is True:
                st.markdown('##### Target corpus metadata:')
                cat_counts = Counter(metadata_target.get(KEY_DOCCATS)[0]['cats'])
                cat_df = _utils.formatters.add_category_description(
                    cat_counts,
                    session,
                    corpus_type="target")
                st.dataframe(cat_df, hide_index=True)

        # --- Reference Tab (if loaded) ---
        if has_reference:
            with tabs[1]:
                st.info(_utils.content.message_reference_info(metadata_reference))
                with st.expander("Documents in reference corpus:"):
                    st.write(metadata_reference.get(KEY_DOCIDS)[0]['ids'])

                # Try to process and display reference metadata if target has metadata
                if session.get(KEY_HAS_META)[0]:
                    try:
                        st.markdown('##### Reference corpus metadata:')
                        # Extract categories from doc ids using get_doc_cats
                        ref_doc_ids = metadata_reference.get(KEY_DOCIDS)[0]['ids']
                        doc_cats_ref = _utils.process.get_doc_cats(ref_doc_ids)
                        if doc_cats_ref:
                            cat_counts_ref = Counter(doc_cats_ref)
                            cat_df_ref = _utils.formatters.add_category_description(
                                cat_counts_ref,
                                session,
                                corpus_type="reference")
                            st.dataframe(cat_df_ref, hide_index=True)
                        else:
                            st.warning(
                                "Not categories found in reference corpus file names.",
                                icon=":material/info:"
                            )
                    except Exception:
                        st.warning(
                            "Could not process metadata for the reference corpus. "
                            "This may be due to missing or malformed category information.",
                            icon=":material/info:"
                        )

        if not session.get(KEY_HAS_META)[0]:
            st.sidebar.markdown('### Target corpus metadata:')
            load_cats = st.sidebar.radio(
                "Do you have categories in your file names to process?",
                ("No", "Yes"),
                horizontal=True,
                help=(
                    "Metadata can be encoded into your file names, "
                    "which can be used for further analysis. "
                    "The tool can detect information that comes before "
                    "the first underscore in the file name, and will "
                    "use that information to assign categories to your "
                    "documents. For example, if your file names are "
                    "`cat1_doc1.txt`, `cat2_doc2.txt`, etc., "
                    "the tool will assign `cat1` and `cat2` as categories. "
                    )
            )
            if load_cats == 'Yes':
                if st.sidebar.button(
                    label="Process Document Metadata",
                    icon=ICON_PROCESS_TARGET
                ):
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
        if not has_reference:
            # Reference corpus not loaded: offer options to load one
            st.markdown("---")
            st.markdown('##### Reference corpus:')
            load_ref = st.radio(
                "Would you like to load a reference corpus?",
                ("No", "Yes"),
                horizontal=True,
                help=(
                    "A reference corpus is a pre-processed corpus "
                    "or set of documents that you can use "
                    "to compare against your target corpus "
                    "with the **Compare Corpora** app. "
                    "If you choose to load a reference corpus, "
                    "be considered about the data that you choose. "
                    "What are trying to learn from the comparison?"
                    )
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
                    st.markdown(
                        """
                        :material/priority:
                        Select a saved corpus from the lists in the sidebar.

                        :material/priority: Only corpora tagged with the same model
                        as your target corpus will be available as a reference.
                        """
                        )
                    st.sidebar.markdown("### Reference corpora")
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
                    st.markdown(
                        """
                        :material/priority:
                        Use the widget to select the corpus you'd like to load,
                        either by browsing for them or dragging-and-dropping..

                        :material/priority:
                        Once you've selected your file,
                        click the **UPLOAD REFERENCE** button
                        and a processing button will appear in the sidebar.
                        """
                        )

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
                    st.markdown(
                        """
                        :material/priority:
                        Use the widget to **select the files**
                        you'd like process, either by browsing for them
                        or dragging-and-dropping.

                        :material/priority:
                        Once you've selected your files, click the **UPLOAD REFERENCE**
                        button and a processing button will appear in the sidebar.

                        :material/priority: Your reference will be tagged with
                        **the same model** as your target corpus.

                        :material/priority: Be sure that all file names are unique
                        and that they don't share names with your target corpus.

                        :material/timer: Processing times may vary, but you can expect
                        the initial corpus processing to take roughly
                        1 minute for every 1 million words.
                        """
                        )

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
        st.sidebar.markdown(
            body='### Reset all tools and files:'
            )
        st.sidebar.markdown(
            body=(
                ":warning: Using the **reset** button will cause "
                "all files, tables, and plots to be cleared."
            ),
            help=(
                "If you have any unsaved plots or tables "
                "that you'd like to retain, "
                "go back and save them before resetting."
            ))
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
        st.markdown(
            """
            * From this page you can **load a saved corpus** or **process a new one**
            by selecting the desired (**.txt**) files. You can also reset
            your target corpus or manage any corpora you have saved.

            * Once you have loaded a target corpus, you can add a **reference corpus**
            for comparison. Also note that you can encode metadata into your filenames,
            which can used for further analysis.
            (See the **About new corpora** expander.)
            """
            )

        st.markdown("##### :material/lightbulb: Learn more...")
        col_1, col_2, col_3, col_4 = st.columns(4)
        with col_1:
            # Expanders for corpus info
            with st.expander("About internal corpora", icon=":material/database:"):
                st.link_button(
                    label="MICUSP",
                    url="https://browndw.github.io/docuscope-docs/datasets/micusp.html",
                    icon=":material/quick_reference:")
                st.link_button(
                    label="BAWE",
                    url="https://browndw.github.io/docuscope-docs/datasets/bawe.html",
                    icon=":material/quick_reference:")
                st.link_button(
                    label="ELSEVIER",
                    url="https://browndw.github.io/docuscope-docs/datasets/elsevier.html",
                    icon=":material/quick_reference:")
                st.link_button(
                    label="HAP-E",
                    url="https://browndw.github.io/docuscope-docs/datasets/hape.html",
                    icon=":material/quick_reference:")
        with col_2:
            with st.expander("About external corpora", icon=":material/upload:"):
                st.link_button(
                    label="Preparing an external corpus",
                    url="https://browndw.github.io/docuscope-docs/vignettes/external-corpus.html",  # noqa: E501
                    icon=":material/quick_reference:")
        with col_3:
            with st.expander("About new corpora", icon=":material/library_books:"):
                st.link_button(
                    label="Preparing a new corpus",
                    url="https://browndw.github.io/docuscope-docs/vignettes/new-corpus.html",  # noqa: E501
                    icon=":material/quick_reference:")
        with col_4:
            with st.expander("About the models", icon=":material/modeling:"):
                st.link_button(
                    label="Compare models",
                    url="https://browndw.github.io/docuscope-docs/tagsets/model-comparison.html",  # noqa: E501
                    icon=":material/quick_reference:")
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
            index=None,
            help="Click on the expanders above to learn more about each option."
            )

        # Option 1: Load internal target corpus
        if corpus_source == 'Internal':
            st.markdown("---")
            st.markdown(
                """
                :material/priority:
                Select a saved corpus from the lists in the sidebar.

                :material/priority:  Note that corpora are organized by model
                with which they were tagged.
                """
                )
            st.sidebar.markdown("### Corpora")
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
            st.markdown(
                """
                :material/priority:
                Use the widget to select the corpus you'd like to load,
                either by browsing for them or dragging-and-dropping..

                :material/priority:
                Once you've selected your file,
                click the **UPLOAD TARGET** button
                and a processing button will appear in the sidebar.
                """
                )

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
            st.markdown(
                """
                :material/priority:
                Use the widget to **select the files**
                you'd like process, either by browsing for them
                or dragging-and-dropping.

                :material/priority:
                Once you've selected your files, click the **UPLOAD** button
                and a processing button will appear in the sidebar.

                :material/priority:
                Select **a model** from the sidebar.

                :material/priority:
                After processing, you will have the option
                to save your corpus to use for future analysis.

                :material/priority:
                Be sure that all file names are unique.

                :material/timer:
                Processing times may vary, but you can expect
                the initial corpus processing to take roughly
                1 minute for every 1 million words.
                """
                )

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
                options=MODEL_OPTIONS,
                help="The Large Dictionary model has a more eleaborated tagset than the Common Dictionary model. Click 'About the models' (on the right) to learn more.",  # noqa: E501
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
