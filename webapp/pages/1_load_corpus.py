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
import spacy
import streamlit as st

# Path setup is inherited from index.py entrypoint
project_root = pathlib.Path(__file__).parent.parents[1].resolve()  # Inherited path setup

from webapp.utilities.session import (  # noqa: E402
    get_or_init_user_session, init_session,
    generate_temp
    )
from webapp.utilities.session.metadata_handlers import (  # noqa: E402
    handle_target_metadata_processing
    )
from webapp.utilities.analysis import (  # noqa: E402
    find_saved, find_saved_reference
    )
from webapp.utilities.configuration import (  # noqa: E402
    import_options_general
    )
from webapp.utilities.processing import (  # noqa: E402
    process_external, process_internal,
    process_new, handle_uploaded_parquet,
    handle_uploaded_text, sidebar_process_section
    )
from webapp.utilities.ui import (  # noqa: E402
    load_and_display_target_corpus, render_corpus_info_expanders
)
from webapp.config.session_keys import (  # noqa: E402
    CorpusKeys,
    LoadCorpusKeys,
    MetadataKeys,
    SessionKeys,
    WarningKeys
    )
from webapp.menu import (   # noqa: E402
    menu,
    require_login
    )

# Set up paths for models and options
MODEL_LARGE = str(project_root.joinpath("webapp/_models/en_docusco_spacy"))
MODEL_SMALL = str(project_root.joinpath("webapp/_models/en_docusco_spacy_cd"))
OPTIONS = str(project_root.joinpath("webapp/config/options.toml"))

# Import global options from config file
_options = import_options_general(OPTIONS)

# Set global flags and limits
DESKTOP = _options['global']['desktop_mode']
CHECK_SIZE = _options['global']['check_size']
ENABLE_DETECT = _options['global']['check_language']
MAX_TEXT = _options['global']['max_bytes_text']
MAX_POLARS = _options['global']['max_bytes_polars']

TITLE = "Manage Corpus Data"
ICON = ":material/database:"

# Define labels and options for the app
CORPUS_SOURCES = ["Internal", "External", "New"]

# Button and form labels
LABEL_PROCESS_TARGET = "Process Target"
LABEL_PROCESS_REFERENCE = "Process Reference"
LABEL_UPLOAD_TARGET = "UPLOAD TARGET"
LABEL_UPLOAD_REFERENCE = "UPLOAD REFERENCE"
LABEL_RESET_CORPUS = "Reset Corpus"

# Model configuration
MODEL_LARGE_LABEL = "Large Dictionary"
MODEL_SMALL_LABEL = "Common Dictionary"
MODEL_OPTIONS = [MODEL_LARGE_LABEL, MODEL_SMALL_LABEL]

# Session state initialization template
STATES = {
    SessionKeys.METADATA_TARGET: {},
    SessionKeys.METADATA_REFERENCE: {},
    SessionKeys.SESSION_DATAFRAME: {},  # Container for SessionKeys DataFrame
    # 'data': {},     # Legacy key - consider removing if unused
    WarningKeys.LOAD_CORPUS: 0,
    LoadCorpusKeys.READY_TO_PROCESS: False,
    LoadCorpusKeys.CORPUS_DF: None,
    LoadCorpusKeys.EXCEPTIONS: None,
    LoadCorpusKeys.MODEL: None,
    LoadCorpusKeys.REF_READY_TO_PROCESS: False,
    LoadCorpusKeys.REF_CORPUS_DF: None,
    LoadCorpusKeys.REF_EXCEPTIONS: None,
}

st.set_page_config(
    page_title=TITLE, page_icon=ICON,
    layout="wide"
    )


# Cache spaCy models for efficiency
@st.cache_resource(show_spinner=False)
def load_models():
    """Load and cache spaCy models for efficient reuse."""
    large_model = spacy.load(MODEL_LARGE)
    small_model = spacy.load(MODEL_SMALL)
    models = {MODEL_LARGE_LABEL: large_model,
              MODEL_SMALL_LABEL: small_model}
    return models


def main() -> None:
    """
    Main function for the Load Corpus page.

    Handles corpus loading, processing, and management including:
    - Loading existing target and reference corpora
    - Processing new corpora from text files
    - Uploading external corpora (parquet files)
    - Managing corpus metadata and categories
    - Resetting corpus data
    """
    # Set login requirements for navigaton
    require_login()
    menu()
    st.markdown(f"## {TITLE}")

    # Get or initialize user session
    user_session_id, session = get_or_init_user_session()

    # Initialize processing state if not exists
    if LoadCorpusKeys.READY_TO_PROCESS not in st.session_state[user_session_id]:
        st.session_state[user_session_id][LoadCorpusKeys.READY_TO_PROCESS] = False

    st.sidebar.link_button(
        label="Help",
        url="https://browndw.github.io/docuscope-docs/guide/load-corpus.html",
        icon=":material/help:"
        )

    # If a target corpus is already loaded
    if session.get('has_target', [False])[0] is True:
        # Load and display corpus information
        load_and_display_target_corpus(session, user_session_id)

        # Get metadata for sidebar operations
        metadata_target = st.session_state[user_session_id]['metadata_target'].to_dict()

        # Sidebar: Target corpus management
        if not session.get(SessionKeys.HAS_META, [False])[0]:
            handle_target_metadata_processing(metadata_target, user_session_id)

        # If reference corpus is loaded, show info and warnings
        has_reference = session.get(SessionKeys.HAS_REFERENCE, [False])[0] is True
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
                    saved_corpora, saved_ref = find_saved_reference(  # noqa: E501
                        metadata_target.get(LoadCorpusKeys.MODEL)[0],
                        session.get(SessionKeys.TARGET_DB)[0]
                        )
                    to_load = st.sidebar.selectbox(
                        'Select a saved corpus to load:',
                        (sorted(saved_ref))
                        )
                    sidebar_process_section(
                        section_title=LABEL_PROCESS_REFERENCE,
                        button_label=LABEL_PROCESS_REFERENCE,
                        process_fn=lambda: process_internal(
                                saved_corpora.get(to_load),
                                user_session_id,
                                CorpusKeys.REFERENCE
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
                        st.session_state[user_session_id][WarningKeys.LOAD_CORPUS] = 0

                    # Use the helper function for upload and validation
                    tok_pl, ready = handle_uploaded_parquet(
                        ref_file, CHECK_SIZE, MAX_POLARS,
                        target_docs=metadata_target.get(MetadataKeys.DOCIDS)[0]['ids']
                    )

                    if ready:
                        st.session_state[user_session_id][LoadCorpusKeys.READY_TO_PROCESS] = True  # noqa: E501

                    # Sidebar UI for processing reference corpus
                    if st.session_state[user_session_id][LoadCorpusKeys.READY_TO_PROCESS]:
                        sidebar_process_section(
                            section_title=LABEL_PROCESS_REFERENCE,
                            button_label=LABEL_UPLOAD_REFERENCE,
                            process_fn=lambda: process_external(
                                tok_pl, user_session_id, CorpusKeys.REFERENCE
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

                    # Initialize variables with default values
                    corp_df, ready, exceptions = None, False, []

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
                            st.session_state[user_session_id][WarningKeys.LOAD_CORPUS] = 0

                        # Check text files to ensure they are valid
                        # and ready for processing
                        if submitted:  # Only process if form was submitted
                            corp_df, ready, exceptions = handle_uploaded_text(  # noqa: E501
                                ref_files,
                                CHECK_SIZE,
                                MAX_TEXT,
                                check_language_flag=ENABLE_DETECT,
                                check_ref=True,
                                target_docs=metadata_target.get(MetadataKeys.DOCIDS)[0]['ids']  # noqa: E501
                            )

                            # Store the reference corpus dataframe and exceptions
                            if ready and corp_df is not None:
                                st.session_state[user_session_id][LoadCorpusKeys.REF_CORPUS_DF] = corp_df  # noqa: E501
                                st.session_state[user_session_id][LoadCorpusKeys.REF_EXCEPTIONS] = exceptions  # noqa: E501

                    if ready:
                        st.session_state[user_session_id][LoadCorpusKeys.REF_READY_TO_PROCESS] = True  # noqa: E501

                    # Sidebar UI for processing reference corpus
                    if st.session_state[user_session_id][LoadCorpusKeys.REF_READY_TO_PROCESS]:  # noqa: E501
                        # Retrieve stored reference corpus data from session state
                        stored_ref_corp_df = st.session_state[user_session_id].get(LoadCorpusKeys.REF_CORPUS_DF)  # noqa: E501
                        stored_ref_exceptions = st.session_state[user_session_id].get(LoadCorpusKeys.REF_EXCEPTIONS)  # noqa: E501

                        models = load_models()
                        selected_dict = metadata_target.get('model')[0]
                        nlp = models[selected_dict]

                        sidebar_process_section(
                            section_title=LABEL_PROCESS_REFERENCE,
                            button_label=LABEL_PROCESS_REFERENCE,
                            process_fn=lambda: process_new(
                                stored_ref_corp_df,
                                nlp,
                                user_session_id,
                                CorpusKeys.REFERENCE,
                                stored_ref_exceptions
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
        if st.sidebar.button(label=LABEL_RESET_CORPUS,
                             icon=":material/refresh:"):
            st.session_state[user_session_id] = {}
            generate_temp(
                STATES.items(),
                user_session_id
                )
            init_session(
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

        render_corpus_info_expanders()

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
                saved_corpora = find_saved('ld')
                to_load = st.sidebar.selectbox(
                    'Select a saved corpus to load:',
                    (sorted(saved_corpora))
                    )
            if from_model == 'Common Dictionary':
                saved_corpora = find_saved('cd')
                to_load = st.sidebar.selectbox(
                    'Select a saved corpus to load:',
                    (sorted(saved_corpora))
                    )
            sidebar_process_section(
                section_title=LABEL_PROCESS_TARGET,
                button_label=LABEL_PROCESS_TARGET,
                process_fn=lambda: process_internal(
                        saved_corpora.get(to_load),
                        user_session_id,
                        CorpusKeys.TARGET
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
                    st.session_state[user_session_id][WarningKeys.LOAD_CORPUS] = 0

                # Use the helper function for upload and validation
                tok_pl, ready = handle_uploaded_parquet(
                    corp_file, CHECK_SIZE, MAX_POLARS
                )

            if ready:
                st.session_state[user_session_id][LoadCorpusKeys.READY_TO_PROCESS] = True

            # Sidebar UI for processing target corpus
            if st.session_state[user_session_id][LoadCorpusKeys.READY_TO_PROCESS]:
                sidebar_process_section(
                    section_title=LABEL_PROCESS_TARGET,
                    button_label=LABEL_PROCESS_TARGET,
                    process_fn=lambda: process_external(
                        tok_pl, user_session_id, CorpusKeys.TARGET
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

            # Initialize variables with default values
            corp_df, ready, exceptions = None, False, []

            with st.form("corpus-form", clear_on_submit=True):
                corp_files = st.file_uploader(
                    "Upload your target corpus",
                    type=["txt"],
                    accept_multiple_files=True
                )
                submitted = st.form_submit_button(LABEL_UPLOAD_TARGET)

                if submitted:
                    st.session_state[user_session_id][WarningKeys.LOAD_CORPUS] = 0
                if submitted and not corp_files:
                    st.warning(
                        "Please select at least one file to upload.",
                        icon=":material/warning:")

                # Check text files to ensure they are valid
                # and ready for processing
                if submitted:  # Only process if form was submitted
                    corp_df, ready, exceptions = handle_uploaded_text(  # noqa: E501
                        corp_files,
                        CHECK_SIZE,
                        MAX_TEXT,
                        check_language_flag=ENABLE_DETECT
                    )

                    # Store the corpus dataframe and exceptions in session state
                    if ready and corp_df is not None:
                        st.session_state[user_session_id][LoadCorpusKeys.CORPUS_DF] = corp_df  # noqa: E501
                        st.session_state[user_session_id][LoadCorpusKeys.EXCEPTIONS] = exceptions  # noqa: E501

            if ready:
                st.session_state[user_session_id][LoadCorpusKeys.READY_TO_PROCESS] = True

            # Sidebar UI for model selection and processing
            st.sidebar.markdown("### Models")
            models = load_models()
            selected_dict = st.sidebar.selectbox(
                "Select a DocuScope model:",
                options=MODEL_OPTIONS,
                help="The Large Dictionary model has a more eleaborated tagset than the Common Dictionary model. Click 'About the models' (on the right) to learn more.",  # noqa: E501
                )
            nlp = models[selected_dict]
            st.session_state[user_session_id][LoadCorpusKeys.MODEL] = selected_dict

            st.sidebar.markdown("---")

            if st.session_state[user_session_id][LoadCorpusKeys.READY_TO_PROCESS]:
                # Retrieve stored corpus data from session state
                stored_corp_df = st.session_state[user_session_id].get(LoadCorpusKeys.CORPUS_DF)  # noqa: E501
                stored_exceptions = st.session_state[user_session_id].get(LoadCorpusKeys.EXCEPTIONS)  # noqa: E501

                sidebar_process_section(
                    section_title=LABEL_PROCESS_TARGET,
                    button_label=LABEL_PROCESS_TARGET,
                    process_fn=lambda: process_new(
                        stored_corp_df,
                        nlp,
                        user_session_id,
                        CorpusKeys.TARGET,
                        stored_exceptions
                    ))


if __name__ == "__main__":
    main()
