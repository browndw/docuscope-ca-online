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

import pathlib
import sys

import polars as pl
import streamlit as st

# Ensure project root is in sys.path for both desktop and online
project_root = pathlib.Path(__file__).parent.parents[1].resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import webapp.utilities as _utils   # noqa: E402
from webapp.menu import menu, require_login   # noqa: E402

TITLE = "Compare Corpora"
ICON = ":material/compare_arrows:"

st.set_page_config(
    page_title=TITLE, page_icon=ICON,
    layout="wide"
    )


def main():
    # Set login requirements for navigaton
    require_login()
    menu()
    st.markdown(f"## {TITLE}")
    # Get or initialize user session
    user_session_id, session = _utils.handlers.get_or_init_user_session()

    st.sidebar.link_button(
        label="Help",
        url="https://browndw.github.io/docuscope-docs/guide/compare-corpora.html",
        icon=":material/help:"
        )

    if session.get('keyness_table')[0] is True:

        _utils.handlers.load_widget_state(
            pathlib.Path(__file__).stem,
            user_session_id
            )
        metadata_target = _utils.handlers.load_metadata(
            'target',
            user_session_id
            )
        metadata_reference = _utils.handlers.load_metadata(
            'reference',
            user_session_id
            )

        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown(
                _utils.content.message_target_info(metadata_target)
                )
        with col2:
            st.markdown(
                _utils.content.message_reference_info(metadata_reference)
                )

        # --- Show user selections ---
        st.info(
            f"**p-value threshold:** {st.session_state[user_session_id]['pval_threshold']} &nbsp;&nbsp; "  # noqa: E501
            f"**Swapped:** {'Yes' if st.session_state[user_session_id]['swap_target'] else 'No'}"  # noqa: E501
        )

        st.sidebar.markdown("### Comparison")
        table_radio = st.sidebar.radio(
            "Select the keyness table to display:",
            ("Tokens", "Tags Only"),
            key=_utils.handlers.persist(
                "kt_radio1", pathlib.Path(__file__).stem,
                user_session_id),
            horizontal=True)

        st.sidebar.markdown("---")

        if table_radio == 'Tokens':
            tag_radio = st.sidebar.radio(
                "Select tags to display:",
                ("Parts-of-Speech", "DocuScope"),
                key=_utils.handlers.persist(
                    "kt_radio2",
                    pathlib.Path(__file__).stem,
                    user_session_id),
                horizontal=True
                )

            if tag_radio == 'Parts-of-Speech':
                tag_type = st.sidebar.radio(
                    "Select from general or specific tags",
                    ("General", "Specific"),
                    horizontal=True
                    )

                if tag_type == 'General':
                    df = st.session_state[user_session_id]["target"]["kw_pos"]
                    df = _utils.analysis.freq_simplify_pl(df)
                else:
                    df = st.session_state[user_session_id]["target"]["kw_pos"]

            else:
                df = st.session_state[user_session_id]["target"]["kw_ds"]

            if df.height == 0 or df is None:
                cats = []
            elif df.height > 0:
                cats = sorted(df.get_column("Tag").unique().to_list())

            filter_vals = st.multiselect("Select tags to filter:", (cats))
            if len(filter_vals) > 0:
                df = df.filter(pl.col("Tag").is_in(filter_vals))

            st.dataframe(df,
                         hide_index=True,
                         column_config=_utils.formatters.get_streamlit_column_config(df)
                         )

            st.sidebar.markdown("---")

            _utils.formatters.toggle_download(
                label="Excel",
                convert_func=_utils.formatters.convert_to_excel,
                convert_args=(df.to_pandas(),),
                file_name="keywords_tokens.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                message=_utils.content.message_download,
                location=st.sidebar
            )

            st.sidebar.markdown("---")
            st.sidebar.markdown(_utils.content.message_reset_table)
            if st.sidebar.button("Generate New Keyness Table", icon=":material/refresh:"):
                # Clear keyness tables for this session
                for key in ["kw_pos", "kw_ds", "kt_pos", "kt_ds"]:
                    if key not in st.session_state[user_session_id]["target"]:
                        st.session_state[user_session_id]["target"][key] = {}
                    st.session_state[user_session_id]["target"][key] = {}
                # Reset keyness_table state
                _utils.handlers.update_session('keyness_table', False, user_session_id)
                # Optionally clear warnings
                st.session_state[user_session_id]["keyness_warning"] = None
                st.rerun()

            st.sidebar.markdown("---")

        else:
            st.sidebar.markdown("### Tagset")
            tag_radio_tags = st.sidebar.radio(
                "Select tags to display:",
                ("Parts-of-Speech", "DocuScope"),
                key=_utils.handlers.persist(
                    "kt_radio3",
                    pathlib.Path(__file__).stem,
                    user_session_id),
                horizontal=True)

            if tag_radio_tags == 'Parts-of-Speech':
                df = st.session_state[
                    user_session_id
                    ]["target"]["kt_pos"].filter(pl.col("Tag") != "FU")
            else:
                df = st.session_state[
                    user_session_id
                    ]["target"]["kt_ds"].filter(pl.col("Tag") != "Untagged")

            tab1, tab2 = st.tabs(["Keyness Table", "Keyness Plot"])
            with tab1:
                if df.height == 0 or df is None:
                    cats = []
                elif df.height > 0:
                    cats = sorted(df.get_column("Tag").unique().to_list())

                filter_vals = st.multiselect("Select tags to filter:", (cats))
                if len(filter_vals) > 0:
                    df = df.filter(pl.col("Tag").is_in(filter_vals))

                st.markdown("**Showing keywords that reach significance at *p* < 0.01**")

                st.dataframe(
                    df,
                    hide_index=True,
                    column_config=_utils.formatters.get_streamlit_column_config(df)
                    )

            with tab2:
                if df.height > 0 and df is not None:
                    fig = _utils.formatters.plot_compare_corpus_bar(df)
                    st.plotly_chart(fig, use_container_width=True)

            st.sidebar.markdown("---")

            _utils.formatters.toggle_download(
                label="Excel",
                convert_func=_utils.formatters.convert_to_excel,
                convert_args=(df.to_pandas(),),
                file_name="keywords_tags.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                message=_utils.content.message_download,
                location=st.sidebar
            )

            st.sidebar.markdown("---")
            st.sidebar.markdown(_utils.content.message_reset_table)
            if st.sidebar.button("Generate New Keyness Table", icon=":material/refresh:"):
                # Clear keyness tables for this session
                for key in ["kw_pos", "kw_ds", "kt_pos", "kt_ds"]:
                    if key not in st.session_state[user_session_id]["target"]:
                        st.session_state[user_session_id]["target"][key] = {}
                    st.session_state[user_session_id]["target"][key] = {}
                # Reset keyness_table state
                _utils.handlers.update_session('keyness_table', False, user_session_id)
                # Optionally clear warnings
                st.session_state[user_session_id]["keyness_warning"] = None
                st.rerun()

            st.sidebar.markdown("---")

    else:

        st.markdown(
            """
            :material/manufacturing: Use the button to generate a table.

            * To use this tool, be sure that you have loaded **a reference corpus**.

            * Loading a reference can be done from:.
            """
            )
        st.page_link(
            page="pages/1_load_corpus.py",
            label="Manage Corpus Data",
            icon=":material/database:",
        )

        # --- Add options for p-value threshold and swap target/reference ---
        # Initialize if not present
        if "pval_threshold" not in st.session_state[user_session_id]:
            st.session_state[user_session_id]["pval_threshold"] = 0.01
        if "swap_target" not in st.session_state[user_session_id]:
            st.session_state[user_session_id]["swap_target"] = False

        # Load metadata for size check
        try:
            metadata_target = _utils.handlers.load_metadata('target', user_session_id)
            target_tokens = metadata_target.get('tokens_pos', [0])[0] if metadata_target else 0  # noqa: E501
        except Exception:
            target_tokens = 0

        try:
            metadata_reference = _utils.handlers.load_metadata('reference', user_session_id)
            reference_tokens = metadata_reference.get('tokens_pos', [0])[0] if metadata_reference else 0  # noqa: E501
        except Exception:
            reference_tokens = 0

        TOKEN_LIMIT = 1_500_000

        if target_tokens > TOKEN_LIMIT or reference_tokens > TOKEN_LIMIT:
            pval_options = [0.05, 0.01]
            st.sidebar.warning(
                "Corpora are large (>1.5 million tokens). "
                "p < .001 is disabled to prevent memory issues."
            )
        elif target_tokens == 0 or reference_tokens == 0:
            # If metadata is missing, notthing to display
            pval_options = []
        else:
            pval_options = [0.05, 0.01, 0.001]
        # Select p-value threshold
        pval_idx = pval_options.index(st.session_state[user_session_id]["pval_threshold"]) \
            if st.session_state[user_session_id]["pval_threshold"] in pval_options else 1

        st.sidebar.markdown("### Select threshold")
        pval_selected = st.sidebar.selectbox(
            "p-value threshold",
            options=pval_options,
            format_func=lambda x: f"{x:.3f}" if x < 0.01 else f"{x:.2f}",
            index=pval_idx,
            key=f"pval_threshold_{user_session_id}",
            help=(
                "Select the p-value threshold for keyness analysis. "
                "Lower values are more stringent, but may be useful for larger corpora. "
                "For smaller corpora, a threshold of 0.05 is often sufficient."
            )
        )

        st.session_state[user_session_id]["pval_threshold"] = pval_selected

        if target_tokens > 0 and reference_tokens > 0:
            st.sidebar.markdown("### Swap target/reference corpora")
            swap_selected = st.sidebar.toggle(
                "Swap Target/Reference",
                value=st.session_state[user_session_id]["swap_target"],
                key=f"swap_target_{user_session_id}",
                help=(
                    "If selected, the target corpus will be used as the reference "
                    "and the reference corpus will be used as the target for keyness analysis. "  # noqa: E501
                    "This will show what is more frequent in the reference corpus "
                    "compared to the target corpus. "
                )
            )
            # Store the swap selection in session state
            st.session_state[user_session_id]["swap_target"] = swap_selected

        # --- End options block ---
        st.sidebar.markdown(_utils.content.message_generate_table)

        _utils.handlers.sidebar_action_button(
            button_label="Keyness Table",
            button_icon=":material/manufacturing:",
            preconditions=[
                session.get('has_target')[0],
                session.get('has_reference')[0]
            ],
            action=lambda: _utils.handlers.generate_keyness_tables(
                user_session_id,
                threshold=st.session_state[user_session_id]["pval_threshold"],
                swap_target=st.session_state[user_session_id]["swap_target"]
            ),
            spinner_message="Generating keywords..."
        )

        if st.session_state[user_session_id].get("keyness_warning"):
            msg, icon = st.session_state[user_session_id]["keyness_warning"]
            st.error(msg, icon=icon)

        st.sidebar.markdown("---")


if __name__ == "__main__":
    main()
