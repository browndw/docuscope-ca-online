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

import base64
import io
import pathlib
import sys

import polars as pl
import streamlit as st
from matplotlib import pyplot as plt

# Ensure project root is in sys.path for both desktop and online
project_root = pathlib.Path(__file__).parent.parents[1].resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import webapp.utilities as _utils   # noqa: E402
from webapp.menu import menu, require_login   # noqa: E402

TITLE = "AI-Assisted Analysis"
ICON = ":material/smart_toy:"

st.set_page_config(
    page_title=TITLE, page_icon=ICON,
    layout="wide"
    )

OPTIONS = str(project_root.joinpath("webapp/options.toml"))
_options = _utils.handlers.import_options_general(OPTIONS)

DESKTOP = _options['global']['desktop_mode']
LLM_PARAMS = _options['llm']['llm_parameters']
LLM_MODEL = _options['llm']['llm_model']
QUOTA = _options['llm']['quota']

if DESKTOP:
    CACHE = False
else:
    CACHE = _options['cache']['cache_mode']


def main():
    # Set login requirements for navigaton
    require_login()
    menu()
    st.markdown(f"## {TITLE}")
    # Get or initialize user session
    user_session_id, session = _utils.handlers.get_or_init_user_session()

    st.sidebar.link_button(
        label="Help",
        url="https://browndw.github.io/docuscope-docs/guide/assisted-analysis.html",
        icon=":material/help:"
        )

    if session.get('has_target')[0] is True:
        metadata_target = st.session_state[
            user_session_id
            ]['metadata_target'].to_dict()

    if "pandasai" not in st.session_state[user_session_id]:
        st.session_state[user_session_id]["pandasai"] = []

    if "user_key" not in st.session_state[user_session_id]:
        st.session_state[user_session_id]["user_key"] = None

    if DESKTOP is False:
        try:
            community_key = st.secrets["openai"]["api_key"]
        except Exception:
            community_key = None
    else:
        community_key = None

    if CACHE:
        daily_tokens = _utils.cache.get_query_count(st.user.email)
        if daily_tokens >= QUOTA:
            community_key = None

    if community_key is not None:
        api_key = community_key
    else:
        api_key = st.session_state[user_session_id]["user_key"]

    if session.get('tags_table')[0] is True:

        st.markdown(_utils.content.message_pandabot_home)

        _utils.handlers.load_widget_state(
            pathlib.Path(__file__).stem,
            user_session_id
            )

        st.markdown("### Data Selection")

        with st.expander("About data",
                         icon=":material/table:",
                         expanded=False):

            st.markdown(_utils.content.message_plotbot_data)

        pandasai_corpus = st.radio(
            "Select corpus:",
            ("Target", "Reference", "Keywords", "Grouped"),
            key=_utils.handlers.persist(
                "pandasai_corpus", pathlib.Path(__file__).stem,
                user_session_id),
            horizontal=True
                )

        if session.get('has_meta')[0] is True:
            groups = metadata_target.get('doccats')[0]['cats']
        elif session.get('has_meta')[0] is False:
            groups = []

        pandasai_query = st.selectbox(
                "Select data to analyze:",
                (_utils.llms.tables_to_list(session_id=user_session_id,
                                            corpus=pandasai_corpus,
                                            categories=groups)),
                key=_utils.handlers.persist(
                    "pandasai_query", pathlib.Path(__file__).stem,
                    user_session_id),
                index=None,
                placeholder="Select data..."
                )

        st.markdown("### Data Preview")
        df = _utils.llms.table_from_list(user_session_id,
                                         pandasai_corpus,
                                         pandasai_query,
                                         categories=groups)

        if df is not None and df.shape[0] > 0:
            if (
                df.get_column("Group", default=None) is not None or
                str("Document-Term Matrix") in pandasai_query
            ):
                col1, col2 = st.columns(2)

                with col1:
                    pivot_table = st.toggle(
                        "Pivot Table",
                        key=_utils.handlers.persist(
                            "pivot_table", pathlib.Path(__file__).stem,
                            user_session_id)
                    )
                    if (
                        pivot_table and
                        str("Document-Term Matrix") in pandasai_query and
                        df.get_column("Group", default=None) is None
                    ):
                        df = df.pivot(
                            "Tag",
                            index="doc_id",
                            values="RF"
                            )
                    elif (
                        pivot_table and
                        str("Document-Term Matrix") in pandasai_query and
                        df.get_column("Group", default=None) is not None
                    ):
                        df = df.pivot(
                            "Tag",
                            index=["doc_id", "Group"],
                            values="RF"
                            )
                    elif pivot_table:
                        df = df.pivot(
                            "Tag",
                            index="Group",
                            values="RF"
                            )

                with col2:
                    make_percent = st.toggle(
                        "Make Percent",
                        key=_utils.handlers.persist(
                            "make_percent", pathlib.Path(__file__).stem,
                            user_session_id),
                        value=False
                    )
                    if (
                        make_percent and
                        str("Document-Term Matrix") in pandasai_query
                    ):
                        df = df.with_columns(
                            (
                                pl.selectors.numeric().mul(100)
                                )
                        )
                    elif (
                        make_percent and
                        str("Tags Table") in pandasai_query
                    ):
                        df = df.with_columns(
                            (
                                pl.selectors.numeric()
                                .exclude(["AF", "Range"]).mul(100)
                                )
                        )

            st.dataframe(df.head(10), use_container_width=True)

        if st.session_state[user_session_id]["pandasai"]:
            for i, message in enumerate(
                st.session_state[user_session_id]["pandasai"]
            ):
                with st.chat_message(message["role"]):
                    if message['type'] == 'dataframe':
                        st.dataframe(
                            message['value'],
                            use_container_width=True
                            )
                    elif message['type'] == 'number':
                        st.markdown(
                            str(message['value']),
                            unsafe_allow_html=True
                            )
                    elif message['type'] == 'string':
                        st.markdown(
                            message['value'],
                            unsafe_allow_html=True
                            )
                    elif message['type'] == 'plot':
                        buf = io.BytesIO()
                        plt.imsave(buf,
                                   message['value'],
                                   format='png')
                        buf.seek(0)
                        img_bytes = buf.getvalue()

                        st.image(
                            img_bytes
                            )

                        b64 = base64.b64encode(img_bytes).decode()
                        href = f'<a href="data:image/png;base64,{b64}" download="plot.png">Download PNG</a>'  # noqa: E501
                        st.markdown(href, unsafe_allow_html=True)

                    elif message['type'] == 'error':
                        st.markdown(
                            message['value'],
                            unsafe_allow_html=True
                            )
                    else:
                        st.markdown(
                            """:thinking_face: Sorry,
                            I had a problem executing your pompt.
                            """,
                            unsafe_allow_html=True
                        )

        if st.session_state[user_session_id]["pandasai"]:
            prompt_position = sum(
                1 for message in st.session_state[user_session_id]["pandasai"]
                if message["role"] == "user"
                ) + 1
        else:
            prompt_position = 1

        if api_key is not None:

            prompt = st.chat_input("Enter your prompt:")

            if (
                prompt and
                (df is not None and df.shape[0] > 0)
            ):
                with st.spinner(":sparkles: Generating response..."):
                    _utils.llms.pandabot_user_query(
                        df=df.to_pandas(),
                        prompt=prompt,
                        prompt_position=prompt_position,
                        api_key=api_key,
                        session_id=user_session_id,
                        cache_mode=CACHE,
                        )

                st.rerun()

            elif (
                prompt and
                (df is None or df.shape[0] == 0)
            ):
                st.error(
                    """
                    Please select a data frame to analyze.
                    """
                    )

        else:
            st.markdown(
                """
                You need to enter your OpenAI API key
                to use Pandabot.
                If you don't have one, you can get it
                from [OpenAI](https://platform.openai.com/signup).
                """
            )

            user_api_key = st.text_input(
                "Enter your OpenAI API key:",
                type='password'
                )

            if st.button("Check API Key"):
                if user_api_key:
                    if _utils.llms.is_openai_key_valid(user_api_key):
                        st.success("API key is valid!")
                        st.session_state[
                            user_session_id
                            ]["user_key"] = user_api_key
                        st.rerun()
                    else:
                        st.error(
                            "API key is invalid. Please check and try again."
                            )
                else:
                    st.warning("Please enter an API key.")

        if CACHE:
            remaining_tokens = max(0, QUOTA - daily_tokens)
            used_tokens_pct = int((daily_tokens / QUOTA) * 100)

            st.sidebar.markdown(
                f" ##### Daily queries - used: `{min(QUOTA, daily_tokens)}` - remaining: `{remaining_tokens}`"  # noqa: E501
                )
            st.sidebar.progress(min(used_tokens_pct / 100, 1.0))

            if remaining_tokens < 10 and remaining_tokens > 0:
                st.warning(
                    f"""
                    You have only `{remaining_tokens}` queries left
                    for today using the community API.
                    When these run out, you will need login
                    with your own OpenAI API key.
                    If you don't have one, you can get it
                    from [OpenAI](https://platform.openai.com/signup).
                    """,
                    icon=":material/warning:"
                    )

            elif remaining_tokens <= 0:
                st.error(
                    """
                    You have used all your queries for today
                    using the community API.
                    Please login with your own OpenAI API key
                    to continue using the AI-assisted tools.
                    If you don't have one, you can get it
                    from [OpenAI](https://platform.openai.com/signup).
                    """,
                    icon=":material/hourglass_disabled:"
                    )

        st.sidebar.markdown("""
                            ### Clear Pandabot History

                            :panda_face:
                            Pandabot is a chat assistant
                            designed to work with tabular data
                            (or data frames).
                            If I'm not generating anything useful,
                            it can be helpul to start over.
                            Click the button below.
                            This will remove all previous messages and
                            start a new conversation.
                            """)

        if st.sidebar.button("Clear chat history"):
            if "pandasai" not in st.session_state[user_session_id]:
                st.session_state[user_session_id]["pandasai"] = []
            _utils.llms.clear_pandasai(user_session_id)
            st.rerun()

        st.sidebar.markdown("---")

        with st.sidebar.expander(
            "Pandabot Tips",
            icon=":material/lightbulb_2:",
            expanded=False
        ):
            st.markdown(_utils.content.message_plotbot_tips)

        st.sidebar.markdown("---")

        with st.sidebar.expander(
            "Current LLM Parameters",
            icon=":material/build:",
            expanded=False
        ):
            st.markdown(_utils.llms.print_settings(LLM_PARAMS))

        st.sidebar.markdown("---")

    else:

        st.markdown(_utils.content.message_plotbot)

        st.sidebar.markdown(_utils.content.message_generate_table)
        _utils.handlers.sidebar_action_button(
            button_label="Load Tables",
            button_icon=":material/manufacturing:",
            preconditions=[
                session.get('has_target')[0],
            ],
            action=lambda: _utils.handlers.generate_tags_table(
                user_session_id
            ),
            spinner_message="Loading tables..."
        )

        if st.session_state[user_session_id].get("tags_warning"):
            msg, icon = st.session_state[user_session_id]["tags_warning"]
            st.warning(msg, icon=icon)

        st.sidebar.markdown("---")


if __name__ == "__main__":
    main()
