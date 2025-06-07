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

# Ensure project root is in sys.path for both desktop and online
project_root = pathlib.Path(__file__).parent.parents[1].resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import webapp.utilities as _utils   # noqa: E402
from webapp.menu import menu, require_login   # noqa: E402

TITLE = "AI-Assisted Plotting"
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
        url="https://browndw.github.io/docuscope-docs/guide/assisted-plotting.html",
        icon=":material/help:"
        )

    if session.get('has_target')[0] is True:
        metadata_target = st.session_state[
            user_session_id
            ]['metadata_target'].to_dict()

    # Initialize chat history
    if "plotbot" not in st.session_state[user_session_id]:
        st.session_state[user_session_id]["plotbot"] = []

    if "plotbot_user_prompt_count" not in st.session_state[user_session_id]:
        st.session_state[user_session_id]["plotbot_user_prompt_count"] = 0

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

        st.markdown(_utils.content.message_plotbot_home)

        _utils.handlers.load_widget_state(
            pathlib.Path(__file__).stem,
            user_session_id
            )

        st.markdown("### Data Selection")

        with st.expander("About data",
                         icon=":material/table:",
                         expanded=False):

            st.markdown(_utils.content.message_plotbot_data)

        plotbot_corpus = st.radio(
            "Select corpus:",
            ("Target", "Reference", "Keywords", "Grouped"),
            key=_utils.handlers.persist(
                "plotbot_corpus", pathlib.Path(__file__).stem,
                user_session_id),
            on_change=_utils.llms.clear_plotbot,
            args=(user_session_id,),
            index=0,
            horizontal=True
                )

        if session.get('has_meta')[0] is True:
            groups = metadata_target.get('doccats')[0]['cats']
        elif session.get('has_meta')[0] is False:
            groups = []

        plotbot_query = st.selectbox(
                "Select data to plot:",
                (_utils.llms.tables_to_list(session_id=user_session_id,
                                            corpus=plotbot_corpus,
                                            categories=groups)),
                key=_utils.handlers.persist(
                    "plotbot_query", pathlib.Path(__file__).stem,
                    user_session_id),
                on_change=_utils.llms.clear_plotbot,
                args=(user_session_id,),
                index=None,
                placeholder="Select data..."
                )

        st.markdown("### Data Preview")
        df = _utils.llms.table_from_list(user_session_id,
                                         plotbot_corpus,
                                         plotbot_query,
                                         categories=groups)

        if df is not None and df.shape[0] > 0:
            if (
                df.get_column("Group", default=None) is not None or
                str("Document-Term Matrix") in plotbot_query
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
                        str("Document-Term Matrix") in plotbot_query and
                        df.get_column("Group", default=None) is None
                    ):
                        df = df.pivot(
                            "Tag",
                            index="doc_id",
                            values="RF"
                            )
                    elif (
                        pivot_table and
                        str("Document-Term Matrix") in plotbot_query and
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
                        str("Document-Term Matrix") in plotbot_query
                    ):
                        df = df.with_columns(
                            (
                                pl.selectors.numeric().mul(100)
                                )
                        )
                    elif (
                        make_percent and
                        str("Tags Table") in plotbot_query
                    ):
                        df = df.with_columns(
                            (
                                pl.selectors.numeric()
                                .exclude(["AF", "Range"]).mul(100)
                                )
                        )

            st.dataframe(df.head(10), use_container_width=True)

        st.markdown("### Plotting Library")

        with st.expander("About libraries",
                         icon=":material/add_chart:",
                         expanded=False):

            st.markdown(_utils.content.message_plotbot_libraries)

        plot_lib = st.radio(
            "Select the plotting library:",
            ("plotly.express", "matplotlib", "seaborn"),
            key=_utils.handlers.persist(
                "plot_radio", pathlib.Path(__file__).stem,
                user_session_id),
            on_change=_utils.llms.clear_plotbot,
            args=(user_session_id, False,),
            horizontal=True
                )

        if st.session_state[user_session_id]["plotbot"]:
            for i, message in enumerate(
                st.session_state[user_session_id]["plotbot"]
            ):
                with st.chat_message(message["role"]):
                    if message['type'] == 'string':
                        st.markdown(
                            message['value'],
                            unsafe_allow_html=True
                            )
                    elif message['type'] == 'error':
                        st.markdown(
                            message['value'],
                            unsafe_allow_html=True
                            )
                    elif message['type'] == 'code':
                        st.code(
                            message['value'],
                            language='python'
                            )
                    # Render plots consitently as static images
                    # rather than interactive HTML.
                    # This ensures that what users download
                    # is consistent with what's rendered in streamlit.
                    elif message['type'] == 'plot':
                        if (
                            plot_lib == "matplotlib" or
                            plot_lib == "seaborn"
                        ):
                            # Save the matplotlib figure to a BytesIO buffer
                            scale = 2  # or any desired scale factor
                            dpi = 100 * scale  # base dpi (e.g., 100) x scale

                            buf = io.BytesIO()
                            message['value'].savefig(
                                buf,
                                format="png",
                                bbox_inches="tight",
                                dpi=dpi
                                )
                            buf.seek(0)
                            img_bytes = buf.getvalue()

                            st.image(img_bytes)

                            # Add download link
                            b64 = base64.b64encode(img_bytes).decode()
                            href = f'<a href="data:image/png;base64,{b64}" download="plot.png">Download PNG</a>'  # noqa: E501
                            st.markdown(href, unsafe_allow_html=True)

                        elif (
                            plot_lib == "plotly.express"
                        ):
                            # Set desired resolution (scale)
                            scale = 2
                            fig = message['value']
                            fig.update_xaxes(automargin=True)
                            fig.update_yaxes(automargin=True)
                            img_bytes = fig.to_image(
                                format="png",
                                scale=scale
                            )
                            st.image(img_bytes)

                            # Add download link
                            b64 = base64.b64encode(img_bytes).decode()
                            href = f'<a href="data:image/png;base64,{b64}" download="plot.png">Download PNG</a>'  # noqa: E501
                            st.markdown(href, unsafe_allow_html=True)

        last_code = _utils.llms.previous_code_chunk(
            st.session_state[user_session_id]["plotbot"]
            )

        if st.session_state[user_session_id]["plotbot"]:
            prompt_position = st.session_state[user_session_id]["plotbot_user_prompt_count"]
        else:
            prompt_position = 1

        if (
            api_key is not None and
            (last_code is None or len(last_code) == 0)
        ):
            input_initial = st.chat_input(
                """Please describe what kind of plot you'd like to create.
                """)

            if input_initial:
                if input_initial:
                    with st.spinner(":sparkles: Generating response..."):
                        st.session_state[user_session_id]["plotbot"].append(
                            {"role": "user", "type": "string", "value": input_initial}
                        )
                        # Increment user prompt count
                        if "plotbot_user_prompt_count" not in st.session_state[user_session_id]:  # noqa: E501
                            st.session_state[user_session_id]["plotbot_user_prompt_count"] = 1  # noqa: E501
                        else:
                            st.session_state[user_session_id]["plotbot_user_prompt_count"] += 1  # noqa: E501

                    if (
                        df is not None and
                        df.height > 0
                    ):
                        _utils.llms.plotbot_user_query(
                            session_id=user_session_id,
                            df=df.to_pandas(),
                            plot_lib=plot_lib,
                            user_input=input_initial,
                            api_key=api_key,
                            llm_params=LLM_PARAMS,
                            prompt_position=prompt_position,
                            cache_mode=CACHE
                            )
                    else:
                        error_message = """
                        :confused: I don't have any data to plot.
                        Please select a table from the drop down list above.
                        """
                        st.session_state[user_session_id]["plotbot"].append(
                            {"role": "assistant",
                             "type": "string",
                             "value": error_message}
                            )

                    st.rerun()

        elif (
            api_key is not None and
            (last_code is not None and len(last_code) > 0)
        ):

            input_update = st.chat_input(
                """Please describe how you'd like to update the plot.
                """)

            if input_update:
                with st.spinner(":sparkles: Generating response..."):
                    st.session_state[user_session_id]["plotbot"].append(
                        {"role": "user", "type": "string", "value": input_update}
                    )
                    # Increment user prompt count
                    if "plotbot_user_prompt_count" not in st.session_state[user_session_id]:
                        st.session_state[user_session_id]["plotbot_user_prompt_count"] = 1
                    else:
                        st.session_state[user_session_id]["plotbot_user_prompt_count"] += 1

                    _utils.llms.plotbot_user_query(
                        session_id=user_session_id,
                        df=df.to_pandas(),
                        plot_lib=plot_lib,
                        user_input=input_update,
                        api_key=api_key,
                        llm_params=LLM_PARAMS,
                        prompt_position=prompt_position,
                        cache_mode=CACHE,
                        code_chunk=last_code
                        )

                st.rerun()
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
                            ### Clear Plotbot History

                            :robot_face:
                            Plotbot is an **interative** chat assistant.
                            I remember your previous messages and use them
                            to generate new responses.
                            If I'm not generating anything useful,
                            it can be helpul to start over.
                            Click the button below.
                            This will remove all previous messages and
                            start a new conversation.
                            """)

        if st.sidebar.button("Clear chat history"):
            if "plotbot" not in st.session_state[user_session_id]:
                st.session_state[user_session_id]["plotbot"] = []
            _utils.llms.clear_plotbot(user_session_id)

            if "plotbot_corpus" in st.session_state:
                try:
                    del st.session_state.plotbot_corpus
                except KeyError:
                    pass
            if "plotbot_query" in st.session_state:
                try:
                    del st.session_state.plotbot_query
                except KeyError:
                    pass
            st.session_state.plotbot_corpus = 'Target'
            st.session_state.plotbot_query = None

            st.rerun()

        st.sidebar.markdown("---")

        with st.sidebar.expander(
            "Plotbot Tips",
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
