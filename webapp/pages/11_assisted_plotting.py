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
import streamlit as st

from webapp.utilities.session import (  # noqa: E402
    get_or_init_user_session
)
from webapp.utilities.configuration import (  # noqa: E402
    get_ai_configuration
)
from webapp.utilities.ai import (   # noqa: E402
    clear_plotbot,
    previous_code_chunk,
    plotbot_user_query,
    setup_ai_session_state,
    get_api_key,
    render_api_key_input,
    render_data_selection_interface,
    render_data_preview_controls
)
from webapp.utilities.analysis import (   # noqa: E402
    generate_tags_table
)
from webapp.utilities.ui import (  # noqa: E402
    sidebar_help_link,
    render_table_generation_interface
)
from webapp.utilities.state import (  # noqa: E402
    load_widget_state,
    persist
)
from webapp.config.session_keys import (  # noqa: E402
    SessionKeys,
    WarningKeys
)
from webapp.menu import (   # noqa: E402
    menu,
    require_login
)

TITLE = "AI-Assisted Plotting"
ICON = ":material/smart_toy:"

st.set_page_config(
    page_title=TITLE, page_icon=ICON,
    layout="wide"
)

# Get AI configuration
_options, DESKTOP, CACHE, LLM_MODEL, LLM_PARAMS, QUOTA = get_ai_configuration()


def render_plotting_library_selection(user_session_id: str) -> str:
    """Render the plotting library selection interface."""
    st.markdown(
        body="### Plotting Library",
        help=(
            "To create plots, I can use the following libraries:\n\n"
            "* [Plotly express](https://plotly.com/python/plotly-express/)\n"
            "* [Matplotlib](https://matplotlib.org/)\n"
            "* [Seaborn](https://seaborn.pydata.org/)\n\n"
            "Each library has its own aesthetics and features. "
            "If you're unfamiliar with them, you should check out their "
            "documentation, as well as examples of their use."
        )
    )

    plot_lib = st.radio(
        "Select the plotting library:",
        ("plotly.express", "matplotlib", "seaborn"),
        key=persist("plot_radio", user_session_id, "11_assisted_plotting"),
        on_change=clear_plotbot,
        args=(user_session_id, False,),
        horizontal=True
    )

    return plot_lib


def render_plotbot_chat_interface(
    user_session_id: str,
    api_key: str,
    df,
    selected_query: str,
    plot_lib: str
) -> None:
    """Render the chat interface for Plotbot."""
    # Display chat history
    for message in st.session_state[user_session_id]["plotbot"]:
        with st.chat_message(message["role"]):
            if message["type"] == "string":
                st.markdown(message["value"])
            elif message["type"] == "code":
                st.code(message["value"], language="python")
            elif message["type"] == "error":
                st.error(message["value"], icon=":material/error:")
            elif message["type"] == "plot":
                # Handle different plot types
                if plot_lib in ["matplotlib", "seaborn"]:
                    st.image(message['value'])
                    # Add download link
                    img_bytes = message['value'].getvalue()
                    b64 = base64.b64encode(img_bytes).decode()
                    href = (f'<a href="data:image/png;base64,{b64}" '
                            'download="plot.png">Download PNG</a>')
                    st.markdown(href, unsafe_allow_html=True)
                elif plot_lib == "plotly.express":
                    fig = message['value']
                    fig.update_xaxes(automargin=True)
                    fig.update_yaxes(automargin=True)
                    img_bytes = fig.to_image(format="png", scale=2)
                    st.image(img_bytes)
                    # Add download link
                    b64 = base64.b64encode(img_bytes).decode()
                    href = (f'<a href="data:image/png;base64,{b64}" '
                            'download="plot.png">Download PNG</a>')
                    st.markdown(href, unsafe_allow_html=True)

    # Get last code chunk
    last_code = previous_code_chunk(st.session_state[user_session_id]["plotbot"])

    # Chat input
    if last_code is None or len(last_code) == 0:
        input_prompt = st.chat_input(
            "Please describe what kind of plot you'd like to create."
        )

        if input_prompt:
            with st.spinner(":sparkles: Generating response..."):
                st.session_state[user_session_id]["plotbot"].append(
                    {"role": "user", "type": "string", "value": input_prompt}
                )
                # Increment user prompt count
                prompt_count_key = "plotbot_user_prompt_count"
                if prompt_count_key not in st.session_state[user_session_id]:
                    st.session_state[user_session_id][prompt_count_key] = 1
                else:
                    st.session_state[user_session_id][prompt_count_key] += 1

                # Generate response
                plotbot_user_query(
                    session_id=user_session_id,
                    df=df.to_pandas() if hasattr(df, 'to_pandas') else df,
                    plot_lib=plot_lib,
                    user_input=input_prompt,
                    api_key=api_key,
                    llm_params=LLM_PARAMS,
                    prompt_position=st.session_state[user_session_id][prompt_count_key],
                    cache_mode=CACHE
                )
                st.rerun()
    else:
        # Show refinement input
        input_refine = st.chat_input("How would you like me to refine this plot?")

        if input_refine:
            with st.spinner(":sparkles: Refining plot..."):
                st.session_state[user_session_id]["plotbot"].append(
                    {"role": "user", "type": "string", "value": input_refine}
                )
                st.session_state[user_session_id]["plotbot_user_prompt_count"] += 1

                # Generate refined response
                plotbot_user_query(
                    session_id=user_session_id,
                    df=df.to_pandas() if hasattr(df, 'to_pandas') else df,
                    plot_lib=plot_lib,
                    user_input=input_refine,
                    api_key=api_key,
                    llm_params=LLM_PARAMS,
                    prompt_position=st.session_state[user_session_id][
                        "plotbot_user_prompt_count"
                    ],
                    cache_mode=CACHE
                )
                st.rerun()


def render_plotbot_interface(user_session_id: str, session: dict) -> None:
    """Render the main Plotbot interface with data selection and plotting."""
    try:
        # Initialize session state
        setup_ai_session_state(user_session_id, "plotbot")

        # Get API key first
        api_key = get_api_key(user_session_id, DESKTOP, CACHE, QUOTA)

        # Introduction
        st.markdown(
            body=(
                ":robot_face: Plotbot is an **interactive** chat assistant "
                "designed to help you create and refine plots from your data.\n\n"
                ":material/priority: I remember your previous messages "
                "and use them to generate new responses.\n\n"
                ":material/priority: I am not a general-purpose chatbot, "
                "so I can not answer questions that are not related to plotting."
            )
        )

        # Only show data interfaces if user has valid API key
        if api_key:
            # Add clear chat button to sidebar
            st.sidebar.markdown(
                body="### Chat Controls",
                help=(
                    "You can clear the chat history to start a new conversation. "
                    "This will remove all previous messages and plots."
                ))
            if st.sidebar.button(
                "Clear Chat History",
                icon=":material/delete:"
            ):
                clear_plotbot(user_session_id)
                st.rerun()

            # Get metadata if available
            metadata_target = None
            if session.get(SessionKeys.HAS_TARGET, [False])[0]:
                metadata_target = (
                    st.session_state[user_session_id]['metadata_target'].to_dict()
                )

            # Load widget state
            load_widget_state(user_session_id)

            # Data selection interface
            selected_corpus, selected_query, df = render_data_selection_interface(
                user_session_id, session, "plotbot", "11_assisted_plotting",
                clear_plotbot, metadata_target
            )

            # Data preview with controls
            if df is not None:
                df = render_data_preview_controls(
                    df, selected_query, "11_assisted_plotting", user_session_id
                )

            # Plotting library selection
            plot_lib = render_plotting_library_selection(user_session_id)

            # Chat interface
            render_plotbot_chat_interface(
                user_session_id, api_key, df, selected_query, plot_lib
            )
        else:
            # Show API key input if no valid key available
            render_api_key_input(user_session_id)

    except Exception as e:
        st.error(f"Error loading Plotbot interface: {str(e)}", icon=":material/error:")


def main():
    """Main function to run the Streamlit app for AI-assisted plotting."""
    # Set login requirements for navigation
    require_login()
    menu()
    st.markdown(
        body=f"## {TITLE}",
        help=(
            "To use Plotbot, you need to load tables from the sidebar, "
            "then select one from the interface. "
            "Once you have selected a table, you can enter your prompt "
            "in the chat input box. "
            "Plotbot will then generate a response based on the table you selected.\n\n"
            "If you are using the online version, you can use the API key "
            "provided by CMU, though there is a daily quota limit. "
            "If you're using the desktop version or you reach your quota, "
            "you can enter your own OpenAI API key to use Plotbot "
            "without any quota limits."
        )
    )

    # Get or initialize user session
    user_session_id, session = get_or_init_user_session()

    # Add help link
    sidebar_help_link("assisted-plotting.html")

    # Check if tags table is available
    if session.get(SessionKeys.TAGS_TABLE, [False])[0]:
        render_plotbot_interface(user_session_id, session)
    else:
        # Show generation interface for tags table
        render_table_generation_interface(
            user_session_id=user_session_id,
            session=session,
            table_type="tags table",
            button_label="Tags Table",
            generation_func=generate_tags_table,
            session_key=SessionKeys.TAGS_TABLE,
            warning_key=WarningKeys.TAGS
        )

    st.sidebar.markdown("---")


if __name__ == "__main__":
    main()
