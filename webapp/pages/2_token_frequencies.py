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

import docuscospacy as ds
import streamlit as st

# Ensure project root is in sys.path for both desktop and online
project_root = pathlib.Path(__file__).parent.parents[1].resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from webapp.utilities.handlers import (  # noqa: E402
    generate_frequency_table,
    get_or_init_user_session,
    load_metadata
    )
from webapp.utilities.ui import (   # noqa: E402
    load_widget_state,
    persist,
    render_dataframe,
    sidebar_action_button,
    sidebar_help_link,
    tag_filter_multiselect,
    tagset_selection,
    target_info,
    toggle_download,
    )
from webapp.utilities.formatters import (  # noqa: E402
    convert_to_excel
    )
from webapp.menu import (   # noqa: E402
    menu,
    require_login
    )

TITLE = "Token Frequencies"
ICON = ":material/table_view:"

st.set_page_config(
    page_title=TITLE, page_icon=ICON,
    layout="wide"
    )


def main() -> None:
    """
    Main function to run the Streamlit app for token frequencies.
    It initializes the user session, loads the necessary data,
    and provides the UI for generating and displaying token frequency tables.
    """
    # Set login requirements for navigaton
    require_login()
    menu()
    st.markdown(
        body=f"## {TITLE}",
        help=(
            "This app allows you to generate and view token frequency tables "
            "for the loaded target corpus. You can filter by tags and download "
            "the table in Excel format."
            )
        )
    # Get or initialize user session
    user_session_id, session = get_or_init_user_session()

    sidebar_help_link("token-frequencies.html")

    # Check if frequency table is already generated
    if session.get('freq_table', [False])[0] is True:
        load_widget_state(
            pathlib.Path(__file__).stem,
            user_session_id
        )
        metadata_target = load_metadata(
            'target',
            user_session_id
        )
        # Load the tags table for the target
        df, tag_options, tag_radio, tag_type = tagset_selection(
            user_session_id=user_session_id,
            session_state=st.session_state,
            persist_func=persist,
            page_stem=pathlib.Path(__file__).stem,
            tagset_keys={
                "Parts-of-Speech": {"General": "ft_pos", "Specific": "ft_pos"},
                "DocuScope": "ft_ds"
            },
            simplify_funcs={
                "Parts-of-Speech": {"General": ds.freq_simplify, "Specific": None}
            },
            tag_filters={
                # Add filters here to exclude tags for specific tagsets/subtypes
                # Example: "DocuScope": lambda df: df.filter(pl.col("Tag") != "Untagged")
            },
            tag_radio_key="ft_radio",
            tag_type_key="ft_type_radio"
        )

        # Display the target information and the token frequencies
        st.info(target_info(metadata_target))

        # Display the tagset selection radio buttons
        df = tag_filter_multiselect(df)
        render_dataframe(df)

        st.sidebar.markdown("---")
        # Toggle download options for the frequency table
        toggle_download(
            label="Excel",
            convert_func=convert_to_excel,
            convert_args=(df.to_pandas(),) if (df is not None and getattr(df, "height", 0) > 0) else (None,),  # noqa: E501
            file_name="token_frequencies.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            location=st.sidebar
        )

        st.sidebar.markdown("---")

    else:
        st.markdown(
            body=(
                ":material/manufacturing: Use the button in the sidebar to **generate a frequency table**.\n\n"  # noqa: E501
                ":material/priority: A **target corpus** must be loaded first.\n\n"
                ":material/priority: After the table has been generated, "
                "you will be able to **toggle between the tagsets**."
                )
        )

        # Display the sidebar header for generating frequency table
        st.sidebar.markdown(
            body=(
                "### Generate table\n\n"
                "Use the button to process a table."
                ),
            help=(
                "Tables are generated based on the loaded target corpus. "
                "You can filter the table by tags after it has been generated. "
                "The table will include token frequencies for the selected tagsets.\n\n"
                "Click on the **Help** button for more information on how to use this app."
                )
            )
        # Action button to generate frequency table
        sidebar_action_button(
            button_label="Frequency Table",
            button_icon=":material/manufacturing:",
            preconditions=[
                session.get('has_target', [False])[0],
            ],
            action=lambda: generate_frequency_table(
                user_session_id
            ),
            spinner_message="Processing frequencies..."
        )
        # Display any frequency warnings
        if st.session_state[user_session_id].get("frequency_warning"):
            msg, icon = st.session_state[user_session_id]["frequency_warning"]
            st.warning(msg, icon=icon)

        st.sidebar.markdown("---")


if __name__ == "__main__":
    main()
