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

import altair as alt
import pandas as pd
import polars as pl
import streamlit as st
import streamlit.components.v1 as components

# Ensure project root is in sys.path for both desktop and online
project_root = pathlib.Path(__file__).parent.parents[1].resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import webapp.utilities as _utils   # noqa: E402
from webapp.menu import menu, require_login   # noqa: E402

HEX_HIGHLIGHTS = ['#5fb7ca', '#e35be5', '#ffc701', '#fe5b05', '#cb7d60']

TITLE = "Single Documents"
ICON = ":material/find_in_page:"

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

    if session.get('doc')[0] is True:

        _utils.handlers.load_widget_state(
            pathlib.Path(__file__).stem,
            user_session_id
            )
        metadata_target = _utils.handlers.load_metadata(
            'target',
            user_session_id
            )

        st.sidebar.markdown("### Tagset")

        st.sidebar.markdown("""Use the menus to select
                            up to **5 tags** you would like to highlight.
                            """)

        with st.sidebar.expander("About general tags"):
            st.markdown(_utils.content.message_general_tags)

        tag_radio = st.sidebar.radio(
            "Select tags to display:",
            ("Parts-of-Speech", "DocuScope"),
            key=_utils.handlers.persist(
                "sd_radio", pathlib.Path(__file__).stem,
                user_session_id
                ),
            horizontal=True)

        if tag_radio == 'Parts-of-Speech':
            tag_type = st.sidebar.radio(
                "Select from general or specific tags",
                ("General", "Specific"),
                horizontal=True
                )
            if tag_type == 'General':
                tag_loc = st.session_state[
                    user_session_id
                    ]["target"]["doc_simple"]
                html_simple = ''.join(tag_loc.get_column("Text").to_list())
                doc_key = tag_loc.get_column("doc_id").unique().to_list()

                tag_list = st.sidebar.multiselect(
                    'Select tags to highlight',
                    [
                        'Adjective',
                        'Adverb',
                        'Conjunction',
                        'NounCommon',
                        'NounOther',
                        'Preposition',
                        'Pronoun',
                        'VerbBe',
                        'VerbLex',
                        'VerbOther'
                    ],
                    on_change=_utils.handlers.update_tags(
                        html_simple,
                        user_session_id
                        ),
                    key=f"tags_{user_session_id}"
                    )
                tag_colors = HEX_HIGHLIGHTS[:len(tag_list)]
                tag_html = zip(tag_colors, tag_list)
                tag_html = list(map('">'.join, tag_html))
                tag_html = ['<span style="background-color: ' +
                            item + '</span>' for item in tag_html]
                tag_html = '; '.join(tag_html)
                df = (tag_loc
                      .filter(pl.col("Tag") != "Other")
                      .group_by("Tag").len("AF")
                      .with_columns(
                          pl.col("AF")
                          .truediv(pl.sum("AF")).mul(100).alias("RF")
                          )
                      .sort(["AF", "Tag"], descending=[True, False])
                      ).to_pandas()
            else:
                tag_loc = st.session_state[
                    user_session_id
                    ]["target"]["doc_pos"]
                html_pos = ''.join(tag_loc.get_column("Text").to_list())
                doc_key = tag_loc.get_column("doc_id").unique().to_list()

                tag_list = st.sidebar.multiselect(
                    'Select tags to highlight',
                    metadata_target.get('tags_pos')[0]['tags'],
                    on_change=_utils.handlers.update_tags(
                        html_pos, user_session_id),
                    key=f"tags_{user_session_id}")
                tag_colors = HEX_HIGHLIGHTS[:len(tag_list)]
                tag_html = zip(tag_colors, tag_list)
                tag_html = list(map('">'.join, tag_html))
                tag_html = ['<span style="background-color: ' +
                            item + '</span>' for item in tag_html]
                tag_html = '; '.join(tag_html)
                df = (tag_loc
                      .filter(pl.col("Tag") != "Y")
                      .group_by("Tag").len("AF")
                      .with_columns(
                          pl.col("AF")
                          .truediv(pl.sum("AF")).mul(100).alias("RF")
                          )
                      .sort(["AF", "Tag"], descending=[True, False])
                      ).to_pandas()
        else:
            tag_loc = st.session_state[user_session_id]["target"]["doc_ds"]
            html_ds = ''.join(tag_loc.get_column("Text").to_list())
            doc_key = tag_loc.get_column("doc_id").unique().to_list()

            tag_list = st.sidebar.multiselect(
                'Select tags to highlight',
                metadata_target.get('tags_ds')[0]['tags'],
                on_change=_utils.handlers.update_tags(
                    html_ds, user_session_id),
                key=f"tags_{user_session_id}")
            tag_colors = HEX_HIGHLIGHTS[:len(tag_list)]
            tag_html = zip(tag_colors, tag_list)
            tag_html = list(map('">'.join, tag_html))
            tag_html = ['<span style="background-color: ' +
                        item + '</span>' for item in tag_html]
            tag_html = '; '.join(tag_html)
            df = (tag_loc
                  .filter(pl.col("Tag") != "Untagged")
                  .group_by("Tag").len("AF")
                  .with_columns(
                      pl.col("AF")
                      .truediv(pl.sum("AF")).mul(100).alias("RF")
                      )
                  .sort(["AF", "Tag"], descending=[True, False])
                  ).to_pandas()

        if len(tag_list) == 5:
            st.sidebar.markdown(""":warning: You can hightlight
                                a maximum of 5 tags.
                                """)

        st.sidebar.markdown("---")
        st.sidebar.markdown("### Plot tag locations")
        with st.sidebar.expander("Plot explanation"):
            st.write("""The plot(s) shows lines segment
                    where tags occur in what might be called
                    'normalized text time.'
                    For example, if you had a text 100 tokens long
                    and a tag occurred at the 10th, 25th, and 60th token,
                    the plot would show lines at
                    10%, 25%, and 60% along the x-axis.
                    """)

        st.markdown(f"""
                    ###  {doc_key[0]}
                    """)

        if st.sidebar.button("Tag Density Plot"):
            if len(tag_list) > 5:
                st.write(""":no_entry_sign: You can only plot
                         a maximum of 5 tags.
                         """)
            elif len(tag_list) == 0:
                st.write('There are no tags to plot.')
            else:
                plot_colors = tag_html.replace(
                    '<span style="background-color: ', ''
                    )
                plot_colors = plot_colors.replace(
                    '</span>', ''
                    )
                plot_colors = plot_colors.replace(
                    '">', '; '
                    )
                plot_colors = plot_colors.split("; ")
                plot_colors = list(zip(plot_colors[1::2], plot_colors[::2]))
                plot_colors = pd.DataFrame(plot_colors,
                                           columns=['Tag', 'Color'])
                plot_colors = plot_colors.sort_values(by=['Tag'])
                plot_colors = plot_colors['Color'].unique()

                df_plot = tag_loc.to_pandas()
                df_plot['X'] = (df_plot.index + 1)/(len(df_plot.index))
                df_plot = df_plot[df_plot['Tag'].isin(tag_list)]

                base = alt.Chart(
                    df_plot,
                    height={"step": 45}).mark_tick(size=35).encode(
                    x=alt.X(
                        'X:Q',
                        axis=alt.Axis(
                            values=[0, .25, .5, .75, 1],
                            format='%'
                            ), title=None
                        ),
                    y=alt.Y(
                        'Tag:O',
                        title=None,
                        sort=tag_list
                        )
                    )

                lex_density = base.encode(
                    color=alt.Color(
                        'Tag',
                        scale=alt.Scale(range=plot_colors),
                        legend=None),
                )

                st.altair_chart(lex_density, use_container_width=True)

        st.markdown(f"""
                    ##### Tags:  {tag_html}
                    """,
                    unsafe_allow_html=True
                    )

        if 'html_str' not in st.session_state[user_session_id]:
            st.session_state[user_session_id]['html_str'] = ''

        components.html(
            st.session_state[user_session_id]['html_str'],
            height=500,
            scrolling=True
            )

        st.dataframe(df, hide_index=True)

        st.sidebar.markdown("---")
        download_doc = st.sidebar.toggle("Download to Word?")
        if download_doc is True:
            st.sidebar.markdown(_utils.content.message_download)
            with st.sidebar:
                download_file = _utils.formatters.convert_to_word(
                    st.session_state[user_session_id]['html_str'],
                    tag_html,
                    doc_key,
                    df
                    )

                st.download_button(
                    label="Download to Word",
                    data=download_file,
                    file_name="document_tags.docx",
                    mime="docx",
                        )

        st.sidebar.markdown("---")

        st.sidebar.markdown("### Reset document")
        st.sidebar.markdown("""
                            Click the button to explore a new document.
                            """)
        if st.sidebar.button("Select a new document"):
            _TAGS = f"tags_{user_session_id}"

            if "doc_pos" not in st.session_state[user_session_id]["target"]:
                st.session_state[user_session_id]["target"]["doc_pos"] = {}
            st.session_state[user_session_id]["target"]["doc_pos"] = {}

            if "doc_simple" not in st.session_state[user_session_id]["target"]:
                st.session_state[user_session_id]["target"]["doc_simple"] = {}
            st.session_state[user_session_id]["target"]["doc_simple"] = {}

            if "doc_ds" not in st.session_state[user_session_id]["target"]:
                st.session_state[user_session_id]["target"]["doc_ds"] = {}
            st.session_state[user_session_id]["target"]["doc_ds"] = {}

            _utils.handlers.update_session('doc', False, user_session_id)

            if _TAGS in st.session_state:
                del st.session_state[_TAGS]
            st.rerun()

        st.sidebar.markdown("---")

    else:

        st.markdown(_utils.content.message_single_document)

        try:
            metadata_target = _utils.handlers.load_metadata(
                'target',
                user_session_id
                )
        except Exception:
            pass

        st.sidebar.markdown("### Choose document")
        st.sidebar.write("""Use the menus to select
            the tags you would like to highlight.
            """)

        if session.get('has_target')[0] is True:
            doc_key = st.sidebar.selectbox(
                "Select document to view:",
                (sorted(metadata_target.get('docids')[0]['ids']))
                )
        else:
            doc_key = st.sidebar.selectbox(
                "Select document to view:",
                (['No documents to view'])
                )

        _utils.handlers.sidebar_action_button(
            button_label="Process Document",
            button_icon=":material/manufacturing:",
            preconditions=[
                session.get('has_target')[0],
            ],
            action=lambda: _utils.handlers.generate_document_html(
                user_session_id, doc_key
            ),
            spinner_message="Processing document..."
        )

        if st.session_state[user_session_id].get("doc_warning"):
            msg, icon = st.session_state[user_session_id]["doc_warning"]
            st.warning(msg, icon=icon)

        st.sidebar.markdown("---")


if __name__ == "__main__":
    main()
