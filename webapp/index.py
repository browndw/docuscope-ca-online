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
import json
import os
import pathlib
import streamlit as st
import sys

# Ensure project root is in sys.path for both desktop and online
project_root = pathlib.Path(__file__).parent.parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from webapp.menu import menu   # noqa: E402
from webapp.utilities.handlers import import_options_general, get_version_from_pyproject   # noqa: E402, E501

OPTIONS = str(project_root.joinpath("webapp/options.toml"))
TITLE_LOGO = str(project_root.joinpath("webapp/_static/docuscope-logo.png"))
USER_GUIDE_URL = "https://browndw.github.io/docuscope-docs/"
PL_LOGO = str(project_root.joinpath("webapp/_static/porpoise_badge.svg"))
UG_LOGO = str(project_root.joinpath("webapp/_static/user_guide.svg"))
SPACY_META = project_root.joinpath("webapp/_models/en_docusco_spacy/meta.json")

__version__ = get_version_from_pyproject()
_options = import_options_general(OPTIONS)
DESKTOP = _options['global']['desktop_mode']


st.set_page_config(
    page_title="DocuScope Corpus Analysis", page_icon=":material/quick_reference_all:",
    layout="wide"
    )


@st.cache_data
def get_base64_of_bin_file(bin_file):
    with open(bin_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


@st.cache_data
def get_img_with_header(local_img_path):
    img_format = os.path.splitext(local_img_path)[-1].replace(".", "")
    bin_str = get_base64_of_bin_file(local_img_path)
    html_code = f'''
    <div class="image-txt-container" style="background-color: #FFE380; border-radius: 5px">
      <img src="data:image/{img_format};base64,{bin_str}" height="125">
      <h2 style="color: #DE350B; text-align:center">
        DocuScope
      </h2>
      <h2 style="color: #42526E; text-align:center">
        Corpus Analysis & Concordancer
      </h2>

    </div>
      '''  # noqa: E501
    return html_code


@st.cache_data
def get_file_contents(filepath, encoding='utf-8'):
    with open(filepath, encoding=encoding, errors='ignore') as f:
        return f.read()


@st.cache_data
def get_json_contents(filepath):
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        return json.load(f)


@st.cache_data
def get_base64_encoded(content: str) -> str:
    return base64.b64encode(content.encode('utf-8')).decode('utf-8')


@st.cache_data
def version_info(ds_version: str,
                 model_name: str,
                 model_version: str) -> str:
    version_info = f"""<p></p><span style="color:gray">
    DocuScope CA version: {ds_version};
    spaCy model
    <a href="https://huggingface.co/browndw/en_docusco_spacy/" target="_blank">
    {model_name}</a> version: {model_version}
    </span>
    """
    return version_info


def main() -> None:
    """
    index.py: Streamlit entry point for DocuScope CAC.

    Returns
    -------
    None
    """
    menu()

    user_session = st.runtime.scriptrunner_utils.script_run_context.get_script_run_ctx()  # noqa: E501
    user_session_id = user_session.session_id

    if user_session_id not in st.session_state:
        st.session_state[user_session_id] = {}

    st.markdown("""
        <style>
        .image-txt-container {
        display: flex;
        align-items: center;
        flex-direction: row;
        }
        .reportview-container {
        margin-top: -2em;
        }
        </style>
        """, unsafe_allow_html=True)

    try:
        pl_logo_text = get_file_contents(PL_LOGO)
        b64 = get_base64_encoded(pl_logo_text)
        pl_html = r"""
            <a href="https://github.com/browndw/"><img src="data:image/svg +
            xml;base64,%s"/></a>  © 2025 David Brown, Suguru Ishizaki, David Kaufer
                """ % b64  # noqa: E501
        st.markdown(pl_html, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Could not load porpoise badge logo: {e}")

    # Title logo (use your existing cached get_img_with_header if available)
    try:
        st.markdown(
            get_img_with_header(TITLE_LOGO),
            unsafe_allow_html=True
        )
    except Exception as e:
        st.error(f"Could not load title logo: {e}")

    # Model info
    try:
        json_data = get_json_contents(SPACY_META)
        model_name = json_data["name"]
        model_version = json_data["version"]
        st.markdown(
            version_info(
                __version__,
                model_name,
                model_version),
            unsafe_allow_html=True
        )
    except Exception as e:
        st.error(f"Could not load model information: {e}")

    st.markdown("---")

    st.markdown(
        """
        Welcome to **DocuScope Corpus Analaysis & Concordancer**.
        This suite of tools is designed to help those new to corpus analysis and
        NLP explore data, data visualization,
        and the computational analysis of text.
        It is also designed to allow users to easily toggle between
        **rhetorical tags** and more conventional **part-of-speech tags**.
        The application in available online and as a desktop application.
        The online version resitricts the amount of data
        that you can process at one time.
        Both versions are open source and can be accessed from
        the GitHub repository linked at the top of this page.
        """)

    st.sidebar.link_button(
        label="User Guide",
        url="https://browndw.github.io/docuscope-docs/",
        icon=":material/help:"
    )

    st.sidebar.markdown("### Learn more...")

    if DESKTOP or (hasattr(st, "user") and getattr(st.user, "is_logged_in", False)):
        st.markdown(
            """
            All apps in the tools can be accessed by using the
            **:material/explore: Navigation** menu in the sidebar on the left.
            But before you can use any of the other apps, you will need
            to load and process a corpus using:
            """)
        st.page_link(
            page="pages/1_load_corpus.py",
            label="Manage Corpus Data",
            icon=":material/database:"
            )
    else:
        st.markdown(
            """
            Please log in to access the tool.
            """)


if __name__ == "__main__":
    main()
