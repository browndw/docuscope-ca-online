# Copyright (C) 2024 David West Brown

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
import functools
import json
import pathlib
import streamlit as st
import sys
import textwrap
import warnings

import utilities
import apps as _stable_apps
import categories as _categories
from utilities import handlers_database as _handlers
from utilities import content as _content
from utilities import messages as _messages
from _version import __version__


HERE = pathlib.Path(__file__).parent.resolve()
FAVICON = str(HERE.joinpath("_static/docuscope-favicon.ico"))
TITLE_LOGO = str(HERE.joinpath("_static/docuscope-logo.png"))
PL_LOGO = str(HERE.joinpath("_static/porpoise_badge.svg"))
UG_LOGO = str(HERE.joinpath("_static/user_guide.svg"))
STYLE = str(HERE.joinpath("css/style.css"))
SPACY_META = HERE.joinpath("models/en_docusco_spacy/meta.json")

st.set_page_config(page_title="DocuScope CAC", page_icon=FAVICON, layout="wide")
_content.local_css(STYLE)

user_session = st.runtime.scriptrunner.script_run_context.get_script_run_ctx()
user_session_id = user_session.session_id

if user_session_id not in st.session_state:
    st.session_state[user_session_id] = {}

def index(application_options):
    if not sys.warnoptions:
        warnings.simplefilter("ignore")

    with open(PL_LOGO, encoding='utf-8', errors='ignore') as f:
        pl_logo_text = f.read()
    b64 = base64.b64encode(pl_logo_text.encode('utf-8')).decode("utf-8")
    pl_html = r'<a href="https://github.com/browndw/"><img src="data:image/svg+xml;base64,%s"/></a>  © 2024 David Brown, Suguru Ishizaki, David Kaufer' % b64
    
    st.markdown(pl_html, unsafe_allow_html=True)

    st.markdown(_content.get_img_with_header(TITLE_LOGO), unsafe_allow_html=True)
    
    with open(SPACY_META) as json_file:
        json_data = json.load(json_file)
    
    model_name = json_data["name"]
    model_version = json_data["version"]

    st.markdown(_messages.message_version_info(__version__, model_name, model_version), unsafe_allow_html=True)
         
    st.markdown("---")
    #with open(UG_LOGO) as f:
        #ug_logo_text = f.read()
    #b64 = base64.b64encode(ug_logo_text.encode('utf-8')).decode("utf-8")
    #ug_html = r'<a href="https://docuscope.github.io/docuscope-cao.html"><img src="data:image/svg+xml;base64,%s"/></a>' % b64
    
    #st.markdown(ug_html, unsafe_allow_html=True)
    
    st.markdown(_content.message_landing)

    _, central_header_column, _ = st.columns((1, 2, 1))

    num_columns = len(_categories.APPLICATION_CATEGORIES_BY_COLUMN.keys())
    columns = st.columns(num_columns)

    for (
        column_index,
        categories,
    ) in _categories.APPLICATION_CATEGORIES_BY_COLUMN.items():
        column = columns[column_index]

        for category in categories:
            applications_in_this_category = [
                item
                for item in application_options.items()
                if item[1].CATEGORY == category
            ]

            column.write(
                f"""
                ## {category}
                """
            )

            if not applications_in_this_category:
                column.write("> *No applications are currently in this category.*")
                continue

            applications_in_this_category = sorted(
                applications_in_this_category, key=_content._application_sorting_key
            )

            for app_key, application in applications_in_this_category:
                if column.button(application.TITLE):
                    _content.swap_app(app_key)
    
def main():
    if not sys.warnoptions:
        warnings.simplefilter("ignore")

    session_state = utilities.session_state(app=_content.get_url_app())

    stable_apps = _content._get_apps_from_module(_stable_apps)
    application_options = {**stable_apps}

    applications_all = [
        item
        for item in application_options.items()
    ]
    
    if (
        session_state.app != "index"
        and not session_state.app in application_options.keys()
    ):
        _content.swap_app("index")

    if session_state.app != "index":        
        st.markdown(_content.nav_header, unsafe_allow_html=True)
        _, central_header_column, _ = st.columns((1, 2, 1))
    
        num_columns = len(_categories.APPLICATION_CATEGORIES_BY_COLUMN.keys())
        columns = st.columns(num_columns)

        for (
            column_index,
            categories,
        ) in _categories.APPLICATION_CATEGORIES_BY_COLUMN.items():
            column = columns[column_index]
    
            for category in categories:
                applications_in_this_category = [
                    item
                    for item in application_options.items()
                    if item[1].CATEGORY == category
                ]
            
                applications_in_this_category = sorted(
                    (f for f in applications_in_this_category if 1 < _content._application_sorting_key(f) < 11),
                    key=_content._application_sorting_key
                )
    
                for app_key, application in applications_in_this_category:
                    if column.button(application.TITLE):
                        _content.swap_app(app_key)   
        
        if st.button("Manage Corpora"):
            _content.swap_app("load-corpus")
        st.markdown("""---""")
        st.title(application_options[session_state.app].TITLE)
        
        docstring = application_options[session_state.app].main.__doc__
        if docstring is not None:
            docstring = textwrap.dedent(f"    {docstring}")
            st.write(docstring)

        if st.sidebar.button("Return to All Tools"):
            _content.swap_app("index")
        st.sidebar.write("---")
            
    if session_state.app == "index":
        application_function = functools.partial(
            index, application_options=application_options
        )
    else:
        application_function = application_options[session_state.app].main

    application_function()

if __name__ == "__main__":
    main()


