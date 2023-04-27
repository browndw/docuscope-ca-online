# Copyright (C) 2023 David West Brown

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
import os
import pathlib
import re
import sys
import textwrap
import time
import warnings
from importlib.machinery import SourceFileLoader


HERE = pathlib.Path(__file__).parent.resolve()
FAVICON = str(HERE.joinpath("_static/docuscope-favicon.ico"))
TITLE_LOGO = str(HERE.joinpath("_static/docuscope-logo.png"))
PL_LOGO = str(HERE.joinpath("_static/porpoise_badge.svg"))
UG_LOGO = str(HERE.joinpath("_static/user_guide.svg"))
STYLE = str(HERE.joinpath("css/style.css"))
OPTIONS = str(HERE.joinpath("options.toml"))
IMPORTS = str(HERE.joinpath("utilities/handlers_imports.py"))

# import options
_imports = SourceFileLoader("handlers_imports", IMPORTS).load_module()
_options = _imports.import_options_general(OPTIONS)

modules = ['apps', 'categories', 'content', 'handlers', 'states', 'utilities', 'streamlit']
import_params = _imports.import_parameters(_options, modules)

for module in import_params.keys():
	object_name = module
	short_name = import_params[module][0]
	context_module_name = import_params[module][1]
	if not short_name:
		short_name = object_name
	if not context_module_name:
		globals()[short_name] = __import__(object_name)
	else:
		context_module = __import__(context_module_name, fromlist=[object_name])
		globals()[short_name] = getattr(context_module, object_name)

_handlers.generate_temp(_states.STATES.items())

def index(application_options):
    if not sys.warnoptions:
        warnings.simplefilter("ignore")
    
    with open(PL_LOGO) as f:
        pl_logo_text = f.read()
    b64 = base64.b64encode(pl_logo_text.encode('utf-8')).decode("utf-8")
    pl_html = r'<a href="https://github.com/browndw/corpus-tagger/"><img src="data:image/svg+xml;base64,%s"/></a>  © 2023 David Brown, Suguru Ishizaki, David Kaufer' % b64
    
    st.markdown(pl_html, unsafe_allow_html=True)

    st.markdown(_content.get_img_with_header(TITLE_LOGO), unsafe_allow_html=True)
        
    st.markdown("---")
    with open(UG_LOGO) as f:
        ug_logo_text = f.read()
    b64 = base64.b64encode(ug_logo_text.encode('utf-8')).decode("utf-8")
    ug_html = r'<a href="https://docuscope.github.io/docuscope-cao.html"><img src="data:image/svg+xml;base64,%s"/></a>' % b64
    
    st.markdown(ug_html, unsafe_allow_html=True)
    
    st.markdown("""
        
        Welcome to **DocuScope Corpus Analaysis & Concordancer Online**.
        This suite of tools is designed to help those new to corpus analysis and NLP explore data, data visualization, and the computational analysis of text. 
        It is also designed to allow users to easily toggle between **rhetorical tags** and more conventional **part-of-speech tags**.
        Note that the online version is intended to process **small corpora** (< 2 million words). For larger datasets, a desktop version is available. (Check the link to the **User Guide**.)
        Or users with more experience can download (from the GitHub repository linked at the top of this page) and run the **streamlit** app locally using Python.
        To get started:
        
        :point_right: Prepare a corpus of **plain text files**. (The tool does not accept Word files, PDFs, etc.)
        
        :point_right: Use **Manage Corpus Data** to select and process your files.
        
        :point_right: Refer to the **Help** documents for FAQs. For more detailed instructions, refer to the **User Guide** documentation linked above.
    """)

           
    _, central_header_column, _ = st.columns((1, 2, 1))
    
    #title_filter = central_header_column.text_input("Filter")
    #pattern = re.compile(f".*{title_filter}.*", re.IGNORECASE)

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

    try:
        simple = application_options[session_state.app].SIMPLE
    except (AttributeError, KeyError):
        simple = False

    if simple:
        st.set_page_config(page_title="DocuScope CAC", page_icon=FAVICON, layout="centered")
    else:
        st.set_page_config(page_title="DocuScope CAC", page_icon=FAVICON, layout="wide")
        _content.local_css(STYLE)

    application_options = {**stable_apps}

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

        if not simple:
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


