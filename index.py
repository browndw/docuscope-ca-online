import functools
import pathlib
import re
import textwrap
from PIL import Image

import time
import streamlit as st
import os
import base64

import apps as _stable_apps
import utilities
import categories as _categories
import states as _states

HERE = pathlib.Path(__file__).parent.resolve()
FAVICON = str(HERE.joinpath("_static/docuscope-favicon.ico"))
TITLE_LOGO = str(HERE.joinpath("_static/docuscope-logo.png"))
PL_LOGO = str(HERE.joinpath("_static/porpoise_badge.svg"))
UG_LOGO = str(HERE.joinpath("_static/user_guide.svg"))
STYLE = str(HERE.joinpath("css/style.css"))

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
	    Corpus Analysis & Concordancer Online
	  </h2>

	</div>
      '''
    return html_code
    
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

nav_header = """
    <div style="background-color: #FFE380; padding-top:12px; border-radius: 5px">
    <h5 style="color: black; text-align:center;">Common Tools</h5>
    </div>
    """
    
def get_url_app():
    try:
        return st.experimental_get_query_params()["app"][0]
    except KeyError:
        return "index"

def swap_app(app):
    st.experimental_set_query_params(app=app)

    session_state = utilities.session_state()
    session_state.app = app

    # Not sure why this is needed. The `set_query_params` doesn't
    # appear to work if a rerun is undergone immediately afterwards.
    time.sleep(0.01)
    st.experimental_rerun()

def _application_sorting_key(application):
    return application[1].KEY_SORT

def _get_apps_from_module(module):
    apps = {
        item.replace("_", "-"): getattr(module, item)
        for item in dir(module)
        if not item.startswith("_")
    }

    return apps

def index(application_options):
    # set all states from dictionary
    for key, value in _states.STATES.items():
        if key not in st.session_state:
            setattr(st.session_state, key, value)
    
    with open(PL_LOGO) as f:
        pl_logo_text = f.read()
    b64 = base64.b64encode(pl_logo_text.encode('utf-8')).decode("utf-8")
    pl_html = r'<a href="https://github.com/browndw/corpus-tagger/"><img src="data:image/svg+xml;base64,%s"/></a>' % b64
    
    st.markdown(pl_html, unsafe_allow_html=True)

    st.markdown(get_img_with_header(TITLE_LOGO), unsafe_allow_html=True)
        
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
                applications_in_this_category, key=_application_sorting_key
            )

            for app_key, application in applications_in_this_category:
                if column.button(application.TITLE):
                    swap_app(app_key)
    
def main():
    session_state = utilities.session_state(app=get_url_app())

    stable_apps = _get_apps_from_module(_stable_apps)
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
        st.set_page_config(page_title="DocuScope CA", page_icon=FAVICON, layout="centered")
    else:
        st.set_page_config(page_title="DocuScope CA", page_icon=FAVICON, layout="wide")
        local_css(STYLE)

    application_options = {**stable_apps}

    if (
        session_state.app != "index"
        and not session_state.app in application_options.keys()
    ):
        swap_app("index")

    if session_state.app != "index":        
        st.markdown(nav_header, unsafe_allow_html=True)
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
                    (f for f in applications_in_this_category if 1 < _application_sorting_key(f) < 11),
                    key=_application_sorting_key
                )
    
                for app_key, application in applications_in_this_category:
                    if column.button(application.TITLE):
                        swap_app(app_key)   
        
        if st.button("Manage Corpora"):
            swap_app("load-corpus")
        st.markdown("""---""")
        st.title(application_options[session_state.app].TITLE)
        
        docstring = application_options[session_state.app].main.__doc__
        if docstring is not None:
            docstring = textwrap.dedent(f"    {docstring}")
            st.write(docstring)

        if not simple:
            if st.sidebar.button("Return to All Tools"):
                swap_app("index")
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


