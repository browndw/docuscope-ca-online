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
import pathlib
import os
import time

import streamlit as st

import utilities

# set paths
HERE = pathlib.Path(__file__).parents[1].resolve()
OPTIONS = str(HERE.joinpath("options.toml"))
IMPORTS = str(HERE.joinpath("utilities/handlers_imports.py"))

message_landing = """
	Welcome to **DocuScope Corpus Analaysis & Concordancer**.
	This suite of tools is designed to help those new to corpus analysis and NLP explore data, data visualization, and the computational analysis of text. 
	It is also designed to allow users to easily toggle between **rhetorical tags** and more conventional **part-of-speech tags**.
	The application in available online and as a desktop application. The online version resitricts the amount of data that you can process at one time.
	Both versions are open source and can be accessed from the GitHub repository linked at the top of this page.
	To get started:
        
	:point_right: Use one of the pre-processed corpora or prepare a corpus of your own in **plain text files**. (The tool does not accept Word files, PDFs, etc.)
        
	:point_right: Use **Manage Corpus Data** to load the data you want to explore.
        
	:point_right: Refer to the **Help** documents for FAQs. For more detailed instructions, refer to the **User Guide** documentation linked above.
	"""

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
        return st.query_params["app"][0]
    except KeyError:
        return "index"

def swap_app(app):
    st.query_params.app = app

    session_state = utilities.session_state()
    session_state.app = app

    # Not sure why this is needed. The `set_query_params` doesn't
    # appear to work if a rerun is undergone immediately afterwards.
    time.sleep(0.01)
    st.rerun()

def _application_sorting_key(application):
    return application[1].KEY_SORT

def _get_apps_from_module(module):
    apps = {
        item.replace("_", "-"): getattr(module, item)
        for item in dir(module)
        if not item.startswith("_")
    }

    return apps
