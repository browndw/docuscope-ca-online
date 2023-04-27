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
import os
import time
import tomli
import pathlib

HERE = pathlib.Path(__file__).parents[1].resolve()
OPTIONS = str(HERE.joinpath("options.toml"))

# import options
with open(OPTIONS, mode="rb") as fp:
	_options = tomli.load(fp)

DESKTOP = _options['global']['desktop_mode']

if DESKTOP == True:
	from docuscope._streamlit import utilities
	from docuscope._imports import streamlit as st
         
if DESKTOP == False:
	import utilities
	import streamlit as st

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
