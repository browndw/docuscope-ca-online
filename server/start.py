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

# pylint: disable = protected-access

from docuscope._imports import streamlit as st
from docuscope._imports import st_runtime

from . import patches

def main(args):
    start_streamlit_server(args.path, {})


def start_streamlit_server(script_path, config):
    st.web.bootstrap.load_config_options(flag_options=config)
    patches.apply_streamlit_server_patches()

    st_runtime.exists()
    st.web.bootstrap.run(script_path, args=[], flag_options={}, is_hello=False)
