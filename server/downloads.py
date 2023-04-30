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

import pathlib
import uuid
from typing import Union
from importlib.machinery import SourceFileLoader
from . import session

# set paths
HERE = pathlib.Path(__file__).parents[1].resolve()
OPTIONS = str(HERE.joinpath("options.toml"))
IMPORTS = str(HERE.joinpath("utilities/handlers_imports.py"))

# import options
_imports = SourceFileLoader("handlers_imports", IMPORTS).load_module()
_options = _imports.import_options_general(OPTIONS)

modules = ['streamlit']
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

FileContent = Union[str, bytes]


def download(name: str, content: FileContent):
    """Create a Streamlit download link to the given file content.

    Parameters
    ----------
    name : str
        The filename of the download. If a previous download link has
        been provided with the same download name the previous filepath
        will be overwritten.
    content : str or bytes
        The content of the file to be downloaded.
    """

    url = _add_file_get_url(name, content)

    href = f"""
        <a href="{url}" download='{name}'>
            Click to download `{name}`
        </a>
    """
    st.markdown(href, unsafe_allow_html=True)


def get_download_file(session_id: uuid.UUID, name: str) -> bytes:
    report_session = session.get_session(session_id=session_id)
    content: FileContent = report_session.downloads[name]

    if isinstance(content, str):
        content = content.encode()

    return content


def set_download_file(name: str, content: FileContent):
    report_session = session.get_session()

    try:
        report_session.downloads[name] = content
    except AttributeError:
        report_session.downloads = {name: content}


def _add_file_get_url(name: str, content: FileContent) -> str:
    session_id = session.get_session_id()
    url = f"downloads/{session_id}/{name}"

    set_download_file(name, content)

    return url
