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
from typing import BinaryIO, Union
from importlib.machinery import SourceFileLoader

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


File = Union[pathlib.Path, str, BinaryIO]

class SessionState:
    def __init__(self, state):
        for key, val in state.items():
            setattr(self, key, val)

    def update(self, state):
        for key, val in state.items():
            try:
                getattr(self, key)
            except AttributeError:
                setattr(self, key, val)

def session_state(**state):
    session = get_session()

    try:
        session._custom_session_state.update(state)
    except AttributeError:
        session._custom_session_state = SessionState(state)

    return session._custom_session_state


if _options['global']['desktop_mode'] == True and hasattr(st, 'scriptrunner'):	
	
	def get_session_id() -> str:
	    ctx = st.scriptrunner.add_script_run_ctx()
	    session_id: str = ctx.streamlit_script_run_ctx.session_id
	
	    return session_id
	
	
	def get_session(session_id: str = None):
	    if session_id is None:
	        session_id = get_session_id()
	
	    session_info = st.server.server.Server.get_current()._get_session_info(session_id)
	
	    if session_info is None:
	        raise ValueError("No session info found")
	
	    report_session = session_info.session
	
	    return report_session
	
	
else:
	
	def get_session_id() -> str:
	    ctx = st.runtime.scriptrunner.add_script_run_ctx()
	    session_id: str = ctx.streamlit_script_run_ctx.session_id
	
	    return session_id
	
	
	def get_session(session_id: str = None):
	    if session_id is None:
	        session_id = get_session_id()
	
	    session_info = st.runtime.get_instance()._session_mgr.get_active_session_info(session_id)
	
	    if session_info is None:
	        raise ValueError("No session info found")
	
	    report_session = session_info.session
	
	    return report_session
