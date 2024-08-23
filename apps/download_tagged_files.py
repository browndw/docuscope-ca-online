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

import polars as pl
import streamlit as st

import categories as _categories
import states as _states
from utilities import handlers_database as _handlers
from utilities import messages as _messages
	
CATEGORY = _categories.OTHER
TITLE = "Download Tagged Files"
KEY_SORT = 11

def main():
	
	user_session = st.runtime.scriptrunner.script_run_context.get_script_run_ctx()
	user_session_id = user_session.session_id

	if user_session_id not in st.session_state:
		st.session_state[user_session_id] = {}

	try:
		session = pl.DataFrame.to_dict(st.session_state[user_session_id]["session"], as_series=False)
	except:
		_handlers.init_session(user_session_id)
		session = pl.DataFrame.to_dict(st.session_state[user_session_id]["session"], as_series=False)

	st.markdown(_messages.message_download_tagged)
	
	st.sidebar.markdown("### Tagset to embed")
	download_radio = st.sidebar.radio("Select tagset:", ("Parts-of-Speech", "DocuScope"), horizontal=True)

	if download_radio == 'Parts-of-Speech':
		tagset = 'pos'
	else:
		tagset = 'ds'

	if session.get('has_target')[0] == True:
		tok_pl = st.session_state[user_session_id]["target"]["ds_tokens"]

		with st.sidebar:
			download_file = _handlers.convert_to_zip(tok_pl, tagset)

			st.download_button(
    			label="Download to Zip",
    			data=download_file,
    			file_name="tagged_files.zip",
   					 mime="application/zip",
					)

	
	st.sidebar.markdown("---")

if __name__ == "__main__":
    main()
    