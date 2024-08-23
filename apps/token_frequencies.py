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

import pathlib
import polars as pl
import streamlit as st

import categories as _categories
import states as _states
from utilities import handlers_database as _handlers
from utilities import analysis_functions as _analysis
from utilities import messages as _messages
from utilities import warnings as _warnings

CATEGORY = _categories.FREQUENCY
TITLE = "Token Frequencies"
KEY_SORT = 2

def main():

	user_session = st.runtime.scriptrunner.script_run_context.get_script_run_ctx()
	user_session_id = user_session.session_id

	user_session = st.runtime.scriptrunner.script_run_context.get_script_run_ctx()
	user_session_id = user_session.session_id

	if user_session_id not in st.session_state:
		st.session_state[user_session_id] = {}
	
	try:
		session = pl.DataFrame.to_dict(st.session_state[user_session_id]["session"], as_series=False)
	except:
		_handlers.init_session(user_session_id)
		session = pl.DataFrame.to_dict(st.session_state[user_session_id]["session"], as_series=False)
	
	if session.get('freq_table')[0] == True:
	
		_handlers.load_widget_state(pathlib.Path(__file__).stem, user_session_id)
		metadata_target = _handlers.load_metadata('target', user_session_id)
						
		st.sidebar.markdown("### Tagset")

		tag_radio = st.sidebar.radio("Select tags to display:", ("Parts-of-Speech", "DocuScope"), key = _handlers.persist("ft_radio", pathlib.Path(__file__).stem, user_session_id), horizontal=True)
		if tag_radio == 'Parts-of-Speech':
			tag_type = st.sidebar.radio("Select from general or specific tags", ("General", "Specific"), horizontal=True)			
			if tag_type == 'General':
				df = st.session_state[user_session_id]["target"]["ft_pos"]
				df = _analysis.freq_simplify_pl(df)
			else:
				df = st.session_state[user_session_id]["target"]["ft_pos"]
		else:			
			df = st.session_state[user_session_id]["target"]["ft_ds"]
	
		st.markdown(_messages.message_target_info(metadata_target))

		if df.height == 0 or df is None:
			cats = []
		elif df.height > 0:
			cats = sorted(df.get_column("Tag").unique().to_list())

		filter_vals = st.multiselect("Select tags to filter:", (cats))
		if len(filter_vals) > 0:
			df = df.filter(pl.col("Tag").is_in(filter_vals))

		st.dataframe(df, hide_index=True, 
				column_config={
					"Range": st.column_config.NumberColumn(format="%.2f %%"),
					"RF": st.column_config.NumberColumn(format="%.2f")}
		)
				
		with st.expander("Column explanation"):
			st.markdown(_messages.message_columns_tokens)
		
		st.sidebar.markdown("---")

		download_table = st.sidebar.toggle("Download to Excel?")
		if download_table == True:
			with st.sidebar:
				st.markdown(_messages.message_download)
				download_file = _handlers.convert_to_excel(df.to_pandas())

				st.download_button(
					label="Download to Excel",
					data=download_file,
					file_name="token_frequencies.xlsx",
						mime="application/vnd.ms-excel",
						)
		st.sidebar.markdown("---")
				
	else:
		st.markdown(_messages.message_tables)
		st.sidebar.markdown(_messages.message_generate_table)

		if st.sidebar.button("Frequency Table"):
			if session.get('has_target')[0] == False:
				st.markdown(_warnings.warning_11, unsafe_allow_html=True)
			
			else:
				with st.sidebar:
					with st.spinner('Processing frequencies...'):
						_handlers.update_session('freq_table', True, user_session_id)
					
					st.rerun()
		
		st.sidebar.markdown("---")

if __name__ == "__main__":
    main()
