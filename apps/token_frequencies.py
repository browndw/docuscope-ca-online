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

import pandas as pd
import pathlib
import polars as pl
import st_aggrid
import st_aggrid.grid_options_builder
import streamlit as st

import categories as _categories
import states as _states
from utilities import handlers_database as _handlers
from utilities import messages as _messages
from utilities import warnings as _warnings

CATEGORY = _categories.FREQUENCY
TITLE = "Token Frequencies"
KEY_SORT = 2

def main():

	user_session = st.runtime.scriptrunner.script_run_context.get_script_run_ctx()
	user_session_id = user_session.session_id

	if user_session_id not in st.session_state:
		st.session_state[user_session_id] = {}
	try:
		con = st.session_state[user_session_id]["ibis_conn"]
	except:
		con = _handlers.get_db_connection(user_session_id)
		_handlers.generate_temp(_states.STATES.items(), user_session_id, con)

	try:
		session = pl.DataFrame.to_dict(con.table("session").to_polars(), as_series=False)
	except:
		_handlers.init_session(con)
		session = pl.DataFrame.to_dict(con.table("session").to_polars(), as_series=False)
	
	if session.get('freq_table')[0] == True:
	
		_handlers.load_widget_state(pathlib.Path(__file__).stem, user_session_id)
		metadata_target = _handlers.load_metadata('target', con)
						
		st.sidebar.markdown("### Tagset")
		
		tag_radio = st.sidebar.radio("Select tags to display:", ("Parts-of-Speech", "DocuScope"), key = _handlers.persist("ft_radio", pathlib.Path(__file__).stem, user_session_id), horizontal=True)

		if tag_radio == 'Parts-of-Speech':			
			df = con.table("ft_pos", database="target").to_pyarrow_batches(chunk_size=5000)
			df = pl.from_arrow(df).to_pandas()
		if tag_radio == 'DocuScope':			
			df = con.table("ft_ds", database="target").to_pyarrow_batches(chunk_size=5000)
			df = pl.from_arrow(df).to_pandas()
	
		st.markdown(_messages.message_target_info(metadata_target))
		
		gb = st_aggrid.GridOptionsBuilder.from_dataframe(df)
		gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=100) #Add pagination
		gb.configure_column("Token", filter="agTextColumnFilter", headerCheckboxSelection = True, headerCheckboxSelectionFilteredOnly = True)
		gb.configure_column("Tag", filter="agTextColumnFilter")
		gb.configure_column("RF", type=["numericColumn","numberColumnFilter","customNumericFormat"], precision=2)
		gb.configure_column("Range", type=["numericColumn","numberColumnFilter"], valueFormatter="(data.Range).toFixed(1)+'%'")
		gb.configure_selection('multiple', use_checkbox=True, groupSelectsChildren="Group checkbox select children") #Enable multi-row selection
		gb.configure_grid_options(sideBar = {"toolPanels": ['filters']})
		go = gb.build()
		
		grid_response = st_aggrid.AgGrid(
			df,
			gridOptions=go,
			enable_enterprise_modules = False,
			data_return_mode='FILTERED_AND_SORTED',
			update_mode='MODEL_CHANGED',
			columns_auto_size_mode='FIT_CONTENTS',
			theme='alpine',
			height=500, 
			width='100%',
			reload_data=False
			)
				
		with st.expander("Column explanation"):
			st.markdown(_messages.message_columns_tokens)
		
		st.sidebar.markdown("---")
			
		selected = grid_response['selected_rows']

		if selected is not None:
			df = pd.DataFrame(selected)
			n_selected = len(df.index)
			st.markdown(f"""##### Selected rows:
			   
			Number of selected tokens: {n_selected}
			""")

		with st.sidebar.expander("Filtering and saving"):
			st.markdown(_messages.message_filters)
	
		with st.sidebar:
			st.markdown(_messages.message_download)
			download_file = _handlers.convert_to_excel(df)

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
						_handlers.update_session('freq_table', True, con)
					
					st.rerun()
		
		st.sidebar.markdown("---")

if __name__ == "__main__":
    main()
