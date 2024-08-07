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
import polars as pl
import st_aggrid
import streamlit as st

import categories as _categories
import states as _states
from utilities import analysis_functions as _analysis
from utilities import handlers_database as _handlers
from utilities import messages as _messages
from utilities import warnings as _warnings

CATEGORY = _categories.COLLOCATION
TITLE = "Collocates"
KEY_SORT = 7

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

	if session.get('collocations')[0] == True:
		
		metadata_target = _handlers.load_metadata('target', con)

		df = con.table("collocations", database="target").to_pyarrow_batches(chunk_size=5000)
		df = pl.from_arrow(df).to_pandas()
		
		col1, col2 = st.columns([1,1])
		with col1:
			st.markdown(_messages.message_target_info(metadata_target))
		with col2:
			st.markdown(_messages.message_collocation_info(metadata_target.get('collocations')[0]['temp']))
			
		gb = st_aggrid.GridOptionsBuilder.from_dataframe(df)
		gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=100) #Add pagination
		gb.configure_default_column(filter="agTextColumnFilter")
		gb.configure_column("Token", filter="agTextColumnFilter", headerCheckboxSelection = True, headerCheckboxSelectionFilteredOnly = True)
		gb.configure_column("MI", type=["numericColumn","numberColumnFilter","customNumericFormat"], precision=3)
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
			st.markdown(_messages.message_columns_collocations)
	
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
    			file_name="collocations.xlsx",
   					 mime="application/vnd.ms-excel",
					)
		st.sidebar.markdown("---")

		st.sidebar.markdown(_messages.message_reset_table)
		
		if st.sidebar.button("Create New Collocations Table"):
			try:
				con.drop_table("collocations", database="target")
			except:
				pass
			_handlers.update_session('collocations', False, con)
			st.rerun()
		st.sidebar.markdown("---")
				
	else:

		st.markdown(_messages.message_collocations)

		if session.get("has_target")[0] == True:
			metadata_target = _handlers.load_metadata('target', con)

		st.sidebar.markdown("### Node word")
		st.sidebar.markdown("""Enter a node word without spaces.
					""")				
		node_word = st.sidebar.text_input("Node word:")
							
		st.sidebar.markdown("---")
		with st.sidebar.expander("Span explanation"):
			st.markdown(_messages.message_span)
		
		st.sidebar.markdown("### Span")
		to_left = st.sidebar.slider("Choose a span to the left of the node word:", 0, 9, (4))
		to_right = st.sidebar.slider("Choose a span to the right of the node word:", 0, 9, (4))
		
		st.sidebar.markdown("---")
		with st.sidebar.expander("Statistics explanation"):
			st.markdown(_messages.message_association_measures)

		st.sidebar.markdown("### Association measure")			
		stat_mode = st.sidebar.radio("Select a statistic:",
							   ["NPMI", "PMI 2", "PMI 3", "PMI"], 
							   horizontal=True)
		
		if stat_mode == "PMI":
			stat_mode = "pmi"
		elif stat_mode == "PMI 2":
			stat_mode = "pmi2"
		elif stat_mode == "PMI 3":
			stat_mode = "pmi3"
		elif stat_mode == "NPMI":
			stat_mode = "npmi"
		
		st.sidebar.markdown("---")
		with st.sidebar.expander("Anchor tag for node word explanation"):
			st.markdown(_messages.message_anchor_tags)
		
		st.sidebar.markdown("### Anchor tag")
		tag_radio = st.sidebar.radio("Select tagset for node word:", ("No Tag", "Parts-of-Speech", "DocuScope"), horizontal=True)
		if tag_radio == 'Parts-of-Speech':
			tag_type = st.sidebar.radio("Select from general or specific tags", ("General", "Specific"), horizontal=True)
			if tag_type == 'General':
				node_tag = st.sidebar.selectbox("Select tag:", ("Noun Common", "Verb Lex", "Adjective", "Adverb"))
				if node_tag == "Noun Common":
					node_tag = "NN"
				elif node_tag == "Verb Lex":
					node_tag = "VV"
				elif node_tag == "Adjective":
					node_tag = "JJ"
				elif node_tag == "Adverb":
					node_tag = "R"
			else:
				if session.get('has_target')[0] == False:
					node_tag = st.sidebar.selectbox('Choose a tag:', ['No tags currently loaded'])
				else:
					node_tag = st.sidebar.selectbox('Choose a tag:', metadata_target.get('tags_pos')[0]['tags'])
			ignore_tags = False
			count_by = 'pos'
		elif tag_radio == 'DocuScope':
			if session.get('has_target')[0] == False:
				node_tag = st.sidebar.selectbox('Choose a tag:', ['No tags currently loaded'])
			else:
				node_tag = st.sidebar.selectbox('Choose a tag:', metadata_target.get('tags_ds')[0]['tags'])
				ignore_tags = False
				count_by = 'ds'
		else:
			node_tag = None
			ignore_tags = False
			count_by = 'pos'
		
		st.sidebar.markdown("---")
		st.sidebar.markdown(_messages.message_generate_table)
		if st.sidebar.button("Collocations"):
			if session.get('has_target')[0] == False:
				st.markdown(_warnings.warning_11, unsafe_allow_html=True)
			elif node_word == "":
				st.markdown(_warnings.warning_14, unsafe_allow_html=True)
			elif node_word.count(" ") > 0:
				st.markdown(_warnings.warning_15, unsafe_allow_html=True)
			elif len(node_word) > 15:
				st.markdown(_warnings.warning_16, unsafe_allow_html=True)
			else:
				with st.sidebar:
					with st.spinner('Processing collocates...'):
						tok_pl = con.table("ds_tokens", database="target").to_pyarrow_batches(chunk_size=5000)
						tok_pl = pl.from_arrow(tok_pl)

						coll_df = _analysis.collocations_pl(tok_pl, node_word=node_word, node_tag=node_tag, preceding=to_left, following=to_right, statistic=stat_mode, count_by=count_by)
				
				if coll_df.is_empty():
					st.markdown(_warnings.warning_12, unsafe_allow_html=True)
					
				else:
					con.create_table("collocations", obj=coll_df, database="target", overwrite=True)
					_handlers.update_session('collocations', True, con)
					_handlers.update_metadata('target', key='collocations', value=[node_word, stat_mode, str(to_left), str(to_right)], ibis_conn=con)
					st.rerun()

	
		st.sidebar.markdown("---")
		
if __name__ == "__main__":
    main()