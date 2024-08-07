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

CATEGORY = _categories.FREQUENCY
TITLE = "N-gram Frequencies"
KEY_SORT = 4

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


	if session.get('ngrams')[0] == True:
		
		metadata_target = _handlers.load_metadata('target', con)
		
		df = con.table("ngrams", database="target").to_pyarrow_batches(chunk_size=5000)
		df = pl.from_arrow(df).to_pandas()
	
		st.markdown(_messages.message_target_info(metadata_target))
			
		gb = st_aggrid.GridOptionsBuilder.from_dataframe(df)
		gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=100) #Add pagination
		gb.configure_default_column(filter="agTextColumnFilter")
		gb.configure_column("Token_1", filter="agTextColumnFilter", headerCheckboxSelection = True, headerCheckboxSelectionFilteredOnly = True)
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
    			file_name="ngrams.xlsx",
   					 mime="application/vnd.ms-excel",
					)
		st.sidebar.markdown("---")

		st.sidebar.markdown("### Generate new table")
		st.sidebar.markdown("""
							Click the button to reset the n-grams table.
							""")
	
		if st.sidebar.button("Create a New Ngrams Table"):
			try:
				con.drop_table("ngrams", database="target")
			except:
				pass
			_handlers.update_session('ngrams', False, con)
			st.rerun()
		st.sidebar.markdown("---")
			
	else:
		
		try:
			metadata_target = _handlers.load_metadata('target', con)
		except:
			pass
		
		st.markdown(_messages.message_ngrams)

		st.markdown("---")

		ngram_type = st.radio("What kind of table would you like to generate?",
			["N-grams", "Clusters"],
			captions=[":abacus: Create a table of n-grams with a relative frequency > 10 (per million words).",
			":link: Create counts of clusters that contain a specific word, part-of-a-word, or tag."], 
			horizontal=False,
			index=None)
		
		if ngram_type == 'N-grams':
			st.sidebar.markdown("### Span")
			ngram_span = st.sidebar.radio('Span of your n-grams:', (2, 3, 4), horizontal=True)

			st.sidebar.markdown("---")

			tag_radio = st.sidebar.radio("Select a tagset:", ("Parts-of-Speech", "DocuScope"), horizontal=True)
			if tag_radio == 'Parts-of-Speech':
				ts = 'pos'
			if tag_radio == 'DocuScope':
				ts = 'ds'

			st.sidebar.markdown("---")
			
			st.sidebar.markdown(_messages.message_generate_table)

			if st.sidebar.button("N-grams Table"):
				if session.get('has_target')[0] == False:
					st.markdown(_warnings.warning_11, unsafe_allow_html=True)
				else:
					with st.sidebar:
						with st.spinner('Processing n-grams...'):
							tok_pl = con.table("ds_tokens", database="target").to_pyarrow_batches(chunk_size=5000)
							tok_pl = pl.from_arrow(tok_pl)
							
							ngram_df = _analysis.ngrams_pl(tok_pl, ngram_span, count_by=ts)
					
					#cap size of dataframe
					if ngram_df.height < 2:
						st.markdown(_warnings.warning_12, unsafe_allow_html=True)
					else:
						con.create_table("ngrams", obj=ngram_df, database="target", overwrite=True)
						_handlers.update_session('ngrams', True, con)
						st.rerun()

			
			st.sidebar.markdown("---")

		
		if ngram_type == 'Clusters':
		
			st.sidebar.markdown("### Search mode")
			st.sidebar.markdown("Create n-grams from a token or from a tag.")
			from_anchor = st.sidebar.radio("Enter token or a tag:", ("Token", "Tag"), horizontal=True)
			
			if from_anchor == 'Token':
				node_word = st.sidebar.text_input("Node word:")	
			
				search_mode = st.sidebar.radio("Select search type:", ("Fixed", "Starts with", "Ends with", "Contains"), horizontal=True)		
				if search_mode == "Fixed":
					search = "fixed"
				elif search_mode == "Starts with":
					search = "starts_with"
				elif search_mode == "Ends with":
					search = "ends_with"
				else:
					search = "contains"
				
				tag_radio = st.sidebar.radio("Select a tagset:", ("Parts-of-Speech", "DocuScope"), horizontal=True)
				
				if tag_radio == 'Parts-of-Speech':
					ts = 'pos'
				if tag_radio == 'DocuScope':
					ts = 'ds'
				
			if from_anchor == 'Tag':
				tag_radio = st.sidebar.radio("Select a tagset:", ("Parts-of-Speech", "DocuScope"), horizontal=True)
				
				if tag_radio == 'Parts-of-Speech':
					if session.get('has_target')[0] == False:
						tag = st.sidebar.selectbox('Choose a tag:', ['No tags currently loaded'])
					else:
						tag = st.sidebar.selectbox('Choose a tag:', metadata_target.get('tags_pos')[0]['tags'])
						ts = 'pos'
						node_word = 'by_tag'
				if tag_radio == 'DocuScope':
					if session.get('has_target')[0] == False:
						tag = st.sidebar.selectbox('Choose a tag:', ['No tags currently loaded'])
					else:
						tag = st.sidebar.selectbox('Choose a tag:', metadata_target.get('tags_ds')[0]['tags'])
						ts = 'ds'
						node_word = 'by_tag'
			
			st.sidebar.markdown("---")
			
			st.sidebar.markdown("### Span & position")
			ngram_span = st.sidebar.radio('Span of your n-grams:', (2, 3, 4), horizontal=True)
			position = st.sidebar.selectbox('Position of your node word or tag:', (list(range(1, 1+ngram_span))))

			st.sidebar.markdown("---")

			st.sidebar.markdown(_messages.message_generate_table)
			
			if st.sidebar.button("N-grams Table"):
				if session.get('has_target')[0] == False:
					st.markdown(_warnings.warning_11, unsafe_allow_html=True)
				elif node_word == "":
					st.write(_warnings.warning_14, unsafe_allow_html=True)
				elif node_word.count(" ") > 0:
					st.write(_warnings.warning_15, unsafe_allow_html=True)
				elif len(node_word) > 15:
					st.write(_warnings.warning_16, unsafe_allow_html=True)
				else:
					with st.sidebar:
						with st.spinner('Processing n-grams...'):
							tok_pl = con.table("ds_tokens", database="target").to_pyarrow_batches(chunk_size=5000)
							tok_pl = pl.from_arrow(tok_pl)

							if from_anchor == 'Token':
								ngram_df = _analysis.ngrams_by_token_pl(tok_pl, node_word, position, ngram_span, search, ts)
								#cap size of dataframe
							if from_anchor == 'Tag':
								ngram_df = _analysis.ngrams_by_tag_pl(tok_pl, tag, position, ngram_span, ts)
					
					#cap size of dataframe
					if ngram_df is None or ngram_df.height == 0:
						st.markdown(_warnings.warning_12, unsafe_allow_html=True)
					elif ngram_df.height > 100000:
						st.markdown(_warnings.warning_13, unsafe_allow_html=True)
					else:
						con.create_table("ngrams", obj=ngram_df, database="target", overwrite=True)
						_handlers.update_session('ngrams', True, con)
						st.rerun()
					
			st.sidebar.markdown("---")

if __name__ == "__main__":
    main()