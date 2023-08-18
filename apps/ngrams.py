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
from io import BytesIO
import pathlib
from importlib.machinery import SourceFileLoader

# set paths
HERE = pathlib.Path(__file__).parents[1].resolve()
OPTIONS = str(HERE.joinpath("options.toml"))
IMPORTS = str(HERE.joinpath("utilities/handlers_imports.py"))

# import options
_imports = SourceFileLoader("handlers_imports", IMPORTS).load_module()
_options = _imports.import_options_general(OPTIONS)

modules = ['analysis', 'categories', 'handlers', 'messages', 'states', 'warnings', 'streamlit', 'docuscospacy', 'pandas', 'st_aggrid']
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


CATEGORY = _categories.FREQUENCY
TITLE = "N-gram Frequencies"
KEY_SORT = 4

def main():

	user_session = st.runtime.scriptrunner.script_run_context.get_script_run_ctx()
	user_session_id = user_session.session_id

	session = _handlers.load_session(user_session_id)

	if session.get('ngrams') == True:
		
		metadata_target = _handlers.load_metadata('target', user_session_id)
		df = _handlers.load_table('ngrams', user_session_id)
	
		st.markdown(_messages.message_target_info(metadata_target))
			
		gb = st_aggrid.GridOptionsBuilder.from_dataframe(df)
		gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=25) #Add pagination
		gb.configure_default_column(filter="agTextColumnFilter")
		gb.configure_column("Token1", filter="agTextColumnFilter", headerCheckboxSelection = True, headerCheckboxSelectionFilteredOnly = True)
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
			height=None, 
			width='100%',
			reload_data=False
			)
			
		with st.expander("Column explanation"):
			st.markdown(_messages.message_columns_tokens)
			
		selected = grid_response['selected_rows'] 
		if selected:
			st.write('Selected rows')
			df = pd.DataFrame(selected).drop('_selectedRowNodeInfo', axis=1)
			st.dataframe(df)
		
		with st.sidebar.expander("Filtering and saving"):
			st.markdown(_messages.message_filters)
		
		st.sidebar.markdown(_messages.message_download)
		if st.sidebar.button("Download"):
			with st.sidebar:
				with st.spinner('Creating download link...'):
					towrite = BytesIO()
					downloaded_file = df.to_excel(towrite, encoding='utf-8', index=False, header=True)
					towrite.seek(0)  # reset pointer
					b64 = base64.b64encode(towrite.read()).decode()  # some strings
					st.success('Link generated!')
					linko= f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="ngram_frequencies.xlsx">Download Excel file</a>'
					st.markdown(linko, unsafe_allow_html=True)

		st.sidebar.markdown("---")
		st.sidebar.markdown("### Generate new table")
		st.sidebar.markdown("""
							Click the button to reset the n-grams table.
							""")
	
		if st.sidebar.button("Create a New Ngrams Table"):
			_handlers.clear_table('ngrams', user_session_id)
			_handlers.update_session('ngrams', False, user_session_id)
			st.experimental_rerun()
		st.sidebar.markdown("---")			
			
	else:
		
		try:
			metadata_target = _handlers.load_metadata('target', user_session_id)
		except:
			metadata_target = {}
		
		st.markdown(_messages.message_ngrams)
		
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
				if session.get('target_path') == None:
					total = 0
				else:
					total = metadata_target.get('words')
			if tag_radio == 'DocuScope':
				ts = 'ds'
				if session.get('target_path') == None:
					total = 0
				else:
					total = metadata_target.get('tokens')
			
		if from_anchor == 'Tag':
			tag_radio = st.sidebar.radio("Select a tagset:", ("Parts-of-Speech", "DocuScope"), horizontal=True)
			
			if tag_radio == 'Parts-of-Speech':
				if session.get('target_path') == None:
					tag = st.sidebar.selectbox('Choose a tag:', ['No tags currently loaded'])
				else:
					tag = st.sidebar.selectbox('Choose a tag:', metadata_target.get('tags_pos'))
					ts = 'pos'
					total = metadata_target.get('words')
					node_word = 'by_tag'
			if tag_radio == 'DocuScope':
				if session.get('target_path') == None:
					tag = st.sidebar.selectbox('Choose a tag:', ['No tags currently loaded'])
				else:
					tag = st.sidebar.selectbox('Choose a tag:', metadata_target.get('tags_ds'))
					ts = 'ds'
					total = metadata_target.get('tokens')
					node_word = 'by_tag'
		
		st.sidebar.markdown("---")
		
		st.sidebar.markdown("### Span & position")
		ngram_span = st.sidebar.radio('Span of your n-grams:', (2, 3, 4), horizontal=True)
		position = st.sidebar.selectbox('Position of your node word or tag:', (list(range(1, 1+ngram_span))))

		st.sidebar.markdown(_messages.message_generate_table)
		
		if st.sidebar.button("N-grams Table"):
			if session.get('target_path') == None:
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
						tp = _handlers.load_corpus_session('target', session, user_session_id)
						metadata_target = _handlers.load_metadata('target', user_session_id)
						if from_anchor == 'Token':
							ngram_df = _analysis.ngrams_by_token(tp, node_word, position, ngram_span, total, search, ts)
							#cap size of dataframe
						if from_anchor == 'Tag':
							ngram_df = _analysis.ngrams_by_tag(tp, tag, position, ngram_span, total, ts)
				#cap size of dataframe
				if len(ngram_df.index) < 2:
					st.markdown(_warnings.warning_12, unsafe_allow_html=True)
				elif len(ngram_df.index) > 100000:
					st.markdown(_warnings.warning_13, unsafe_allow_html=True)
				else:
					_handlers.save_table(ngram_df, 'ngrams', user_session_id)
					_handlers.update_session('ngrams', True, user_session_id)
					st.experimental_rerun()				
				
		st.sidebar.markdown("---")

if __name__ == "__main__":
    main()