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

CATEGORY = _categories.COLLOCATION
TITLE = "Collocates"
KEY_SORT = 7

def main():

	user_session = st.runtime.scriptrunner.script_run_context.get_script_run_ctx()
	user_session_id = user_session.session_id

	session = _handlers.load_session(user_session_id)

	if bool(session['collocations']) == True:
		
		metadata_target = _handlers.load_metadata('target', user_session_id)
		df = _handlers.load_table('collocations', user_session_id)
		
		col1, col2 = st.columns([1,1])
		with col1:
			st.markdown(_messages.message_target_info(metadata_target))
		with col2:
			st.markdown(_messages.message_collocation_info(session['collocations']))
			
		gb = st_aggrid.GridOptionsBuilder.from_dataframe(df)
		gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=25) #Add pagination
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
			height=None,
			width='100%',
			reload_data=False
			)
		
		with st.expander("Column explanation"):
			st.markdown(_messages.message_columns_collocations)
	
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
					linko= f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="collocations.xlsx">Download Excel file</a>'
					st.markdown(linko, unsafe_allow_html=True)
		
		st.sidebar.markdown("---")
		st.sidebar.markdown(_messages.message_reset_table)
		
		if st.sidebar.button("Create New Collocations Table"):
			_handlers.clear_table('collocations', user_session_id)
			_handlers.update_session('collocations', dict(), user_session_id)
			st.experimental_rerun()
		st.sidebar.markdown("---")
				
	else:
	
		try:
			metadata_target = _handlers.load_metadata('target', user_session_id)
		except:
			metadata_target = {}
		
		st.markdown(_messages.message_collocations)

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
		stat_mode = st.sidebar.radio("Select a statistic:", ("PMI", "NPMI", "PMI 2", "PMI 3"), horizontal=True)
		
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
				node_tag = st.sidebar.selectbox("Select tag:", ("Noun", "Verb", "Adjective", "Adverb"))
				if node_tag == "Noun":
					node_tag = "NN"
				elif node_tag == "Verb":
					node_tag = "VV"
				elif node_tag == "Adjective":
					node_tag = "JJ"
				elif node_tag == "Adverb":
					node_tag = "R"
			else:
				if session.get('target_path') == None:
					node_tag = st.sidebar.selectbox('Choose a tag:', ['No tags currently loaded'])
				else:
					node_tag = st.sidebar.selectbox('Choose a tag:', metadata_target.get('tags_pos'))
			ignore_tags = False
			count_by = 'pos'
		elif tag_radio == 'DocuScope':
			if session.get('target_path') == None:
				node_tag = st.sidebar.selectbox('Choose a tag:', ['No tags currently loaded'])
			else:
				node_tag = st.sidebar.selectbox('Choose a tag:', metadata_target.get('tags_ds'))
				ignore_tags = False
				count_by = 'ds'
		else:
			node_tag = None
			ignore_tags = False
			count_by = 'pos'
		
		st.sidebar.markdown("---")
		st.sidebar.markdown(_messages.message_generate_table)
		if st.sidebar.button("Collocations"):
			if session.get('target_path') == None:
				st.markdown(_warnings.warning_11, unsafe_allow_html=True)
			elif node_word == "":
				st.markdown(_warnings.warning_14, unsafe_allow_html=True)
			elif node_word.count(" ") > 0:
				st.markdown(_warnings.warning_15, unsafe_allow_html=True)
			elif len(node_word) > 15:
				st.markdown(_warnings.warning_16, unsafe_allow_html=True)
			else:
				tp = _handlers.load_corpus_session('target', session, user_session_id)
				metadata_target = _handlers.load_metadata('target', user_session_id)
				with st.sidebar:
					with st.spinner('Processing collocates...'):
						coll_df = _analysis.coll_table(tp, node_word=node_word, node_tag=node_tag, l_span=to_left, r_span=to_right, statistic=stat_mode, tag_ignore=ignore_tags, count_by=count_by)
				if len(coll_df.index) > 0:
					_handlers.save_table(coll_df, 'collocations', user_session_id)
					_handlers.update_collocations(node_word, stat_mode, to_left, to_right, user_session_id)
					st.experimental_rerun()
				else:
					st.markdown(_warnings.warning_12, unsafe_allow_html=True)
	
		st.sidebar.markdown("---")
		
if __name__ == "__main__":
    main()