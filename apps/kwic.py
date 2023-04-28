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

modules = ['analysis', 'categories', 'handlers', 'messages', 'states', 'warnings', 'streamlit', 'pandas', 'st_aggrid']
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


CATEGORY = _categories.OTHER
TITLE = "KWIC Tables"
KEY_SORT = 8

def main():

	session = _handlers.load_session()
	
	if session.get('kwic') == True:
		
		df = _handlers.load_table('kwic')	
		
		gb = st_aggrid.GridOptionsBuilder.from_dataframe(df)
		gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=100) #Add pagination
		gb.configure_default_column(filter="agTextColumnFilter")
		gb.configure_column("Doc ID", filter="agTextColumnFilter", headerCheckboxSelection = True, headerCheckboxSelectionFilteredOnly = True)
		gb.configure_column("Pre-Node", type="rightAligned")
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
					linko= f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="kwic.xlsx">Download Excel file</a>'
					st.markdown(linko, unsafe_allow_html=True)
		
		st.sidebar.markdown("---")
		st.sidebar.markdown(_messages.message_reset_table)
													
		if st.sidebar.button("Create New KWIC Table"):
				_handlers.clear_table('kwic')
				_handlers.update_session('kwic', False)
				st.experimental_rerun()
		st.sidebar.markdown("---")
			
	else:
		
		try:
			metadata_target = _handlers.load_metadata('target')
		except:
			metadata_target = {}
		
		st.markdown(_messages.message_kwic)
		
		st.sidebar.markdown("### Node word")
		st.sidebar.markdown("""Use the text field to enter a node word and other desired options.
					Once a node word has been entered and options selected, click the button below to generate a KWIC table.
					""")				
		node_word = st.sidebar.text_input("Node word")
		
		st.sidebar.markdown("---")
		st.sidebar.markdown("### Search mode")
		search_mode = st.sidebar.radio("Select search type:", ("Fixed", "Starts with", "Ends with", "Contains"), horizontal=True)
		
		if search_mode == "Fixed":
			search_type = "fixed"
		elif search_mode == "Starts with":
			search_type = "starts_with"
		elif search_mode == "Ends with":
			search_type = "ends_with"
		else:
			search_type = "contains"
		
		st.sidebar.markdown("---")
		st.sidebar.markdown("### Case")
		case_sensitive = st.sidebar.checkbox("Make search case sensitive")
		
		if bool(case_sensitive) == True:
			ignore_case = False
		else:
			ignore_case = True
		
		st.sidebar.markdown("---")
		st.sidebar.markdown(_messages.message_generate_table)
		if st.sidebar.button("KWIC"):
			if session.get('target_path') == None:
				st.write(_warnings.warning_11, unsafe_allow_html=True)
			elif node_word == "":
				st.write(_warnings.warning_14, unsafe_allow_html=True)
			elif node_word.count(" ") > 0:
				st.write(_warnings.warning_15, unsafe_allow_html=True)
			elif len(node_word) > 15:
				st.write(_warnings.warning_16, unsafe_allow_html=True)
			else:
				tp = _handlers.load_corpus_session('target', session)
				with st.spinner('Processing KWIC...'):
					kwic_df = _analysis.kwic_st(tp, node_word=node_word, search_type=search_type, ignore_case=ignore_case)
				if bool(isinstance(kwic_df, pd.DataFrame)) == True:
					_handlers.save_table(kwic_df, 'kwic')
					_handlers.update_session('kwic', True)
					st.experimental_rerun()
				else:
					st.markdown(":warning: Your search didn't return any matches.")
		st.sidebar.markdown("---")
		
if __name__ == "__main__":
    main()