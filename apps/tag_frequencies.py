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

modules = ['categories', 'handlers', 'messages', 'states', 'warnings', 'altair', 'streamlit', 'docuscospacy', 'pandas', 'st_aggrid']
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
TITLE = "Tag Frequencies"
KEY_SORT = 3

def main():

	user_session = st.runtime.scriptrunner.script_run_context.get_script_run_ctx()
	user_session_id = user_session.session_id

	session = _handlers.load_session(user_session_id)

	if session.get('tags_table') == True:
	
		_handlers.load_widget_state(pathlib.Path(__file__).stem, user_session_id)
		metadata_target = _handlers.load_metadata('target', user_session_id)

		st.sidebar.markdown("### Tagset")
		tag_radio = st.sidebar.radio("Select tags to display:", ("Parts-of-Speech", "DocuScope"), key = _handlers.persist("tt_radio", pathlib.Path(__file__).stem, user_session_id), horizontal=True)
	
		if tag_radio == 'Parts-of-Speech':
			df = _handlers.load_table('tt_pos', user_session_id)
		else:
			df = _handlers.load_table('tt_ds', user_session_id)
		
		st.markdown(_messages.message_target_info(metadata_target))
		
		gb = st_aggrid.GridOptionsBuilder.from_dataframe(df)
		gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=100) #Add pagination
		gb.configure_column("Tag", filter="agTextColumnFilter", headerCheckboxSelection = True, headerCheckboxSelectionFilteredOnly = True)
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
			st.markdown(_messages.message_columns_tags)
		
		selected = grid_response['selected_rows'] 
		if selected:
			st.write('Selected rows')
			df = pd.DataFrame(selected).drop('_selectedRowNodeInfo', axis=1)
			st.dataframe(df)
	
		st.sidebar.markdown("---")
		st.sidebar.markdown(_messages.message_generate_plot)
		
		if st.sidebar.button("Plot Frequencies"):			
			base = alt.Chart(df, height={"step": 24}).mark_bar(size=12).encode(
					x=alt.X('RF', title='Frequency (per 100 tokens)'), 
					y=alt.Y('Tag', sort='-x', title=None, axis=alt.Axis(labelLimit=200)),
					tooltip=[
					alt.Tooltip('Tag'),
					alt.Tooltip('RF', title="RF:", format='.2')
					])
						
			st.altair_chart(base, use_container_width=True)
	
		st.sidebar.markdown("---")
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
					linko= f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="tag_frequencies.xlsx">Download Excel file</a>'
					st.markdown(linko, unsafe_allow_html=True)
		st.sidebar.markdown("---")
	
	else:
		
		st.markdown(_messages.message_tables)
		
		st.sidebar.markdown(_messages.message_generate_table)

		if st.sidebar.button("Tags Table"):
			if session.get('target_path') == None:
				st.markdown(_warnings.warning_11, unsafe_allow_html=True)
			
			else:
				with st.sidebar:
					with st.spinner('Processing frequencies...'):
						tp = _handlers.load_corpus_session('target', session, user_session_id)
						metadata_target = _handlers.load_metadata('target', user_session_id)
						tc_pos = ds.tags_table(tp, metadata_target.get('words'))
						tc_ds = ds.tags_table(tp, metadata_target.get('tokens'), count_by='ds')
					_handlers.save_table(tc_pos, 'tt_pos', user_session_id)
					_handlers.save_table(tc_ds, 'tt_ds', user_session_id)
					_handlers.update_session('tags_table', True, user_session_id)
					st.experimental_rerun()

		st.sidebar.markdown("---")

if __name__ == "__main__":
    main()