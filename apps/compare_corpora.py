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

CATEGORY = _categories.KEYNESS
TITLE = "Compare Corpora"
KEY_SORT = 5

def main():
	
	session = _handlers.load_session()	

	if session.get('keyness_table') == True:
	
		_handlers.load_widget_state(pathlib.Path(__file__).stem)
		metadata_target = _handlers.load_metadata('target')
		metadata_reference = _handlers.load_metadata('reference')

		st.sidebar.markdown("### Comparison")	
		table_radio = st.sidebar.radio("Select the keyness table to display:", ("Tokens", "Tags Only"), key = _handlers.persist("kt_radio1", pathlib.Path(__file__).stem), horizontal=True)
		if table_radio == 'Tokens':
			st.sidebar.markdown("---")
			st.sidebar.markdown("### Tagset")
			tag_radio_tokens = st.sidebar.radio("Select tags to display:", ("Parts-of-Speech", "DocuScope"), key = _handlers.persist("kt_radio2", pathlib.Path(__file__).stem), horizontal=True)
	
			if tag_radio_tokens == 'Parts-of-Speech':
				df = _handlers.load_table('kw_pos')
			else:
				df = _handlers.load_table('kw_ds')
			
			col1, col2 = st.columns([1,1])
			with col1:
				st.markdown(_messages.message_target_info(metadata_target))
			with col2:
				st.markdown(_messages.message_reference_info(metadata_reference))	
		
			gb = st_aggrid.GridOptionsBuilder.from_dataframe(df)
			gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=100) #Add pagination
			gb.configure_column("Token", filter="agTextColumnFilter", headerCheckboxSelection = True, headerCheckboxSelectionFilteredOnly = True)
			gb.configure_column("Tag", filter="agTextColumnFilter")
			gb.configure_column("LL", type=["numericColumn","numberColumnFilter","customNumericFormat"], precision=2)
			gb.configure_column("LR", type=["numericColumn","numberColumnFilter","customNumericFormat"], precision=3)
			gb.configure_column("PV", type=["numericColumn","numberColumnFilter","customNumericFormat"], precision=4)
			gb.configure_column("RF", type=["numericColumn","numberColumnFilter","customNumericFormat"], precision=2)
			gb.configure_column("Range", type=["numericColumn","numberColumnFilter"], valueFormatter="(data.Range).toFixed(1)+'%'")
			gb.configure_column("RF Ref", type=["numericColumn","numberColumnFilter","customNumericFormat"], precision=2)
			gb.configure_column("Range Ref", type=["numericColumn","numberColumnFilter"], valueFormatter="(data.Range).toFixed(1)+'%'")
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
				st.markdown(_messages.message_columns_keyness)
				
			selected = grid_response['selected_rows'] 
			if selected:
				st.write('Selected rows')
				df = pd.DataFrame(selected).drop('_selectedRowNodeInfo', axis=1)
				st.dataframe(df)
			
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
			st.sidebar.markdown("### Tagset")
			tag_radio_tags = st.sidebar.radio("Select tags to display:", ("Parts-of-Speech", "DocuScope"), key = _handlers.persist("kt_radio3", pathlib.Path(__file__).stem), horizontal=True)
	
			if tag_radio_tags == 'Parts-of-Speech':
				df = _handlers.load_table('kt_pos')
			else:
				df = _handlers.load_table('kt_ds')
	
			col1, col2 = st.columns([1,1])
			with col1:
				st.markdown(_messages.message_target_info(metadata_target))
			with col2:
				st.markdown(_messages.message_reference_info(metadata_reference))			
		
			gb = st_aggrid.GridOptionsBuilder.from_dataframe(df)
			gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=100) #Add pagination
			gb.configure_column("Tag", filter="agTextColumnFilter", headerCheckboxSelection = True, headerCheckboxSelectionFilteredOnly = True)
			gb.configure_column("LL", type=["numericColumn","numberColumnFilter","customNumericFormat"], precision=2)
			gb.configure_column("LR", type=["numericColumn","numberColumnFilter","customNumericFormat"], precision=3)
			gb.configure_column("PV", type=["numericColumn","numberColumnFilter","customNumericFormat"], precision=4)
			gb.configure_column("RF", type=["numericColumn","numberColumnFilter","customNumericFormat"], precision=2)
			gb.configure_column("Range", type=["numericColumn","numberColumnFilter"], valueFormatter="(data.Range).toFixed(1)+'%'")
			gb.configure_column("RF Ref", type=["numericColumn","numberColumnFilter","customNumericFormat"], precision=2)
			gb.configure_column("Range Ref", type=["numericColumn","numberColumnFilter"], valueFormatter="(data.Range).toFixed(1)+'%'")
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
				st.markdown(_messages.message_columns_keyness)
		
			selected = grid_response['selected_rows'] 
			if selected:
				st.write('Selected rows')
				df = pd.DataFrame(selected).drop('_selectedRowNodeInfo', axis=1)
				st.dataframe(df)
				
			st.sidebar.markdown("---")
			st.sidebar.markdown(_messages.message_generate_plot)
			
			if st.sidebar.button('Plot resutls'):
				df_plot = df[['Tag', 'RF', 'RF Ref']]
				df_plot['Mean'] = df_plot.mean(numeric_only=True, axis=1)
				df_plot.rename(columns={'Tag': 'Tag', 'Mean': 'Mean', 'RF': 'Target', 'RF Ref': 'Reference'}, inplace = True)
				df_plot = pd.melt(df_plot, id_vars=['Tag', 'Mean'],var_name='Corpus', value_name='RF')
				df_plot.sort_values(by=['Mean', 'Corpus'], ascending=[True, True], inplace=True)
					
				order = ['Target', 'Reference']
				base = alt.Chart(df_plot, height={"step": 12}).mark_bar(size=10).encode(
							x=alt.X('RF', title='Frequency (per 100 tokens)'),
							y=alt.Y('Corpus:N', title=None, sort=order, axis=alt.Axis(labels=False, ticks=False)),
							color=alt.Color('Corpus:N', sort=order),
							row=alt.Row('Tag', title=None, header=alt.Header(orient='left', labelAngle=0, labelAlign='left'), sort=alt.SortField(field='Mean', order='descending')),
							tooltip=[
							alt.Tooltip('RF:Q', title="Per 100 Tokens", format='.2')
							]).configure_facet(spacing=0.5).configure_legend(orient='top')				
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
		
		st.markdown(_messages.message_keyness)
		
		st.sidebar.markdown(_messages.message_generate_table)
		
		if st.sidebar.button("Keyness Table"):
			if session.get('target_path') == None:
				st.markdown(_warnings.warning_11, unsafe_allow_html=True)
			elif session.get('reference_path') == None:
				st.markdown(_warnings.warning_17, unsafe_allow_html=True)
			else:
				with st.spinner('Generating keywords...'):
					tp_ref = _handlers.load_corpus_session('reference', session)
					metadata_reference = _handlers.load_metadata('reference')
					wc_ref_pos = ds.corpus_analysis.frequency_table(tp_ref, metadata_reference.get('words'))
					wc_ref_ds  = ds.corpus_analysis.frequency_table(tp_ref, metadata_reference.get('tokens'), count_by='ds')
					tc_ref_pos = ds.corpus_analysis.tags_table(tp_ref, metadata_reference.get('words'))
					tc_ref_ds  = ds.corpus_analysis.tags_table(tp_ref, metadata_reference.get('tokens'), count_by='ds')
					
					if session.get('tags_table') == False:
						tp = _handlers.load_corpus_session('target', session)
						metadata_target = _handlers.load_metadata('target')
						tc_pos = ds.corpus_analysis.tags_table(tp, metadata_target.get('words'))
						tc_ds  = ds.corpus_analysis.tags_table(tp, metadata_target.get('tokens'), count_by='ds')
						_handlers.save_table(tc_pos, 'tt_pos')
						_handlers.save_table(tc_ds, 'tt_ds')
						_handlers.update_session('tags_table', True)
					
					if session.get('freq_table') == False:
						tp = _handlers.load_corpus_session('target', session)
						metadata_target = _handlers.load_metadata('target')
						wc_pos = ds.corpus_analysis.frequency_table(tp, metadata_target.get('words'))
						wc_ds  = ds.corpus_analysis.frequency_table(tp, metadata_target.get('tokens'), count_by='ds')
						_handlers.save_table(wc_pos, 'ft_pos')
						_handlers.save_table(wc_ds, 'ft_ds')
						_handlers.update_session('freq_table', True)
				
					kw_pos = ds.corpus_analysis.keyness_table(_handlers.load_table('ft_pos'), wc_ref_pos)
					kw_ds  = ds.corpus_analysis.keyness_table(_handlers.load_table('ft_ds'), wc_ref_ds)
					kt_pos = ds.corpus_analysis.keyness_table(_handlers.load_table('tt_pos'), tc_ref_pos, tags_only=True)
					kt_ds  = ds.corpus_analysis.keyness_table(_handlers.load_table('tt_ds'), tc_ref_ds, tags_only=True)
					_handlers.save_table(kw_pos, 'kw_pos')
					_handlers.save_table(kw_ds, 'kw_ds')
					_handlers.save_table(kt_pos, 'kt_pos')
					_handlers.save_table(kt_ds, 'kt_ds')
					_handlers.update_session('keyness_table', True)
					st.success('Keywords generated!')
					st.experimental_rerun()
		

if __name__ == "__main__":
    main()