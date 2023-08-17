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

modules = ['analysis', 'categories', 'handlers', 'messages', 'states', 'warnings', 'streamlit', 'altair', 'docuscospacy', 'pandas', 'st_aggrid']
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
TITLE = "Compare Corpus Parts"
KEY_SORT = 6

def main():

	
	user_session = st.runtime.scriptrunner.script_run_context.get_script_run_ctx()
	user_session_id = user_session.session_id

	session = _handlers.load_session(user_session_id)

	if bool(session['keyness_parts']) == True:
	
		_handlers.load_widget_state(pathlib.Path(__file__).stem, user_session_id)
		metadata_target = _handlers.load_metadata('target', user_session_id)

		st.sidebar.markdown("### Comparison")	
		table_radio = st.sidebar.radio("Select the keyness table to display:", ("Tokens", "Tags Only"), key = _handlers.persist("cp_radio1", pathlib.Path(__file__).stem, user_session_id), horizontal=True)
		if table_radio == 'Tokens':
			st.sidebar.markdown("---")
			st.sidebar.markdown("### Tagset")
			tag_radio_tokens = st.sidebar.radio("Select tags to display:", ("Parts-of-Speech", "DocuScope"), key = _handlers.persist("cp_radio2", pathlib.Path(__file__).stem, user_session_id), horizontal=True)
	
			if tag_radio_tokens == 'Parts-of-Speech':
				df = _handlers.load_table('kw_pos_cp', user_session_id)
			else:
				df = _handlers.load_table('kw_ds_cp', user_session_id)
	
			col1, col2 = st.columns([1,1])
			with col1:
				st.markdown(_messages.message_target_parts(session['keyness_parts']))
			with col2:
				st.markdown(_messages.message_reference_parts(session['keyness_parts']))
	
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
			st.sidebar.markdown("### Generate new table")
			st.sidebar.markdown("""
							Click the button to reset the keyness table.
							""")			
			if st.sidebar.button("Compare New Categories"):
				_handlers.clear_table('kw_pos_cp', user_session_id)
				_handlers.clear_table('kw_ds_cp', user_session_id)
				_handlers.clear_table('kt_pos_cp', user_session_id)
				_handlers.clear_table('kt_ds_cp', user_session_id)
				_handlers.update_session('keyness_parts', dict(), user_session_id)
				st.experimental_rerun()
			st.sidebar.markdown("---")
			
		else:
			
			st.sidebar.markdown("### Tagset")
			tag_radio_tags = st.sidebar.radio("Select tags to display:", ("Parts-of-Speech", "DocuScope"), key = _handlers.persist("cp_radio3", pathlib.Path(__file__).stem, user_session_id), horizontal=True)
	
			if tag_radio_tags == 'Parts-of-Speech':
				df = _handlers.load_table('kt_pos_cp', user_session_id)
			else:
				df = _handlers.load_table('kt_ds_cp', user_session_id)
	
			col1, col2 = st.columns([1,1])
			with col1:
				st.markdown(_messages.message_target_parts(session['keyness_parts']))
			with col2:
				st.markdown(_messages.message_reference_parts(session['keyness_parts']))
		
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
			if st.sidebar.button("Plot resutls"):
				df_plot = df[["Tag", "RF", "RF Ref"]]
				df_plot["Mean"] = df_plot.mean(numeric_only=True, axis=1)
				df_plot.rename(columns={"Tag": "Tag", "Mean": "Mean", "RF": "Target", "RF Ref": "Reference"}, inplace = True)
				df_plot = pd.melt(df_plot, id_vars=['Tag', 'Mean'],var_name='Corpus', value_name='RF')
				df_plot.sort_values(by=["Mean", "Corpus"], ascending=[True, True], inplace=True)
					
				order = ['Target', 'Reference']
				base = alt.Chart(df_plot, height={"step": 12}).mark_bar(size=10).encode(
								x=alt.X('RF', title='Frequency (per 100 tokens)'),
								y=alt.Y('Corpus:N', title=None, sort=order, axis=alt.Axis(labels=False, ticks=False)),
								color=alt.Color('Corpus:N', sort=order),
								row=alt.Row('Tag', title=None, header=alt.Header(orient='left', labelAngle=0, labelAlign='left'), sort=alt.SortField(field='Mean', order='descending')),
								tooltip=[
									alt.Tooltip('Tag'),
									alt.Tooltip('RF:Q', title="RF", format='.2')
								]).configure_facet(spacing=2.5).configure_legend(orient='top')

				st.markdown(_messages.message_disable_full, unsafe_allow_html=True)
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
			st.sidebar.markdown(_messages.message_reset_table)			
			if st.sidebar.button("Compare New Categories"):
				_handlers.clear_table('kw_pos_cp', user_session_id)
				_handlers.clear_table('kw_ds_cp', user_session_id)
				_handlers.clear_table('kt_pos_cp', user_session_id)
				_handlers.clear_table('kt_ds_cp', user_session_id)
				_handlers.update_session('keyness_parts', dict(), user_session_id)
				st.experimental_rerun()
			st.sidebar.markdown("---")
	
	else:		
		
		st.markdown(_messages.message_corpus_parts)
		
		try:
			metadata_target = _handlers.load_metadata('target', user_session_id)
		except:
			metadata_target = {}
		
		st.sidebar.markdown("### Select categories to compare")
		st.sidebar.markdown("After **target** and **reference** categories have been selected, click the button to generate a keyness table.")
		
		if session['has_meta'] == True:
			st.sidebar.markdown('#### Target corpus categories:')
			st.session_state[user_session_id]['tar'] = st.sidebar.multiselect("Select target categories:", (sorted(set(metadata_target['doccats']))), on_change = _handlers.update_tar(user_session_id), key=f"tar_{user_session_id}")
		else:
			st.sidebar.multiselect("Select reference categories:", (['No categories to select']), key='empty_tar')
		
		if session['has_meta'] == True:
			st.sidebar.markdown('#### Reference corpus categories:')
			st.session_state[user_session_id]['ref'] = st.sidebar.multiselect("Select reference categories:", (sorted(set(metadata_target['doccats']))), on_change = _handlers.update_ref(user_session_id), key=f"ref_{user_session_id}")
		else:
			st.sidebar.multiselect("Select reference categories:", (['No categories to select']), key='empty_ref')
		st.sidebar.markdown("---")
		
		st.sidebar.markdown(_messages.message_generate_table)
		if st.sidebar.button("Keyness Table of Corpus Parts"):
			if session.get('target_path') == None:
				st.markdown(_warnings.warning_11, unsafe_allow_html=True)
			elif session['has_meta'] == False:
				st.markdown(_warnings.warning_21, unsafe_allow_html=True)
			elif len(list(st.session_state[user_session_id]['tar'])) == 0 or len(list(st.session_state[user_session_id]['ref'])) == 0:
				st.markdown(_warnings.warning_22, unsafe_allow_html=True)
			else:
				with st.sidebar:
					with st.spinner('Generating keywords...'):
						tp = _handlers.load_corpus_session('target', session, user_session_id)
						tar_list = [item + "_" for item in list(st.session_state[user_session_id]['tar'])]
						ref_list = [item + "_" for item in list(st.session_state[user_session_id]['ref'])]
						
						tar_docs = {key: value for key, value in tp.items() if key.startswith(tuple(tar_list))}
						ref_docs = {key: value for key, value in tp.items() if key.startswith(tuple(ref_list))}
						
						#get target counts
						tar_docs, ref_docs, tar_words, ref_words, tar_tokens, ref_tokens, tar_ndocs, ref_ndocs = _analysis.split_corpus(tp, tar_docs, ref_docs)
					
						wc_tar_pos = ds.frequency_table(tar_docs, tar_words)
						wc_tar_ds = ds.frequency_table(tar_docs, tar_tokens, count_by='ds')
						tc_tar_pos = ds.tags_table(tar_docs, tar_words)
						tc_tar_ds = ds.tags_table(tar_docs, tar_tokens, count_by='ds')
		
						wc_ref_pos = ds.frequency_table(ref_docs, ref_words)
						wc_ref_ds = ds.frequency_table(ref_docs, ref_tokens, count_by='ds')
						tc_ref_pos = ds.tags_table(ref_docs, ref_words)
						tc_ref_ds = ds.tags_table(ref_docs, ref_tokens, count_by='ds')
					
						kw_pos_cp = ds.keyness_table(wc_tar_pos, wc_ref_pos)
						kw_ds_cp = ds.keyness_table(wc_tar_ds, wc_ref_ds)
						kt_pos_cp = ds.keyness_table(tc_tar_pos, tc_ref_pos, tags_only=True)
						kt_ds_cp = ds.keyness_table(tc_tar_ds, tc_ref_ds, tags_only=True)
					
					_handlers.save_table(kw_pos_cp, 'kw_pos_cp', user_session_id)
					_handlers.save_table(kw_ds_cp, 'kw_ds_cp', user_session_id)
					_handlers.save_table(kt_pos_cp, 'kt_pos_cp', user_session_id)
					_handlers.save_table(kt_ds_cp, 'kt_ds_cp', user_session_id)
					_handlers.update_keyness_parts(tar_words, ref_words, tar_tokens, ref_tokens, tar_ndocs, ref_ndocs, st.session_state[user_session_id]['tar'], st.session_state[user_session_id]['ref'], user_session_id)
					
					st.success('Keywords generated!')
					st.experimental_rerun()
		
		st.sidebar.markdown("---")
		
if __name__ == "__main__":
    main()		