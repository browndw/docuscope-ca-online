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

import altair as alt
import pandas as pd
import pathlib
import polars as pl
import st_aggrid
import streamlit as st

import categories as _categories
import states as _states
from utilities import analysis_functions as _analysis
from utilities import handlers_database as _handlers
from utilities import messages as _messages
from utilities import warnings as _warnings

CATEGORY = _categories.KEYNESS
TITLE = "Compare Corpora"
KEY_SORT = 5

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

	if session.get('keyness_table')[0] == True:
	
		_handlers.load_widget_state(pathlib.Path(__file__).stem, user_session_id)
		metadata_target = _handlers.load_metadata('target', con)
		metadata_reference = _handlers.load_metadata('reference', con)

		st.sidebar.markdown("### Comparison")	
		table_radio = st.sidebar.radio("Select the keyness table to display:", ("Tokens", "Tags Only"), key = _handlers.persist("kt_radio1", pathlib.Path(__file__).stem, user_session_id), horizontal=True)
		if table_radio == 'Tokens':
			st.sidebar.markdown("---")
			st.sidebar.markdown("### Tagset")
			tag_radio_tokens = st.sidebar.radio("Select tags to display:", ("Parts-of-Speech", "DocuScope"), key = _handlers.persist("kt_radio2", pathlib.Path(__file__).stem,  user_session_id), horizontal=True)
	
			if tag_radio_tokens == 'Parts-of-Speech':
				df = con.table("kw_pos", database="target").to_pandas()
			else:
				df = con.table("kw_ds", database="target").to_pandas()
			
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
			gb.configure_column("RF_Ref", type=["numericColumn","numberColumnFilter","customNumericFormat"], precision=2)
			gb.configure_column("Range_Ref", type=["numericColumn","numberColumnFilter"], valueFormatter="(data.Range_Ref).toFixed(1)+'%'")
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

			if selected is not None:
				df = pd.DataFrame(selected)
				n_selected = len(df.index)
				st.markdown(f"""##### Selected rows:
				
				Number of selected tokens: {n_selected}
				""")
			
			st.sidebar.markdown("---")

			with st.sidebar.expander("Filtering and saving"):
				st.markdown(_messages.message_filters)

			with st.sidebar:
				st.markdown(_messages.message_download)
				download_file = _handlers.convert_to_excel(df)

				st.download_button(
					label="Download to Excel",
					data=download_file,
					file_name="keywords_tokens.xlsx",
						mime="application/vnd.ms-excel",
						)
			st.sidebar.markdown("---")
	
		else:
			st.sidebar.markdown("### Tagset")
			tag_radio_tags = st.sidebar.radio("Select tags to display:", ("Parts-of-Speech", "DocuScope"), key = _handlers.persist("kt_radio3", pathlib.Path(__file__).stem, user_session_id), horizontal=True)
	
			if tag_radio_tags == 'Parts-of-Speech':
				df = con.table("kt_pos", database="target").to_pandas()
			else:
				df = con.table("kt_ds", database="target").to_polars().filter(pl.col("Tag") != "Untagged").to_pandas()
	
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
			gb.configure_column("RF_Ref", type=["numericColumn","numberColumnFilter","customNumericFormat"], precision=2)
			gb.configure_column("Range_Ref", type=["numericColumn","numberColumnFilter"], valueFormatter="(data.Range_Ref).toFixed(1)+'%'")
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

			if selected is not None:
				df = pd.DataFrame(selected)
				n_selected = len(df.index)
				st.markdown(f"""##### Selected rows:
				
				Number of selected tokens: {n_selected}
				""")

				
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
							color=alt.Color('Corpus:N', sort=order, scale=alt.Scale(scheme='category10')),
							row=alt.Row('Tag', title=None, header=alt.Header(orient='left', labelAngle=0, labelAlign='left'), sort=alt.SortField(field='Mean', order='descending')),
							tooltip=[
								alt.Tooltip('Tag'),
								alt.Tooltip('RF:Q', title="RF", format='.2')
							]).configure_facet(spacing=2.5).configure_legend(orient='top')				
				
				st.markdown(_messages.message_disable_full, unsafe_allow_html=True)
				st.altair_chart(base, use_container_width=True)
		
			with st.sidebar.expander("Filtering and saving"):
				st.markdown(_messages.message_filters)
			
			with st.sidebar:
				st.markdown(_messages.message_download)
				download_file = _handlers.convert_to_excel(df)

				st.download_button(
					label="Download to Excel",
					data=download_file,
					file_name="keywords_tags.xlsx",
						mime="application/vnd.ms-excel",
						)
			st.sidebar.markdown("---")
	
	else:
		
		st.markdown(_messages.message_keyness)
		
		st.sidebar.markdown(_messages.message_generate_table)
		
		if st.sidebar.button("Keyness Table"):
			if session.get('has_target')[0] == False:
				st.markdown(_warnings.warning_11, unsafe_allow_html=True)
			elif session.get('has_reference')[0] == False:
				st.markdown(_warnings.warning_17, unsafe_allow_html=True)
			else:
				with st.sidebar:
					with st.spinner('Generating keywords...'):
					
						wc_tar_pos = con.table("ft_pos", database="target").to_pyarrow_batches(chunk_size=5000)
						wc_tar_pos = pl.from_arrow(wc_tar_pos)
						wc_tar_ds = con.table("ft_ds", database="target").to_pyarrow_batches(chunk_size=5000)
						wc_tar_ds = pl.from_arrow(wc_tar_ds)
						tc_tar_pos = con.table("tt_pos", database="target").to_polars()
						tc_tar_ds = con.table("tt_ds", database="target").to_polars()

						wc_ref_pos = con.table("ft_pos", database="reference").to_pyarrow_batches(chunk_size=5000)
						wc_ref_pos = pl.from_arrow(wc_ref_pos)
						wc_ref_ds = con.table("ft_ds", database="reference").to_pyarrow_batches(chunk_size=5000)
						wc_ref_ds = pl.from_arrow(wc_ref_ds)
						tc_ref_pos = con.table("tt_pos", database="reference").to_polars()
						tc_ref_ds = con.table("tt_ds", database="reference").to_polars()
						
						kw_pos = _analysis.keyness_pl(wc_tar_pos, wc_ref_pos)
						kw_ds  = _analysis.keyness_pl(wc_tar_ds, wc_ref_ds)
						kt_pos = _analysis.keyness_pl(tc_tar_pos, tc_ref_pos, tags_only=True)
						kt_ds  = _analysis.keyness_pl(tc_tar_ds, tc_ref_ds, tags_only=True)

						con.create_table("kw_pos", obj=kw_pos, database="target", overwrite=True)
						con.create_table("kw_ds", obj=kw_ds, database="target", overwrite=True)
						con.create_table("kt_pos", obj=kt_pos, database="target", overwrite=True)
						con.create_table("kt_ds", obj=kt_ds, database="target", overwrite=True)

						_handlers.update_session('keyness_table', True, con)
						st.success('Keywords generated!')
						st.rerun()
		
		st.sidebar.markdown("---")		

if __name__ == "__main__":
    main()