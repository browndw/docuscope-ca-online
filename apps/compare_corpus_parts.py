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
import streamlit as st

import categories as _categories
import states as _states
from utilities import analysis_functions as _analysis
from utilities import handlers_database as _handlers
from utilities import messages as _messages
from utilities import warnings as _warnings

CATEGORY = _categories.KEYNESS
TITLE = "Compare Corpus Parts"
KEY_SORT = 6

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

	session = pl.DataFrame.to_dict(con.table("session").to_polars(), as_series=False)

	if session.get('keyness_parts')[0] == True:
	
		_handlers.load_widget_state(pathlib.Path(__file__).stem, user_session_id)
		metadata_target = _handlers.load_metadata('target', con)

		col1, col2 = st.columns([1,1])
		with col1:
			st.markdown(_messages.message_target_parts(metadata_target.get('keyness_parts')[0]['temp']))
		with col2:
			st.markdown(_messages.message_reference_parts(metadata_target.get('keyness_parts')[0]['temp']))
		
		st.markdown("Showing keywords that reach significance at *p* < 0.01")

		st.sidebar.markdown("### Comparison")	
		table_radio = st.sidebar.radio("Select the keyness table to display:", ("Tokens", "Tags Only"), key = _handlers.persist("cp_radio1", pathlib.Path(__file__).stem, user_session_id), horizontal=True)

		st.sidebar.markdown("---")
		if table_radio == 'Tokens':
			tag_radio_tokens = st.sidebar.radio("Select tags to display:", ("Parts-of-Speech", "DocuScope"), key = _handlers.persist("cp_radio2", pathlib.Path(__file__).stem, user_session_id), horizontal=True)
			if tag_radio_tokens == 'Parts-of-Speech':
				tag_type = st.sidebar.radio("Select from general or specific tags", ("General", "Specific"), horizontal=True)			
				if tag_type == 'General':
					df = con.table("kw_pos_cp", database="target").to_pyarrow_batches(chunk_size=5000)
					df = pl.from_arrow(df)
					df = _analysis.freq_simplify_pl(df)
				else:
					df = con.table("kw_pos_cp", database="target").to_pyarrow_batches(chunk_size=5000)
					df = pl.from_arrow(df)
			else:			
				df = con.table("kw_ds_cp", database="target").to_pyarrow_batches(chunk_size=5000)
				df = pl.from_arrow(df)
		
			if df.height == 0 or df is None:
				cats = []
			elif df.height > 0:
				cats = sorted(df.get_column("Tag").unique().to_list())

			filter_vals = st.multiselect("Select tags to filter:", (cats))
			if len(filter_vals) > 0:
				df = df.filter(pl.col("Tag").is_in(filter_vals))

			st.dataframe(df, hide_index=True, 
					column_config={
						"Range": st.column_config.NumberColumn(format="%.2f %%"),
						"Range_Ref": st.column_config.NumberColumn(format="%.2f %%"),
						"RF": st.column_config.NumberColumn(format="%.2f"),
						"RF_Ref": st.column_config.NumberColumn(format="%.2f")}
			)
		
			with st.expander("Column explanation"):
				st.markdown(_messages.message_columns_keyness)

			st.sidebar.markdown("---")
						
			download_table = st.sidebar.toggle("Download to Excel?")
			if download_table == True:
				with st.sidebar:
					st.markdown(_messages.message_download)
					download_file = _handlers.convert_to_excel(df.to_pandas())

					st.download_button(
						label="Download to Excel",
						data=download_file,
						file_name="keywords_tokens.xlsx",
							mime="application/vnd.ms-excel",
							)

			st.sidebar.markdown("---")

			st.sidebar.markdown("### Generate new table")
			st.sidebar.markdown("""
							Click the button to reset the keyness table.
							""")			
			if st.sidebar.button("Compare New Categories"):
				try:
					con.drop_table("kw_pos_cp", database="target")
					con.drop_table("kw_ds_cp", database="target")
					con.drop_table("kt_pos_cp", database="target")
					con.drop_table("kt_ds_cp", database="target")
				except:
					pass
				_handlers.update_session('keyness_parts', False, con)
				st.rerun()
			st.sidebar.markdown("---")
			
		else:
			
			st.sidebar.markdown("### Tagset")
			tag_radio_tags = st.sidebar.radio("Select tags to display:", ("Parts-of-Speech", "DocuScope"), key = _handlers.persist("cp_radio3", pathlib.Path(__file__).stem, user_session_id), horizontal=True)
	
			if tag_radio_tags == 'Parts-of-Speech':
				df = con.table("kt_pos_cp", database="target").to_pyarrow_batches(chunk_size=5000)
				df = pl.from_arrow(df).filter(pl.col("Tag") != "FU")
			else:
				df = con.table("kt_ds_cp", database="target").to_pyarrow_batches(chunk_size=5000)
				df = pl.from_arrow(df).filter(pl.col("Tag") != "Untagged")
	
			if df.height == 0 or df is None:
				cats = []
			elif df.height > 0:
				cats = sorted(df.get_column("Tag").unique().to_list())

			filter_vals = st.multiselect("Select tags to filter:", (cats))
			if len(filter_vals) > 0:
				df = df.filter(pl.col("Tag").is_in(filter_vals))

			st.dataframe(df, hide_index=True, 
					column_config={
						"Range": st.column_config.NumberColumn(format="%.2f %%"),
						"Range_Ref": st.column_config.NumberColumn(format="%.2f %%"),
						"RF": st.column_config.NumberColumn(format="%.2f"),
						"RF_Ref": st.column_config.NumberColumn(format="%.2f")}
			)
	
			with st.expander("Column explanation"):
				st.markdown(_messages.message_columns_keyness)
					
			st.sidebar.markdown("---")
			
			st.sidebar.markdown(_messages.message_generate_plot)
			if st.sidebar.button("Plot resutls"):
				df_plot = df.to_pandas()
				df_plot = df_plot[["Tag", "RF", "RF_Ref"]]
				df_plot["Mean"] = df_plot.mean(numeric_only=True, axis=1)
				df_plot.rename(columns={"Tag": "Tag", "Mean": "Mean", "RF": "Target", "RF_Ref": "Reference"}, inplace = True)
				df_plot = pd.melt(df_plot, id_vars=['Tag', 'Mean'],var_name='Corpus', value_name='RF')
				df_plot.sort_values(by=["Mean", "Corpus"], ascending=[True, True], inplace=True)
					
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
	
			st.sidebar.markdown("---")
			
			download_table = st.sidebar.toggle("Download to Excel?")
			if download_table == True:
				with st.sidebar:
					st.markdown(_messages.message_download)
					download_file = _handlers.convert_to_excel(df.to_pandas())

					st.download_button(
						label="Download to Excel",
						data=download_file,
						file_name="keywords_tags.xlsx",
							mime="application/vnd.ms-excel",
							)
			
			st.sidebar.markdown("---")
			
			st.sidebar.markdown(_messages.message_reset_table)			
			if st.sidebar.button("Compare New Categories"):
				try:
					con.drop_table("kw_pos_cp", database="target")
					con.drop_table("kw_ds_cp", database="target")
					con.drop_table("kt_pos_cp", database="target")
					con.drop_table("kt_ds_cp", database="target")
				except:
					pass
				_handlers.update_session('keyness_parts', False, con)
				st.rerun()
			st.sidebar.markdown("---")
	
	else:		
		
		st.markdown(_messages.message_corpus_parts)
		
		st.sidebar.markdown("### Select categories to compare")
		st.sidebar.markdown("After **target** and **reference** categories have been selected, click the button to generate a keyness table.")
		
		if session.get('has_meta')[0] == True:
			metadata_target = _handlers.load_metadata('target', con)
			st.sidebar.markdown('#### Target corpus categories:')
			st.session_state[user_session_id]['tar'] = st.sidebar.multiselect("Select target categories:", (sorted(set(metadata_target.get('doccats')[0]['cats']))), _handlers.update_tar(user_session_id), key=f"tar_{user_session_id}")
		else:
			st.sidebar.multiselect("Select reference categories:", (['No categories to select']), key='empty_tar')
		
		if session.get('has_meta')[0] == True:
			metadata_target = _handlers.load_metadata('target', con)
			st.sidebar.markdown('#### Reference corpus categories:')
			st.session_state[user_session_id]['ref'] = st.sidebar.multiselect("Select reference categories:", (sorted(set(metadata_target.get('doccats')[0]['cats']))), _handlers.update_ref(user_session_id), key=f"ref_{user_session_id}")
		else:
			st.sidebar.multiselect("Select reference categories:", (['No categories to select']), key='empty_ref')
		
		st.sidebar.markdown("---")
		
		st.sidebar.markdown(_messages.message_generate_table)
		if st.sidebar.button("Keyness Table of Corpus Parts"):
			if session.get('has_target')[0] == False:
				st.markdown(_warnings.warning_11, unsafe_allow_html=True)
			elif session.get('has_meta')[0] == False:
				st.markdown(_warnings.warning_21, unsafe_allow_html=True)
			elif len(list(st.session_state[user_session_id]['tar'])) == 0 or len(list(st.session_state[user_session_id]['ref'])) == 0:
				st.markdown(_warnings.warning_22, unsafe_allow_html=True)
			else:
				with st.sidebar:
					with st.spinner('Generating keywords...'):
						tar_list = list(st.session_state[user_session_id]['tar'])
						ref_list = list(st.session_state[user_session_id]['ref'])

						tok_pl = con.table("ds_tokens", database="target").to_pyarrow_batches(chunk_size=5000)
						tok_pl = pl.from_arrow(tok_pl)

						tar_pl = _analysis.subset_pl(tok_pl, tar_list)
						ref_pl = _analysis.subset_pl(tok_pl, ref_list)
											
						wc_tar_pos, wc_tar_ds = _analysis.frequency_tables_pl(tar_pl)
						tc_tar_pos, tc_tar_ds = _analysis.tag_tables_pl(tar_pl)
		
						wc_ref_pos, wc_ref_ds = _analysis.frequency_tables_pl(ref_pl)
						tc_ref_pos, tc_ref_ds = _analysis.tag_tables_pl(ref_pl)

						kw_pos_cp = _analysis.keyness_pl(wc_tar_pos, wc_ref_pos)
						kw_ds_cp  = _analysis.keyness_pl(wc_tar_ds, wc_ref_ds)
						kt_pos_cp = _analysis.keyness_pl(tc_tar_pos, tc_ref_pos, tags_only=True)
						kt_ds_cp  = _analysis.keyness_pl(tc_tar_ds, tc_ref_ds, tags_only=True)

						tar_tokens_pos = tar_pl.group_by(["doc_id", "pos_id", "pos_tag"]).agg(pl.col("token").str.concat("")).filter(pl.col("pos_tag") != "Y").height
						ref_tokens_pos = ref_pl.group_by(["doc_id", "pos_id", "pos_tag"]).agg(pl.col("token").str.concat("")).filter(pl.col("pos_tag") != "Y").height
						tar_tokens_ds = tar_pl.group_by(["doc_id", "ds_id", "ds_tag"]).agg(pl.col("token").str.concat("")).filter(~(pl.col("token").str.contains("^[[[:punct:]] ]+$") & pl.col("ds_tag").str.contains("Untagged"))).height
						ref_tokens_ds = ref_pl.group_by(["doc_id", "ds_id", "ds_tag"]).agg(pl.col("token").str.concat("")).filter(~(pl.col("token").str.contains("^[[[:punct:]] ]+$") & pl.col("ds_tag").str.contains("Untagged"))).height
						tar_ndocs = tar_pl.get_column("doc_id").unique().len()
						ref_ndocs = ref_pl.get_column("doc_id").unique().len()
					
					con.create_table("kw_pos_cp", obj=kw_pos_cp, database="target", overwrite=True)
					con.create_table("kw_ds_cp", obj=kw_ds_cp, database="target", overwrite=True)
					con.create_table("kt_pos_cp", obj=kt_pos_cp, database="target", overwrite=True)
					con.create_table("kt_ds_cp", obj=kt_ds_cp, database="target", overwrite=True)

					_handlers.update_session('keyness_parts', True, con)
					_handlers.update_metadata('target', key='keyness_parts', value=[tar_list, ref_list, str(tar_tokens_pos), str(ref_tokens_pos), str(tar_tokens_ds), str(ref_tokens_ds), str(tar_ndocs), str(ref_ndocs)], ibis_conn=con)
			
					st.success('Keywords generated!')
					st.rerun()
		
		st.sidebar.markdown("---")
		
if __name__ == "__main__":
    main()		