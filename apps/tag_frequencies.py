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
import pathlib
import polars as pl
import streamlit as st

import categories as _categories
import states as _states
from utilities import handlers_database as _handlers
from utilities import analysis_functions as _analysis
from utilities import messages as _messages
from utilities import warnings as _warnings

HERE = pathlib.Path(__file__).parents[1].resolve()
TEMP_DIR = HERE.joinpath("_temp")

CATEGORY = _categories.FREQUENCY
TITLE = "Tag Frequencies"
KEY_SORT = 3

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

	if session.get('tags_table')[0] == True:
	
		_handlers.load_widget_state(pathlib.Path(__file__).stem, user_session_id)
		metadata_target = _handlers.load_metadata('target', con)

		st.sidebar.markdown("### Tagset")
		tag_radio = st.sidebar.radio("Select tags to display:", ("Parts-of-Speech", "DocuScope"), key = _handlers.persist("tt_radio", pathlib.Path(__file__).stem, user_session_id), horizontal=True)
		if tag_radio == 'Parts-of-Speech':
			tag_type = st.sidebar.radio("Select from general or specific tags", ("General", "Specific"), horizontal=True)			
			if tag_type == 'General':
				df = con.table("dtm_pos", database="target").to_pyarrow_batches(chunk_size=5000)
				df = pl.from_arrow(df)
				df = _analysis.dtm_simplify_pl(df)
				df = _analysis.tags_simplify_pl(df)
			else:
				df = con.table("tt_pos", database="target").to_pyarrow_batches(chunk_size=5000)
				df = pl.from_arrow(df).filter(pl.col("Tag") != "FU")
		else:			
			df = con.table("tt_ds", database="target").to_pyarrow_batches(chunk_size=5000)
			df = pl.from_arrow(df).filter(pl.col("Tag") != "Untagged")
	
		st.markdown(_messages.message_target_info(metadata_target))

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
					"RF": st.column_config.NumberColumn(format="%.2f")}
		)
	
		st.sidebar.markdown("---")
		st.sidebar.markdown(_messages.message_generate_plot)
		
		if st.sidebar.button("Plot Frequencies"):			
			base = alt.Chart(df.to_pandas(), height={"step": 24}).mark_bar(size=12).encode(
					alt.Color(scale=alt.Scale(scheme='category10')),
					x=alt.X('RF', title='Frequency (per 100 tokens)'), 
					y=alt.Y('Tag', sort='-x', title=None, axis=alt.Axis(labelLimit=200)),
					tooltip=[
					alt.Tooltip('Tag'),
					alt.Tooltip('RF', title="RF:", format='.2')
					])
						
			st.markdown(_messages.message_disable_full, unsafe_allow_html=True)
			st.altair_chart(base, use_container_width=True)
	
		st.sidebar.markdown("---")

		download_table = st.sidebar.toggle("Download to Excel?")
		if download_table == True:	
			with st.sidebar:
				st.sidebar.markdown(_messages.message_download)
				download_file = _handlers.convert_to_excel(df.to_pandas())

				st.download_button(
					label="Download to Excel",
					data=download_file,
					file_name="tag_frequencies.xlsx",
						mime="application/vnd.ms-excel",
						)

		st.sidebar.markdown("---")
	
	else:
		
		st.markdown(_messages.message_tables)
		
		st.sidebar.markdown(_messages.message_generate_table)

		if st.sidebar.button("Tags Table"):
			if session.get('has_target')[0] == False:
				st.markdown(_warnings.warning_11, unsafe_allow_html=True)
			
			else:
				with st.sidebar:
					with st.spinner('Processing frequencies...'):
						_handlers.update_session('tags_table', True, con)
					st.rerun()

		st.sidebar.markdown("---")

if __name__ == "__main__":
    main()