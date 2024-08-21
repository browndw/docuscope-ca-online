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

import polars as pl
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
		df = pl.from_arrow(df)
	
		st.markdown(_messages.message_target_info(metadata_target))
		
		col1, col2 = st.columns(2)

		with col1:
			if df.height == 0 or df is None:
				cats_1 = []
			elif df.height > 0:
				cats_1 = sorted(df.get_column("Tag_1").unique().to_list())

			filter_tag_1 = st.multiselect("Select tags to filter in position 1:", (cats_1))
			if len(filter_tag_1) > 0:
				df = df.filter(pl.col("Tag_1").is_in(filter_tag_1))

			if "Tag_3" in df.columns:
				cats_3 = sorted(df.get_column("Tag_3").unique().to_list())
				filter_tag_3 = st.multiselect("Select tags to filter in position 3:", (cats_3))
				if len(filter_tag_3) > 0:
					df = df.filter(pl.col("Tag_3").is_in(filter_tag_3))
		
		with col2:
			if df.height == 0 or df is None:
				cats_2 = []
			elif df.height > 0:
				cats_2 = sorted(df.get_column("Tag_2").unique().to_list())

			filter_tag_2 = st.multiselect("Select tags to filter in position 2:", (cats_2))
			if len(filter_tag_2) > 0:
				df = df.filter(pl.col("Tag_2").is_in(filter_tag_2))

			if "Tag_4" in df.columns:
				cats_4 = sorted(df.get_column("Tag_4").unique().to_list())
				filter_tag_4 = st.multiselect("Select tags to filter in position 4:", (cats_4))
				if len(filter_tag_4) > 0:
					df = df.filter(pl.col("Tag_4").is_in(filter_tag_4))


		st.dataframe(df, hide_index=True, 
				column_config={
					"Range": st.column_config.NumberColumn(format="%.2f %%"),
					"RF": st.column_config.NumberColumn(format="%.2f")}
		)
		
		download_table = st.sidebar.toggle("Download to Excel?")
		if download_table == True:
			with st.sidebar:
				st.sidebar.markdown(_messages.message_download)
				download_file = _handlers.convert_to_excel(df.to_pandas())

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