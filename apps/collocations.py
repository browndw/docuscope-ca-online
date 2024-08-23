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

CATEGORY = _categories.COLLOCATION
TITLE = "Collocates"
KEY_SORT = 7

def main():

	user_session = st.runtime.scriptrunner.script_run_context.get_script_run_ctx()
	user_session_id = user_session.session_id

	if user_session_id not in st.session_state:
		st.session_state[user_session_id] = {}
	try:
		session = pl.DataFrame.to_dict(st.session_state[user_session_id]["session"], as_series=False)
	except:
		_handlers.init_session(user_session_id)
		session = pl.DataFrame.to_dict(st.session_state[user_session_id]["session"], as_series=False)

	if session.get('collocations')[0] == True:
		
		metadata_target = _handlers.load_metadata('target', user_session_id)

		df = st.session_state[user_session_id]["target"]["collocations"]
		
		col1, col2 = st.columns([1,1])
		with col1:
			st.markdown(_messages.message_target_info(metadata_target))
		with col2:
			st.markdown(_messages.message_collocation_info(metadata_target.get('collocations')[0]['temp']))
	
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
		
		download_table = st.sidebar.toggle("Download to Excel?")
		if download_table == True:
			with st.sidebar:
				st.markdown(_messages.message_download)
				download_file = _handlers.convert_to_excel(df.to_pandas())

				st.download_button(
					label="Download to Excel",
					data=download_file,
					file_name="collocations.xlsx",
						mime="application/vnd.ms-excel",
						)
		st.sidebar.markdown("---")

		st.sidebar.markdown(_messages.message_reset_table)
		
		if st.sidebar.button("Create New Collocations Table"):
			if "collocations" not in st.session_state[user_session_id]["target"]:
				st.session_state[user_session_id]["target"]["collocations"] = {}
			st.session_state[user_session_id]["target"]["collocations"] = {}
			_handlers.update_session('collocations', False, user_session_id)
			st.rerun()
		st.sidebar.markdown("---")
				
	else:

		st.markdown(_messages.message_collocations)

		if session.get("has_target")[0] == True:
			metadata_target = _handlers.load_metadata('target', user_session_id)

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
		stat_mode = st.sidebar.radio("Select a statistic:",
							   ["NPMI", "PMI 2", "PMI 3", "PMI"], 
							   horizontal=True)
		
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
				node_tag = st.sidebar.selectbox("Select tag:", ("Noun Common", "Verb Lex", "Adjective", "Adverb"))
				if node_tag == "Noun Common":
					node_tag = "NN"
				elif node_tag == "Verb Lex":
					node_tag = "VV"
				elif node_tag == "Adjective":
					node_tag = "JJ"
				elif node_tag == "Adverb":
					node_tag = "R"
			else:
				if session.get('has_target')[0] == False:
					node_tag = st.sidebar.selectbox('Choose a tag:', ['No tags currently loaded'])
				else:
					node_tag = st.sidebar.selectbox('Choose a tag:', metadata_target.get('tags_pos')[0]['tags'])
			ignore_tags = False
			count_by = 'pos'
		elif tag_radio == 'DocuScope':
			if session.get('has_target')[0] == False:
				node_tag = st.sidebar.selectbox('Choose a tag:', ['No tags currently loaded'])
			else:
				node_tag = st.sidebar.selectbox('Choose a tag:', metadata_target.get('tags_ds')[0]['tags'])
				ignore_tags = False
				count_by = 'ds'
		else:
			node_tag = None
			ignore_tags = False
			count_by = 'pos'
		
		st.sidebar.markdown("---")
		st.sidebar.markdown(_messages.message_generate_table)
		if st.sidebar.button("Collocations"):
			if session.get('has_target')[0] == False:
				st.markdown(_warnings.warning_11, unsafe_allow_html=True)
			elif node_word == "":
				st.markdown(_warnings.warning_14, unsafe_allow_html=True)
			elif node_word.count(" ") > 0:
				st.markdown(_warnings.warning_15, unsafe_allow_html=True)
			elif len(node_word) > 15:
				st.markdown(_warnings.warning_16, unsafe_allow_html=True)
			else:
				with st.sidebar:
					with st.spinner('Processing collocates...'):
						tok_pl = st.session_state[user_session_id]["target"]["ds_tokens"]

						coll_df = _analysis.collocations_pl(tok_pl, node_word=node_word, node_tag=node_tag, preceding=to_left, following=to_right, statistic=stat_mode, count_by=count_by)
				
				if coll_df.is_empty():
					st.markdown(_warnings.warning_12, unsafe_allow_html=True)
					
				else:
					if "collocations" not in st.session_state[user_session_id]["target"]:
						st.session_state[user_session_id]["target"]["collocations"] = {}
					st.session_state[user_session_id]["target"]["collocations"] = coll_df
					_handlers.update_session('collocations', True, user_session_id)
					_handlers.update_metadata('target', key='collocations', value=[node_word, stat_mode, str(to_left), str(to_right)], session_id=user_session_id)
					st.rerun()

	
		st.sidebar.markdown("---")
		
if __name__ == "__main__":
    main()