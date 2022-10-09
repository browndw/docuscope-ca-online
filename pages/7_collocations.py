import streamlit as st

# NLP Pkgs
import docuscospacy.corpus_analysis as ds
from tmtoolkit.tokenseq import pmi, pmi2, pmi3

import pandas as pd

import base64
from io import BytesIO
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode

st.title("Create tables of collocations")

if 'corpus' not in st.session_state:
	st.session_state.corpus = ''

if 'collocations' not in st.session_state:
	st.session_state.collocations = ''

# a method for preserving button selection on page interactions
# with quick clicking it can lag
if 'count_5' not in st.session_state:
	st.session_state.count_5 = 0

def increment_counter():
	st.session_state.count_5 += 1

if st.session_state.count_5 % 2 == 0:
    idx = 0
else:
    idx = 1

if bool(isinstance(st.session_state.collocations, pd.DataFrame)) == True:
	
	df = st.session_state.collocations
	
	reload_data = False
	if st.button('Reset filters'):
		grid_response = df.copy()
		reload_data = True
	
	gb = GridOptionsBuilder.from_dataframe(df)
	gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=100) #Add pagination
	gb.configure_default_column(filter="agTextColumnFilter")
	gb.configure_column("Pre-Node", type="rightAligned")
	
	gb.configure_side_bar(columns_panel=False) #Add a sidebar
	go = gb.build()

	grid_response = AgGrid(
		df,
		gridOptions=go,
		columns_auto_size_mode='FIT_CONTENTS',
		theme='alpine', #Add theme color to the table
		height=500, 
		width='100%',
		reload_data=reload_data
		)
	
	with st.expander("See explanation"):
		st.write("""
				The chart above shows some numbers I picked for you.
				I rolled actual dice for these, so they're *guaranteed* to
				be random.
		""")
	
	col1, col2 = st.columns([1,1])
	
	with col1:
		if st.button("Download"):
			with st.spinner('Creating download link...'):
				towrite = BytesIO()
				downloaded_file = df.to_excel(towrite, encoding='utf-8', index=False, header=True)
				towrite.seek(0)  # reset pointer
				b64 = base64.b64encode(towrite.read()).decode()  # some strings
				st.success('Link generated!')
				linko= f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="kwic.xlsx">Download Excel file</a>'
				st.markdown(linko, unsafe_allow_html=True)
	
	with col2:
		if st.button("Create a new collocations table"):
			st.session_state.collocations = ''
			st.experimental_rerun()


		
else:
	st.write("Use the button to generate a table of collocations from a node word.")
				
	node_word = st.text_input("Node word")
	
	tag_radio = st.radio("Select tagset for node word:", ("No Tag", "Parts-of-Speech", "DocuScope"), horizontal=True)

	if tag_radio == 'Parts-of-Speech':
		tag_type = st.radio("Select from general or specific tags", ("General", "Specific"), horizontal=True)
		if tag_type == 'General':
			node_tag = st.selectbox("Select tag:", ("Noun", "Verb", "Adjective", "Adverb"))
			if node_tag == "Noun":
				node_tag = "NN"
			elif node_tag == "Verb":
				node_tag = "V"
			elif node_tag == "Adjective":
				node_tag = "J"
			elif node_tag == "Adverb":
				node_tag = "R"
		else:
			node_tag = st.selectbox("Select tag:", (st.session_state.tags_pos))
		ignore_tags = False
	elif tag_radio == 'DocuScope':
		node_tag = st.selectbox("Select tag:", (st.session_state.tags_ds))
		ignore_tags = False
	else:
		node_tag = None
		ignore_tags = True
	
	to_left = st.slider("Choose a span to the left of the node word:", 0, 9, (4))
	to_right = st.slider("Choose a span to the right of the node word:", 0, 9, (4))
		
	stat_mode = st.radio("Select a statistic:", ("NPMI", "PMI 2", "PMI 3", "PMI"), horizontal=True)
	
	if stat_mode == "NPMI":
		stat_mode = "npmi"
	elif stat_mode == "PMI 2":
		stat_mode = "pmi2"
	elif stat_mode == "PMI 3":
		stat_mode = "pmi3"
	elif stat_mode == "PMI":
		stat_mode = "pmi"


	if st.session_state.collocations == 'Empty':
		st.markdown(":warning: Your search didn't return any matches.")
	if node_word.count(" ") > 0:
		st.write("Your node word shouldn't contain any spaces.")
	elif len(node_word) > 15:
		st.write("Your node word contains too many characters. Try something shorter.")
	elif node_word != "":
		if st.button("Collocations"):
			#st.write(token_tuple)
			#wc = load_data()
			if st.session_state.corpus == "":
				st.write("It doesn't look like you've loaded a corpus yet.")
			else:
				tp = st.session_state.corpus
				df = ds.coll_table(tp, node_word=node_word, node_tag=node_tag, l_span=to_left, r_span=to_right, statistic=stat_mode, tag_ignore=ignore_tags)
				st.session_state.collocations = df
				st.experimental_rerun()
