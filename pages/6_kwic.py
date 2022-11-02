import streamlit as st

# NLP Pkgs
import docuscospacy.corpus_analysis as ds
from tmtoolkit.tokenseq import index_windows_around_matches

import pandas as pd
import numpy as np

import base64
from io import BytesIO
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode

def kwic_st(tok, node_word, search_type, ignore_case=True):
	kwic = []
	for i in range(0,len(tok)):
		tpf = list(tok.values())[i]
		doc_id = list(tok.keys())[i]
		# create a boolean vector for node word
		if bool(ignore_case) == True and search_type == "fixed":
			v = [t[0].strip().lower() == node_word.lower() for t in tpf]
		elif bool(ignore_case) == False and search_type == "fixed":
			v = [t[0].strip() == node_word for t in tpf]
		elif bool(ignore_case) == True and search_type == "starts_with":
			v = [t[0].strip().lower().startswith(node_word.lower()) for t in tpf]
		elif bool(ignore_case) == False and search_type == "starts_with":
			v = [t[0].strip().startswith(node_word) for t in tpf]
		elif bool(ignore_case) == True and search_type == "ends_with":
			v = [t[0].strip().lower().endswith(node_word.lower()) for t in tpf]
		elif bool(ignore_case) == False and search_type == "ends_with":
			v = [t[0].strip().endswith(node_word) for t in tpf]
		elif bool(ignore_case) == True and search_type == "contains":
			v = [node_word.lower() in t[0].strip().lower() for t in tpf]
		elif bool(ignore_case) == False and search_type == "contains":
			v = [node_word in t[0].strip() for t in tpf]

		if sum(v) > 0:
			# get indices within window around the node
			idx = list(index_windows_around_matches(np.array(v), left=7, right=7, flatten=False))
			node_idx = [i for i, x in enumerate(v) if x == True]
			start_idx = [min(x) for x in idx]
			end_idx = [max(x) for x in idx]
			in_span = []
			for i in range(len(node_idx)):
				pre_node = "".join([t[0] for t in tpf[start_idx[i]:node_idx[i]]]).strip()
				post_node = "".join([t[0] for t in tpf[node_idx[i]+1:end_idx[i]]]).strip()
				node = tpf[node_idx[i]][0]
				in_span.append((doc_id, pre_node, node, post_node))
			kwic.append(in_span)
	kwic = [x for xs in kwic for x in xs]
	if len(kwic) > 0:
		df = pd.DataFrame(kwic)
		df.columns =['Doc ID', 'Pre-Node', 'Node', 'Post-Node']
	else:
		df = "Empty"
	return(df)

st.title("Create key words in context (KWIC)")

if 'corpus' not in st.session_state:
	st.session_state.corpus = ''

if 'kwic' not in st.session_state:
	st.session_state.kwic = ''

if bool(isinstance(st.session_state.kwic, pd.DataFrame)) == True:
	
	df = st.session_state.kwic	
	
	gb = GridOptionsBuilder.from_dataframe(df)
	gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=100) #Add pagination
	gb.configure_default_column(filter="agTextColumnFilter")
	gb.configure_column("Doc ID", filter="agTextColumnFilter", headerCheckboxSelection = True, headerCheckboxSelectionFilteredOnly = True)
	gb.configure_column("Pre-Node", type="rightAligned")
	gb.configure_selection('multiple', use_checkbox=True, groupSelectsChildren="Group checkbox select children") #Enable multi-row selection
	gb.configure_grid_options(sideBar = {"toolPanels": ['filters']})
	go = gb.build()

	grid_response = AgGrid(
		df,
		gridOptions=go,
		data_return_mode='FILTERED_AND_SORTED', 
		update_mode='MODEL_CHANGED', 
		columns_auto_size_mode='FIT_CONTENTS',
		theme='alpine',
		height=500, 
		width='100%',
		reload_data=False
		)
	
	with st.expander("See explanation"):
		st.write("""
				The chart above shows some numbers I picked for you.
				I rolled actual dice for these, so they're *guaranteed* to
				be random.
		""")

	selected = grid_response['selected_rows'] 
	if selected:
		st.write('Selected rows')
		df = pd.DataFrame(selected).drop('_selectedRowNodeInfo', axis=1)
		st.dataframe(df)
	
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
		if st.button("Create a new KWIC table"):
			st.session_state.kwic = ''
			st.experimental_rerun()


		
else:
	st.write("Use the button to generate a KWIC table for a node word.")
	
	search_mode = st.radio("Select search type:", ("Fixed", "Starts with", "Ends with", "Contains"), horizontal=True)
	
	if search_mode == "Fixed":
		search_type = "fixed"
	elif search_mode == "Starts with":
		search_type = "starts_with"
	elif search_mode == "Ends with":
		search_type = "ends_with"
	else:
		search_type = "contains"
	
	case_sensitive = st.checkbox("Make search case sensitive")
	
	if bool(case_sensitive) == True:
		ignore_case = False
	else:
		ignore_case = True
	
	node_word = st.text_input("Node word")
	if st.session_state.kwic == 'Empty':
		st.markdown(":warning: Your search didn't return any matches.")
	if node_word.count(" ") > 0:
		st.write("Your node word shouldn't contain any spaces.")
	elif len(node_word) > 15:
		st.write("Your node word contains too many characters. Try something shorter.")
	elif node_word != "":
		if st.button("KWIC"):
			if st.session_state.corpus == "":
				st.write(":neutral_face: It doesn't look like you've loaded a corpus yet.")
			else:
				tp = st.session_state.corpus
				df = kwic_st(tp, node_word=node_word, search_type=search_type, ignore_case=ignore_case)
				st.session_state.kwic = df
				st.experimental_rerun()

