import streamlit as st
import docuscospacy.corpus_analysis as ds
import pandas as pd
import numpy as np
import st_aggrid
from tmtoolkit.tokenseq import index_windows_around_matches

import base64
from io import BytesIO

import categories
import states as _states

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
		df = ''
	return(df)

CATEGORY = categories.OTHER
TITLE = "KWIC Tables"
KEY_SORT = 8

def main():
	# check states to prevent unlikely error
	for key, value in _states.STATES.items():
		if key not in st.session_state:
			setattr(st.session_state, key, value)

	if bool(isinstance(st.session_state.kwic, pd.DataFrame)) == True:
		
		df = st.session_state.kwic	
		
		gb = st_aggrid.GridOptionsBuilder.from_dataframe(df)
		gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=100) #Add pagination
		gb.configure_default_column(filter="agTextColumnFilter")
		gb.configure_column("Doc ID", filter="agTextColumnFilter", headerCheckboxSelection = True, headerCheckboxSelectionFilteredOnly = True)
		gb.configure_column("Pre-Node", type="rightAligned")
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
		
		selected = grid_response['selected_rows'] 
		if selected:
			st.write('Selected rows')
			df = pd.DataFrame(selected).drop('_selectedRowNodeInfo', axis=1)
			st.dataframe(df)
		
		with st.sidebar.expander("Filtering and saving"):
			st.markdown("""
					Filters can be accessed by clicking on the three lines that appear while hovering over a column header.
					For text columns, you can filter by 'Equals', 'Starts with', 'Ends with', and 'Contains'.\n
					Rows can be selected before or after filtering using the checkboxes.
					(The checkbox in the header will select/deselect all rows.)\n
					If rows are selected and appear in new table below the main one,
					those selected rows will be available for download in an Excel file.
					If no rows are selected, the full table will be processed for downloading after clicking the Download button.
					""")
		
		st.sidebar.markdown("### Download data")
		st.sidebar.markdown("""
							Click the button to genenerate a download link.
							You can use the checkboxes to download selected rows.
							With no rows selected, the entire table will be prepared for download.
							""")
		
		if st.sidebar.button("Download"):
			with st.sidebar:
				with st.spinner('Creating download link...'):
					towrite = BytesIO()
					downloaded_file = df.to_excel(towrite, encoding='utf-8', index=False, header=True)
					towrite.seek(0)  # reset pointer
					b64 = base64.b64encode(towrite.read()).decode()  # some strings
					st.success('Link generated!')
					linko= f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="kwic.xlsx">Download Excel file</a>'
					st.markdown(linko, unsafe_allow_html=True)
		
		st.sidebar.markdown("---")
		st.sidebar.markdown("### Generate new table")
		st.sidebar.markdown("""
							Click the button to reset the KWIC table.
							""")
													
		if st.sidebar.button("Create New KWIC Table"):
				st.session_state.kwic = ''
				st.experimental_rerun()
		st.sidebar.markdown("---")
			
	else:
		st.sidebar.markdown("### Node word")
		st.sidebar.markdown("""Use the text field to enter a node word and other desired options.
					Once a node word has been entered and options selected, click the button below to generate a KWIC table.
					""")				
		node_word = st.sidebar.text_input("Node word")
		
		st.sidebar.markdown("---")
		st.sidebar.markdown("### Search mode")
		search_mode = st.sidebar.radio("Select search type:", ("Fixed", "Starts with", "Ends with", "Contains"), horizontal=True)
		
		if search_mode == "Fixed":
			search_type = "fixed"
		elif search_mode == "Starts with":
			search_type = "starts_with"
		elif search_mode == "Ends with":
			search_type = "ends_with"
		else:
			search_type = "contains"
		
		st.sidebar.markdown("---")
		st.sidebar.markdown("### Case")
		case_sensitive = st.sidebar.checkbox("Make search case sensitive")
		
		if bool(case_sensitive) == True:
			ignore_case = False
		else:
			ignore_case = True
		
		st.sidebar.markdown("---")
		st.sidebar.markdown("### Generate KWIC")
		st.sidebar.markdown("""
			Use the button to generate a KWIC table from your corpus.
			""")
		if st.sidebar.button("KWIC"):
			if st.session_state.corpus == "":
				st.write(":neutral_face: It doesn't look like you've loaded a corpus yet.")
			elif node_word == "":
				st.write(":warning: It doesn't look like you've entered a node word. Be sure to hit return after typing a word into the field.")
			elif node_word.count(" ") > 0:
				st.write("Your node word shouldn't contain any spaces.")
			elif len(node_word) > 15:
				st.write("Your node word contains too many characters. Try something shorter.")
			else:
				tp = st.session_state.corpus
				with st.spinner('Processing KWIC...'):
					df = kwic_st(tp, node_word=node_word, search_type=search_type, ignore_case=ignore_case)
				if bool(isinstance(df, pd.DataFrame)) == True:
					st.session_state.kwic = df
					st.experimental_rerun()
				else:
					st.markdown(":warning: Your search didn't return any matches.")
		st.sidebar.markdown("---")
		
		st.markdown("""
		Use this tool to generate Key Words in Context for a word or part of a word (like the ending *tion*).
		Note that wildcard characters are **not needed**.
		""")

if __name__ == "__main__":
    main()