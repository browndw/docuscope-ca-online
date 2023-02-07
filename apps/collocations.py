import streamlit as st
import docuscospacy.corpus_analysis as ds
import pandas as pd
import st_aggrid
from tmtoolkit.tokenseq import pmi, pmi2, pmi3

import base64
from io import BytesIO

import categories
import states as _states

CATEGORY = categories.COLLOCATION
TITLE = "Collocates"
KEY_SORT = 7

def main():
	# check states to prevent unlikely error
	for key, value in _states.STATES.items():
		if key not in st.session_state:
			setattr(st.session_state, key, value)

	if bool(isinstance(st.session_state.collocations, pd.DataFrame)) == True:
		
		df = st.session_state.collocations
		
		col1, col2 = st.columns([1,1])
		with col1:
			st.markdown(f"""##### Target corpus information:
			
			Number of tokens in corpus: {st.session_state.tokens}\n    Number of word tokens in corpus: {st.session_state.words}\n    Number of documents in corpus: {st.session_state.ndocs}
			""")
		with col2:
			am = str(st.session_state.stat).upper()
			span = str(st.session_state.span_l) + 'L - ' + str(st.session_state.span_r) + 'R'
			st.markdown(f"""##### Collocate information:
			
			Association measure: {am}\n    Span: {span}\n    Node word: {st.session_state.node}
			""")
			
		gb = st_aggrid.GridOptionsBuilder.from_dataframe(df)
		gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=100) #Add pagination
		gb.configure_default_column(filter="agTextColumnFilter")
		gb.configure_column("Token", filter="agTextColumnFilter", headerCheckboxSelection = True, headerCheckboxSelectionFilteredOnly = True)
		gb.configure_column("MI", type=["numericColumn","numberColumnFilter","customNumericFormat"], precision=3)
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
			st.markdown("""
						The 'Freq Span' columns refers to the collocate's frequency within the given window,
						while 'Freq Total' refers to its overall frequency in the corpus. 
						Note that is possible for a collocate to have a *higher* frequency within a window, than a total frequency.\n
						The 'MI' column refers to the association measure selected when the table was generated
						(one of NPMI, PMI2, PMI3, or PMI).
						""")
	
		selected = grid_response['selected_rows'] 
		if selected:
			st.write('Selected rows')
			df = pd.DataFrame(selected).drop('_selectedRowNodeInfo', axis=1)
			st.dataframe(df)
		
		with st.sidebar.expander("Filtering and saving"):
			st.markdown("""
						Filters can be accessed by clicking on the three that appear while hovering over a column header.
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
					linko= f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="collocations.xlsx">Download Excel file</a>'
					st.markdown(linko, unsafe_allow_html=True)
		
		st.sidebar.markdown("---")
		st.sidebar.markdown("### Generate new table")
		st.sidebar.markdown("""
							Click the button to reset the collocations table.
							""")
		if st.sidebar.button("Create New Collocations Table"):
			st.session_state.collocations = ''
			st.experimental_rerun()
		st.sidebar.markdown("---")
				
	else:
		st.sidebar.markdown("### Node word")
		st.sidebar.markdown("""Use the text field to enter a node word and other desired options.
					Once a node word has been entered and options selected, click the button below to generate a table of collocates.
					""")				
		node_word = st.sidebar.text_input("Node word:")
							
		st.sidebar.markdown("---")
		with st.sidebar.expander("Span explanation"):
			st.markdown("""
						Associations are calculated by counting the observed frequency within a
						span around a node word and comparing that to the frequency that we would expect
						given its overall frequency in a corpus.
						
						You could adjust the span if, for example, you wanted look at
						the subjects of a verb. For that, you would want to search only the left of
						the node word, setting the right span to 0. For verb object, you would want to
						do the opposite. There could be cases when you want a narrower window or a
						wider one.
						""")
		st.sidebar.markdown("### Span")
		to_left = st.sidebar.slider("Choose a span to the left of the node word:", 0, 9, (4))
		to_right = st.sidebar.slider("Choose a span to the right of the node word:", 0, 9, (4))
		
		st.sidebar.markdown("---")
		with st.sidebar.expander("Statistics explanation"):
			st.markdown("""
						The most common statistic for measuring token associations is Pointwise Mutual Information (PMI),
						first developed by [Church and Hanks](https://aclanthology.org/J90-1003/). One potentially problematic
						characteristic of PMI is that it rewards (or generates high scores) for low frequency tokens.
						
						This can be handled by filtering for minimum frequencies and MI scores. Alternatively,
						[other measures have been proposed, which you can select from here.](https://en.wikipedia.org/wiki/Pointwise_mutual_information)
						""")

		st.sidebar.markdown("### Association measure")			
		stat_mode = st.sidebar.radio("Select a statistic:", ("NPMI", "PMI 2", "PMI 3", "PMI"), horizontal=True)
		
		if stat_mode == "NPMI":
			stat_mode = "npmi"
		elif stat_mode == "PMI 2":
			stat_mode = "pmi2"
		elif stat_mode == "PMI 3":
			stat_mode = "pmi3"
		elif stat_mode == "PMI":
			stat_mode = "pmi"
		
		st.sidebar.markdown("---")
		with st.sidebar.expander("Anchor tag explanation"):
			st.markdown("""
						You can choose to 'anchor' at token to a specific tag.
						For example, if you wanted to disambiguate 'can' as a noun (e.g., 'can of soda')
						from 'can' as a modal verb, you could 'anchor' the node word to a part-of-speech
						tag (like 'Noun', 'Verb' or more specifically 'VM').
						
						For most cases, choosing an 'anchor' tag isn't necessary.
						""")
		st.sidebar.markdown("### Anchor tag")
		tag_radio = st.sidebar.radio("Select tagset for node word:", ("No Tag", "Parts-of-Speech", "DocuScope"), horizontal=True)
		if tag_radio == 'Parts-of-Speech':
			tag_type = st.sidebar.radio("Select from general or specific tags", ("General", "Specific"), horizontal=True)
			if tag_type == 'General':
				node_tag = st.sidebar.selectbox("Select tag:", ("Noun", "Verb", "Adjective", "Adverb"))
				if node_tag == "Noun":
					node_tag = "NN"
				elif node_tag == "Verb":
					node_tag = "VV"
				elif node_tag == "Adjective":
					node_tag = "JJ"
				elif node_tag == "Adverb":
					node_tag = "R"
			else:
				node_tag = st.sidebar.selectbox("Select tag:", (st.session_state.tags_pos))
			ignore_tags = False
			count_by = 'pos'
		elif tag_radio == 'DocuScope':
			node_tag = st.sidebar.selectbox("Select tag:", (st.session_state.tags_ds))
			ignore_tags = False
			count_by = 'ds'
		else:
			node_tag = None
			ignore_tags = True
			count_by = 'pos'
		
		st.sidebar.markdown("---")
		st.sidebar.markdown("### Generate collocates")
		st.sidebar.markdown("""
			Use the button to generate a table of collocates from your corpus.
			Note that n-grams take a little longer to process than other kinds of tables.
			""")
		if st.sidebar.button("Collocations"):
			if st.session_state.corpus == "":
				st.write(":neutral_face: It doesn't look like you've loaded a corpus yet.")
			elif node_word == "":
				st.write(":warning: It doesn't look like you've entered a node word. Be sure to hit return after typing a word into the field.")
			elif node_word.count(" ") > 0:
				st.write(":warning: Your node word shouldn't contain any spaces.")
			elif len(node_word) > 15:
				st.write(":warning: Your node word contains too many characters. Try something shorter.")
			else:
				tp = st.session_state.corpus
				with st.spinner('Processing collocates...'):
					df = ds.coll_table(tp, node_word=node_word, node_tag=node_tag, l_span=to_left, r_span=to_right, statistic=stat_mode, tag_ignore=ignore_tags, count_by=count_by)
				if len(df.index) > 0:
					st.session_state.collocations = df
					st.session_state.node = node_word
					st.session_state.stat = stat_mode
					st.session_state.span_l = to_left
					st.session_state.span_r = to_right
					st.experimental_rerun()
				else:
					st.markdown(":warning: Your search didn't return any matches.")
	
		st.sidebar.markdown("---")
		
		st.markdown("""
		Once a corpus has been loaded, use the tool to:
		
		1. Select the node word
		2. Select the span or window
		3. Select the association measure
		4. Select a tag to "anchor" the node word (which is usually 'None')
		5. Generate a table of collocates
		""")
	
		
if __name__ == "__main__":
    main()