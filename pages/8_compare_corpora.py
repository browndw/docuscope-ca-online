import streamlit as st

# NLP Pkgs
import docuscospacy.corpus_analysis as ds

import pandas as pd
import plotly.express as px

import base64
from io import BytesIO
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode

st.title("Create a keyness table")

if 'corpus' not in st.session_state:
	st.session_state.corpus = ''

if 'reference' not in st.session_state:
	st.session_state.reference = ''

if 'ft_pos' not in st.session_state:
	st.session_state.ft_pos = ''

if 'ft_ds' not in st.session_state:
	st.session_state.ft_ds = ''

if 'tt_pos' not in st.session_state:
	st.session_state.tt_pos = ''

if 'tt_ds' not in st.session_state:
	st.session_state.tt_ds = ''

if 'kw_pos' not in st.session_state:
	st.session_state.kw_pos = ''

if 'kw_ds' not in st.session_state:
	st.session_state.kw_ds = ''

# a method for preserving button selection on page interactions
# with quick clicking it can lag
if 'count_5' not in st.session_state:
	st.session_state.count_5 = 0

def increment_counter_5():
	st.session_state.count_5 += 1

if st.session_state.count_5 % 2 == 0:
    idx_5 = 0
else:
    idx_5 = 1

if 'count_6' not in st.session_state:
	st.session_state.count_6 = 0

def increment_counter_6():
	st.session_state.count_6 += 1

if st.session_state.count_6 % 2 == 0:
    idx_6 = 0
else:
    idx_6 = 1

if 'count_7' not in st.session_state:
	st.session_state.count_7 = 0

def increment_counter_7():
	st.session_state.count_7 += 1

if st.session_state.count_7 % 2 == 0:
    idx_7 = 0
else:
    idx_7 = 1

if bool(isinstance(st.session_state.kw_pos, pd.DataFrame)) == True:

	table_radio = st.radio("Select the keyness table to display:", ("Tokens", "Tags Only"), index=idx_5, on_change=increment_counter_5, horizontal=True)
	if table_radio == 'Tokens':
		tag_radio_tokens = st.radio("Select tags to display:", ("Parts-of-Speech", "DocuScope"), index=idx_6, on_change=increment_counter_6, horizontal=True)

		if tag_radio_tokens == 'Parts-of-Speech':
			df = st.session_state.kw_pos
		else:
			df = st.session_state.kw_ds
	
		gb = GridOptionsBuilder.from_dataframe(df)
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
	
		with st.expander("Column explanation"):
			st.write("""
					The 'AF' column refers to the absolute token frequency.
					The 'RF'column refers to the relative token frequency (normalized per 100 tokens).
					Note that for part-of-speech tags, tokens are normalized against word tokens,
					while DocuScope tags are normalized against counts of all tokens including punctuation.
					The 'Range' column refers to the percentage of documents in which the token appears in your corpus.
					""")
	
		with st.expander("Filtering and saving"):
			st.write("""
					Columns can be filtered by hovering over the column header and clicking on the 3 lines that appear.
					Clicking on the middle funnel icon will show the various filtering options.
					Alternatively, filters can be accessed by clicking 'Filters' on the sidebar.\n
					For text columns, you can filter by 'Equals', 'Starts with', 'Ends with', and 'Contains'.
					""")
	
		selected = grid_response['selected_rows'] 
		if selected:
			st.write('Selected rows')
			df = pd.DataFrame(selected).drop('_selectedRowNodeInfo', axis=1)
			st.dataframe(df)


		if st.button("Download"):
			with st.spinner('Creating download link...'):
				towrite = BytesIO()
				downloaded_file = df.to_excel(towrite, encoding='utf-8', index=False, header=True)
				towrite.seek(0)  # reset pointer
				b64 = base64.b64encode(towrite.read()).decode()  # some strings
				st.success('Link generated!')
				linko= f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="tag_frequencies.xlsx">Download Excel file</a>'
				st.markdown(linko, unsafe_allow_html=True)

	else:
		tag_radio_tags = st.radio("Select tags to display:", ("Parts-of-Speech", "DocuScope"), index=idx_7, on_change=increment_counter_7, horizontal=True)

		if tag_radio_tags == 'Parts-of-Speech':
			df = st.session_state.kt_pos
		else:
			df = st.session_state.kt_ds
	
		gb = GridOptionsBuilder.from_dataframe(df)
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
		
		selected = grid_response['selected_rows'] 
		if selected:
			st.write('Selected rows')
			df = pd.DataFrame(selected).drop('_selectedRowNodeInfo', axis=1)
			st.dataframe(df)


		if st.button("Download"):
			with st.spinner('Creating download link...'):
				towrite = BytesIO()
				downloaded_file = df.to_excel(towrite, encoding='utf-8', index=False, header=True)
				towrite.seek(0)  # reset pointer
				b64 = base64.b64encode(towrite.read()).decode()  # some strings
				st.success('Link generated!')
				linko= f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="tag_frequencies.xlsx">Download Excel file</a>'
				st.markdown(linko, unsafe_allow_html=True)

else:
	st.write("Use the button to generate a tag frequency table from your corpus.")

	if st.button("Keyness Table"):
		if st.session_state.corpus == '':
			st.write("It doesn't look like you've loaded a target corpus yet.")
		elif st.session_state.reference == '':
			st.write("It doesn't look like you've loaded a reference corpus yet.")
		else:
			tp_ref = st.session_state.reference
			wc_ref_pos = ds.frequency_table(tp_ref, st.session_state.ref_words)
			wc_ref_ds = ds.frequency_table(tp_ref, st.session_state.ref_tokens, count_by='ds')
			tc_ref_pos = ds.tags_table(tp_ref, st.session_state.ref_words)
			tc_ref_ds = ds.tags_table(tp_ref, st.session_state.ref_tokens, count_by='ds')
			if bool(isinstance(st.session_state.tt_pos, pd.DataFrame)) == False:
				tp = st.session_state.corpus
				tc_pos = ds.tags_table(tp, st.session_state.words)
				tc_ds = ds.tags_table(tp, st.session_state.tokens, count_by='ds')
				st.session_state.tt_pos = tc_pos
				st.session_state.tt_ds = tc_ds
			if bool(isinstance(st.session_state.ft_pos, pd.DataFrame)) == False:
				tp = st.session_state.corpus
				wc_pos = ds.frequency_table(tp, st.session_state.words)
				wc_ds = ds.frequency_table(tp, st.session_state.tokens, count_by='ds')
				st.session_state.ft_pos = wc_pos
				st.session_state.ft_ds = wc_ds
			with st.spinner('Generating keywords...'):
				kw_pos = ds.keyness_table(st.session_state.ft_pos, wc_ref_pos)
				kw_ds = ds.keyness_table(st.session_state.ft_ds, wc_ref_ds)
				kt_pos = ds.keyness_table(st.session_state.tt_pos, tc_ref_pos, tags_only=True)
				kt_ds = ds.keyness_table(st.session_state.tt_ds, tc_ref_ds, tags_only=True)
				st.session_state.kw_pos = kw_pos
				st.session_state.kw_ds = kw_ds
				st.session_state.kt_pos = kt_pos
				st.session_state.kt_ds = kt_ds
				st.success('Processing complete!')
				st.experimental_rerun()