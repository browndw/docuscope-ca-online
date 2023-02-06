import streamlit as st
import docuscospacy.corpus_analysis as ds
import pandas as pd
import altair as alt
import st_aggrid

import base64
from io import BytesIO

import categories
import states as _states

# a method for preserving button selection on page interactions
# with quick clicking it can lag
def increment_counter_5():
	st.session_state.count_5 += 1

def increment_counter_6():
	st.session_state.count_6 += 1

def increment_counter_7():
	st.session_state.count_7 += 1

CATEGORY = categories.KEYNESS
TITLE = "Compare Corpora"
KEY_SORT = 5

def main():
	# check states to prevent unlikely error
	for key, value in _states.STATES.items():
		if key not in st.session_state:
			setattr(st.session_state, key, value)

	if st.session_state.count_5 % 2 == 0:
	    idx_5 = 0
	else:
	    idx_5 = 1
	if st.session_state.count_6 % 2 == 0:
	    idx_6 = 0
	else:
	    idx_6 = 1
	
	if st.session_state.count_7 % 2 == 0:
	    idx_7 = 0
	else:
	    idx_7 = 1

	if bool(isinstance(st.session_state.kw_pos, pd.DataFrame)) == True:
		st.sidebar.markdown("### Comparison")	
		table_radio = st.sidebar.radio("Select the keyness table to display:", ("Tokens", "Tags Only"), index=idx_5, on_change=increment_counter_5, horizontal=True)
		if table_radio == 'Tokens':
			st.sidebar.markdown("---")
			st.sidebar.markdown("### Tagset")
			tag_radio_tokens = st.sidebar.radio("Select tags to display:", ("Parts-of-Speech", "DocuScope"), index=idx_6, on_change=increment_counter_6, horizontal=True)
	
			if tag_radio_tokens == 'Parts-of-Speech':
				df = st.session_state.kw_pos
			else:
				df = st.session_state.kw_ds
			
			col1, col2 = st.columns([1,1])
			with col1:
				st.markdown('### Target corpus:')
				st.write('Number of tokens in target corpus: ', str(st.session_state.tokens))
				st.write('Number of word tokens in target corpus: ', str(st.session_state.words))
				st.write('Number of documents in target corpus: ', str(st.session_state.ndocs))
			with col2:
				st.markdown('### Reference corpus:')
				st.write('Number of tokens in reference corpus: ', str(st.session_state.ref_tokens))
				st.write('Number of word tokens in referencecorpus: ', str(st.session_state.ref_words))
				st.write('Number of documents in creferenceorpus: ', str(st.session_state.ref_ndocs))
				
		
			gb = st_aggrid.GridOptionsBuilder.from_dataframe(df)
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
						The 'LL' column refers to [log-likelihood](https://ucrel.lancs.ac.uk/llwizard.html),
						a hypothesis test measuring observed vs. expected frequencies.
						Note that a negative value means that the token is more frequent in the reference corpus than the target.\n
						'LR' refers to [Log-Ratio](http://cass.lancs.ac.uk/log-ratio-an-informal-introduction/), which is an [effect size](https://www.scribbr.com/statistics/effect-size/).
						And 'PV' refers to the [p-value](https://scottbot.net/friends-dont-let-friends-calculate-p-values-without-fully-understanding-them/).\n
						The 'AF' columns refer to the absolute token frequencies in the target and reference.
						The 'RF' columns refer to the relative token frequencies (normalized per million tokens).
						Note that for part-of-speech tags, tokens are normalized against word tokens,
						while DocuScope tags are normalized against counts of all tokens including punctuation.
						The 'Range' column refers to the percentage of documents in which the token appears in your corpus.
						""")
				
			selected = grid_response['selected_rows'] 
			if selected:
				st.write('Selected rows')
				df = pd.DataFrame(selected).drop('_selectedRowNodeInfo', axis=1)
				st.dataframe(df)
			
			st.sidebar.markdown("---")
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
						linko= f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="tag_frequencies.xlsx">Download Excel file</a>'
						st.markdown(linko, unsafe_allow_html=True)
			st.sidebar.markdown("---")
	
		else:
			st.sidebar.markdown("### Tagset")
			tag_radio_tags = st.sidebar.radio("Select tags to display:", ("Parts-of-Speech", "DocuScope"), index=idx_7, on_change=increment_counter_7, horizontal=True)
	
			if tag_radio_tags == 'Parts-of-Speech':
				df = st.session_state.kt_pos
			else:
				df = st.session_state.kt_ds
	
			col1, col2 = st.columns([1,1])
			with col1:
				st.markdown('### Target corpus:')
				st.write('Number of tokens in target corpus: ', str(st.session_state.tokens))
				st.write('Number of word tokens in target corpus: ', str(st.session_state.words))
				st.write('Number of documents in target corpus: ', str(st.session_state.ndocs))
			with col2:
				st.markdown('### Reference corpus:')
				st.write('Number of tokens in reference corpus: ', str(st.session_state.ref_tokens))
				st.write('Number of word tokens in referencecorpus: ', str(st.session_state.ref_words))
				st.write('Number of documents in creferenceorpus: ', str(st.session_state.ref_ndocs))
		
			gb = st_aggrid.GridOptionsBuilder.from_dataframe(df)
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
						The 'LL' column refers to [log-likelihood](https://ucrel.lancs.ac.uk/llwizard.html),
						a hypothesis test measuring observed vs. expected frequencies.
						Note that a negative value means that the token is more frequent in the reference corpus than the target.\n
						'LR' refers to [Log-Ratio](http://cass.lancs.ac.uk/log-ratio-an-informal-introduction/), which is an [effect size](https://www.scribbr.com/statistics/effect-size/).
						And 'PV' refers to the [p-value](https://scottbot.net/friends-dont-let-friends-calculate-p-values-without-fully-understanding-them/).\n
						The 'AF' columns refer to the absolute token frequencies in the target and reference.
						The 'RF' columns refer to the relative token frequencies (normalized per 100 tokens).
						Note that for part-of-speech tags, tokens are normalized against word tokens,
						while DocuScope tags are normalized against counts of all tokens including punctuation.
						The 'Range' column refers to the percentage of documents in which the token appears in your corpus.
						""")
		
			selected = grid_response['selected_rows'] 
			if selected:
				st.write('Selected rows')
				df = pd.DataFrame(selected).drop('_selectedRowNodeInfo', axis=1)
				st.dataframe(df)
				
			st.sidebar.markdown("---")
			st.sidebar.markdown("### Plot comparisons")
			st.sidebar.markdown("""
								Click the button to genenerate a plot of comparative frequencies.
								You can use the checkboxes to plot selected rows.
								With no rows selected, all variables will be plotted.
								""")
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
							color=alt.Color('Corpus:N', sort=order),
							row=alt.Row('Tag', title=None, header=alt.Header(orient='left', labelAngle=0, labelAlign='left'), sort=alt.SortField(field='Mean', order='descending')),
							tooltip=[
							alt.Tooltip('RF:Q', title="Per 100 Tokens", format='.2')
							]).configure_facet(spacing=0.5).configure_legend(orient='top')				
				st.altair_chart(base, use_container_width=True)
			
			st.sidebar.markdown("---")
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
						linko= f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="tag_frequencies.xlsx">Download Excel file</a>'
						st.markdown(linko, unsafe_allow_html=True)
			st.sidebar.markdown("---")
	
	else:
		st.sidebar.markdown("###  Generate keywords")
		st.sidebar.markdown("Use the button to generate a keywords table from your corpus.")
		if st.sidebar.button("Keyness Table"):
			if st.session_state.corpus == '':
				st.markdown(":neutral_face: It doesn't look like you've loaded a target corpus yet.")
			elif st.session_state.reference == '':
				st.markdown(":neutral_face: It doesn't look like you've loaded a reference corpus yet.")
			else:
				with st.spinner('Generating keywords...'):
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
				
					kw_pos = ds.keyness_table(st.session_state.ft_pos, wc_ref_pos)
					kw_ds = ds.keyness_table(st.session_state.ft_ds, wc_ref_ds)
					kt_pos = ds.keyness_table(st.session_state.tt_pos, tc_ref_pos, tags_only=True)
					kt_ds = ds.keyness_table(st.session_state.tt_ds, tc_ref_ds, tags_only=True)
					st.session_state.kw_pos = kw_pos
					st.session_state.kw_ds = kw_ds
					st.session_state.kt_pos = kt_pos
					st.session_state.kt_ds = kt_ds
					st.success('Keywords generated!')
					st.experimental_rerun()
		
		st.markdown("""
		To use this tool, be sure that you have loaded a reference corpus. 
		Loading a reference can be done from **Manage Corpora**.
		""")

if __name__ == "__main__":
    main()