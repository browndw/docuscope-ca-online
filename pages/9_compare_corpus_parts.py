import streamlit as st

# NLP Pkgs
import docuscospacy.corpus_analysis as ds

import pandas as pd
import altair as alt

import base64
from io import BytesIO
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode

if 'corpus' not in st.session_state:
	st.session_state.corpus = ''

if 'doccats' not in st.session_state:
	st.session_state.doccats = ''

if 'kw_pos_cp' not in st.session_state:
	st.session_state.kw_pos_cp = ''

if 'kw_ds_cp' not in st.session_state:
	st.session_state.kw_ds_cp = ''

# a method for preserving button selection on page interactions
# with quick clicking it can lag
if 'count_8' not in st.session_state:
	st.session_state.count_8 = 0

def increment_counter_8():
	st.session_state.count_8 += 1

if st.session_state.count_8 % 2 == 0:
    idx_8 = 0
else:
    idx_8 = 1

if 'count_9' not in st.session_state:
	st.session_state.count_9 = 0

def increment_counter_9():
	st.session_state.count_9 += 1

if st.session_state.count_9 % 2 == 0:
    idx_9 = 0
else:
    idx_9 = 1

if 'count_10' not in st.session_state:
	st.session_state.count_10 = 0

def increment_counter_10():
	st.session_state.count_10 += 1

if st.session_state.count_10 % 2 == 0:
    idx_10 = 0
else:
    idx_10 = 1

#prevent categories from being chosen in both multiselect
def update_tar():
	if len(list(set(st.session_state.tar) & set(st.session_state.ref))) > 0:
		item = list(set(st.session_state.tar) & set(st.session_state.ref))
		st.session_state.tar = list(set(list(st.session_state.tar))^set(item))

def update_ref():
	if len(list(set(st.session_state.tar) & set(st.session_state.ref))) > 0:
		item = list(set(st.session_state.tar) & set(st.session_state.ref))
		st.session_state.ref = list(set(list(st.session_state.ref))^set(item))

st.title("Create a keyness tables using pairwise comparisions of corpus parts")

st.markdown("[![User Guide](https://raw.githubusercontent.com/browndw/corpus-tagger/main/_static/user_guide.svg)](https://browndw.github.io/docuscope-docs/compare_corpus_parts.html)")

if bool(isinstance(st.session_state.kw_pos_cp, pd.DataFrame)) == True:

	table_radio = st.radio("Select the keyness table to display:", ("Tokens", "Tags Only"), index=idx_8, on_change=increment_counter_8, horizontal=True)
	if table_radio == 'Tokens':
		tag_radio_tokens = st.radio("Select tags to display:", ("Parts-of-Speech", "DocuScope"), index=idx_9, on_change=increment_counter_9, horizontal=True)

		if tag_radio_tokens == 'Parts-of-Speech':
			df = st.session_state.kw_pos_cp
		else:
			df = st.session_state.kw_ds_cp

		col1, col2 = st.columns([1,1])
		
		with col1:
			st.markdown("#### Target corpus:")
			st.write("Document categories: ", ', '.join(st.session_state.tar_cats))
			st.write("Number of tokens: ", str(st.session_state.tar_tokens))
			st.write("Number of word tokens: ", str(st.session_state.tar_words))
		with col2:
			st.markdown("#### Reference corpus:")
			st.write("Document categories: ", ', '.join(st.session_state.ref_cats))
			st.write("Number of tokens: ", str(st.session_state.ref_tokens))
			st.write("Number of word tokens: ", str(st.session_state.ref_words))

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
						The 'AF' column refers to the absolute token frequency.
						The 'RF'column refers to the relative token frequency (normalized per million tokens).
						Note that for part-of-speech tags, tokens are normalized against word tokens,
						while DocuScope tags are normalized against counts of all tokens including punctuation.
						The 'Range' column refers to the percentage of documents in which the token appears in your corpus.
						""")
	
		with st.expander("Filtering and saving"):
			st.markdown("""
						Filters can be accessed by clicking on the three lines that appear while hovering over a column header.
						For text columns, you can filter by 'Equals', 'Starts with', 'Ends with', and 'Contains'.\n
						Rows can be selected before or after filtering using the checkboxes.
						(The checkbox in the header will select/deselect all rows.)\n
						If rows are selected and appear in new table below the main one,
						those selected rows will be available for download in an Excel file.
						If no rows are selected, the full table will be processed for downloading after clicking the Download button.
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
		
		if st.button("Compare New Categories"):
			st.session_state.kw_pos_cp = ''
			st.session_state.kw_ds_cp = ''
			st.session_state.kt_pos_cp = ''
			st.session_state.kt_ds_cp = ''
			st.session_state.tar_tokens = 0
			st.session_state.tar_words = 0
			st.session_state.ref_tokens = 0
			st.session_state.tar_words = 0
			st.session_state.tar_cats = []
			st.session_state.ref_cats = []
			st.experimental_rerun()

	else:
		tag_radio_tags = st.radio("Select tags to display:", ("Parts-of-Speech", "DocuScope"), index=idx_10, on_change=increment_counter_10, horizontal=True)

		if tag_radio_tags == 'Parts-of-Speech':
			df = st.session_state.kt_pos_cp
		else:
			df = st.session_state.kt_ds_cp

		col1, col2 = st.columns([1,1])
		
		with col1:
			st.markdown("#### Target corpus:")
			st.write("Document categories: ", ', '.join(st.session_state.tar_cats))
			st.write("Number of tokens: ", str(st.session_state.tar_tokens))
			st.write("Number of word tokens: ", str(st.session_state.tar_words))
		with col2:
			st.markdown("#### Reference corpus:")
			st.write("Document categories: ", ', '.join(st.session_state.ref_cats))
			st.write("Number of tokens: ", str(st.session_state.ref_tokens))
			st.write("Number of word tokens: ", str(st.session_state.ref_words))
	
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
						The 'AF' column refers to the absolute token frequency.
						The 'RF'column refers to the relative token frequency (normalized per 100 tokens).
						Note that for part-of-speech tags, tokens are normalized against word tokens,
						while DocuScope tags are normalized against counts of all tokens including punctuation.
						The 'Range' column refers to the percentage of documents in which the token appears in your corpus.
						""")
	
		with st.expander("Filtering and saving"):
			st.markdown("""
						Filters can be accessed by clicking on the three lines that appear while hovering over a column header.
						For text columns, you can filter by 'Equals', 'Starts with', 'Ends with', and 'Contains'.\n
						Rows can be selected before or after filtering using the checkboxes.
						(The checkbox in the header will select/deselect all rows.)\n
						If rows are selected and appear in new table below the main one,
						those selected rows will be available for download in an Excel file.
						If no rows are selected, the full table will be processed for downloading after clicking the Download button.
						""")
		
		selected = grid_response['selected_rows'] 
		if selected:
			st.write('Selected rows')
			df = pd.DataFrame(selected).drop('_selectedRowNodeInfo', axis=1)
			st.dataframe(df)


		col1, col2 = st.columns([.8, .2])
		
		with col1:
			if st.button("Plot resutls"):
				df_plot = df[["Tag", "RF", "RF Ref"]]
				df_plot["Mean"] = df_plot.mean(numeric_only=True, axis=1)
				df_plot.rename(columns={"Tag": "Tag", "Mean": "Mean", "RF": "Target", "RF Ref": "Reference"}, inplace = True)
				df_plot = pd.melt(df_plot, id_vars=['Tag', 'Mean'],var_name='Corpus', value_name='RF')
				df_plot.sort_values(by=["Mean", "Corpus"], ascending=[True, True], inplace=True)
				
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

		with col2:
			if st.button("Download"):
				with st.spinner('Creating download link...'):
					towrite = BytesIO()
					downloaded_file = df.to_excel(towrite, encoding='utf-8', index=False, header=True)
					towrite.seek(0)  # reset pointer
					b64 = base64.b64encode(towrite.read()).decode()  # some strings
					st.success('Link generated!')
					linko= f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="tag_frequencies.xlsx">Download Excel file</a>'
					st.markdown(linko, unsafe_allow_html=True)
		
		if st.button("Compare New Categories"):
			st.session_state.kw_pos_cp = ''
			st.session_state.kw_ds_cp = ''
			st.session_state.kt_pos_cp = ''
			st.session_state.kt_ds_cp = ''
			st.session_state.tar_tokens = 0
			st.session_state.tar_words = 0
			st.session_state.ref_tokens = 0
			st.session_state.tar_words = 0
			st.session_state.tar_cats = []
			st.session_state.ref_cats = []
			st.experimental_rerun()


else:
	st.markdown("Use the menus to select **target** and **reference** categories from you metadata.")
	st.markdown(":lock: Selecting of the same category as target and reference is prevented.")
	
	st.markdown('#### Target corpus categories:')
	st.multiselect("Select target categories:", (sorted(set(st.session_state.doccats))), on_change = update_tar, key='tar')
	
	st.markdown('#### Reference corpus categories:')
	st.multiselect("Select reference categories:", (sorted(set(st.session_state.doccats))), on_change = update_ref, key='ref')
	
	if len(list(st.session_state.tar)) > 0 and len(list(st.session_state.ref)) > 0:
		if st.button("Keyness Table of Corpus Parts"):
			with st.spinner('Generating keywords...'):
				tp = st.session_state.corpus
				tar_list = [item + "_" for item in list(st.session_state.tar)]
				ref_list = [item + "_" for item in list(st.session_state.ref)]
				tar_docs = {key: value for key, value in tp.items() if key.startswith(tuple(tar_list))}
				ref_docs = {key: value for key, value in tp.items() if key.startswith(tuple(ref_list))}
				#get target counts
				tar_tok = list(tar_docs.values())
				tar_tags = []
				for i in range(0,len(tar_tok)):
					tags = [x[1] for x in tar_tok[i]]
					tar_tags.append(tags)
				tar_tags = [x for xs in tar_tags for x in xs]
				tar_tokens = len(tar_tags)
				tar_words = len([x for x in tar_tags if not x.startswith('Y')])
				#get reference counts
				ref_tok = list(ref_docs.values())
				ref_tags = []
				for i in range(0,len(ref_tok)):
					tags = [x[1] for x in ref_tok[i]]
					ref_tags.append(tags)
				ref_tags = [x for xs in ref_tags for x in xs]
				ref_tokens = len(ref_tags)
				ref_words = len([x for x in ref_tags if not x.startswith('Y')])
			
				wc_tar_pos = ds.frequency_table(tar_docs, tar_words)
				wc_tar_ds = ds.frequency_table(tar_docs, tar_tokens, count_by='ds')
				tc_tar_pos = ds.tags_table(tar_docs, tar_words)
				tc_tar_ds = ds.tags_table(tar_docs, tar_tokens, count_by='ds')

				wc_ref_pos = ds.frequency_table(ref_docs, ref_words)
				wc_ref_ds = ds.frequency_table(ref_docs, ref_tokens, count_by='ds')
				tc_ref_pos = ds.tags_table(ref_docs, ref_words)
				tc_ref_ds = ds.tags_table(ref_docs, ref_tokens, count_by='ds')
			
				kw_pos_cp = ds.keyness_table(wc_tar_pos, wc_ref_pos)
				kw_ds_cp = ds.keyness_table(wc_tar_ds, wc_ref_ds)
				kt_pos_cp = ds.keyness_table(tc_tar_pos, tc_ref_pos, tags_only=True)
				kt_ds_cp = ds.keyness_table(tc_tar_ds, tc_ref_ds, tags_only=True)
				st.session_state.kw_pos_cp = kw_pos_cp
				st.session_state.kw_ds_cp = kw_ds_cp
				st.session_state.kt_pos_cp = kt_pos_cp
				st.session_state.kt_ds_cp = kt_ds_cp
				st.session_state.tar_tokens = tar_tokens
				st.session_state.tar_words = tar_words
				st.session_state.ref_tokens = ref_tokens
				st.session_state.ref_words = ref_words
				st.session_state.tar_cats = st.session_state.tar
				st.session_state.ref_cats = st.session_state.ref
				st.success('Keywords generated!')
				st.experimental_rerun()

		