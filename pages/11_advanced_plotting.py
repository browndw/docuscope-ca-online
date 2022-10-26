import streamlit as st

# NLP Pkgs
import docuscospacy.corpus_analysis as ds
from tmtoolkit.bow.dtm import dtm_to_dataframe
from tmtoolkit.bow.bow_stats import tf_proportions, tfidf

import pandas as pd
import altair as alt

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import base64
from io import BytesIO

if 'corpus' not in st.session_state:
	st.session_state.corpus = ''

if 'dtm_pos' not in st.session_state:
	st.session_state.dtm_pos = ''

if 'dtm_ds' not in st.session_state:
	st.session_state.dtm_ds = ''

if 'dtm_simple' not in st.session_state:
	st.session_state.dtm_simple = ''

if 'pca' not in st.session_state:
	st.session_state.pca = ''

if 'contrib' not in st.session_state:
	st.session_state.contrib = ''

if 'sums_pos' not in st.session_state:
	st.session_state.sums_pos = 0

if 'sums_ds' not in st.session_state:
	st.session_state.sums_ds = 0

if 'units' not in st.session_state:
	st.session_state.dtm_ds = ''

if 'pca_idx' not in st.session_state:
	st.session_state.pca_idx = 1
	
if 'variance' not in st.session_state:
	st.session_state.pca = []

	
#prevent categories from being chosen in both multiselect
def update_grpa():
	if len(list(set(st.session_state.grpa) & set(st.session_state.grpb))) > 0:
		item = list(set(st.session_state.grpa) & set(st.session_state.grpb))
		st.session_state.grpa = list(set(list(st.session_state.grpa))^set(item))

def update_grpb():
	if len(list(set(st.session_state.grpa) & set(st.session_state.grpb))) > 0:
		item = list(set(st.session_state.grpa) & set(st.session_state.grpb))
		st.session_state.grpb = list(set(list(st.session_state.grpb))^set(item))

def update_pca(coord_data, contrib_data):
	pca_x = coord_data.columns[st.session_state.pca_idx - 1]
	pca_y = coord_data.columns[st.session_state.pca_idx]
	
	contrib_1 = contrib_data[contrib_data[pca_x].abs() > 1]
	contrib_1[pca_x] = contrib_1[pca_x].div(100)
	contrib_2 = contrib_data[contrib_data[pca_y].abs() > 1]
	contrib_2[pca_y] = contrib_2[pca_y].div(100)
	contrib_1.sort_values(by=pca_x, ascending=True, inplace=True)
	contrib_2.sort_values(by=pca_y, ascending=True, inplace=True)
	
	ve_1 = "{:.2%}".format(st.session_state.variance[st.session_state.pca_idx - 1])
	ve_2 = "{:.2%}".format(st.session_state.variance[st.session_state.pca_idx])

	cp_1 = alt.Chart(contrib_1, height={"step": 12}).mark_bar(size=10).encode(
				x=alt.X(pca_x, axis=alt.Axis(format='%')), 
				y=alt.Y('Tag', sort='-x', title=None),
	 			tooltip=[
     	  		 alt.Tooltip(pca_x, title="Percentage of Contribution", format='.2%')
    			])
	
	cp_2 = alt.Chart(contrib_2, height={"step": 12}).mark_bar(size=10).encode(
				x=alt.X(pca_y, 
				axis=alt.Axis(format='%')), 
				y=alt.Y('Tag', sort='-x', title=None),
				tooltip=[
				alt.Tooltip(pca_y, title="Percentage of Contribution", format='.2%')
				])
	
	base = alt.Chart(coord_data).mark_circle(size=50).encode(
		alt.X(pca_x),
		alt.Y(pca_y),
		tooltip=['doc_id:N']
		)
	
	groups = sorted(set(st.session_state.doccats))
	
	# A dropdown filter
	group_dropdown = alt.binding_select(options=groups)
	group_select = alt.selection_single(fields=['Group'], bind=group_dropdown, name="Select")
	group_color_condition = alt.condition(group_select,
                      alt.Color('Group:N', legend=None),
                      alt.value('lightgray'))
    
	highlight_groups = base.add_selection(group_select).encode(color=group_color_condition)
	
	#zero axes
	line_y = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule().encode(y=alt.Y('y', title=pca_y))
	line_x = alt.Chart(pd.DataFrame({'x': [0]})).mark_rule().encode(x=alt.X('x', title=pca_x))

	st.altair_chart(highlight_groups + line_y + line_x, use_container_width = True)
	
	st.markdown(f""" Variance explained: {ve_1}, {ve_2}
				""")
	st.markdown("Variable contribution to principal component (showing only those > 1.0%):")
	
	col1,col2 = st.columns(2)
	col1.altair_chart(cp_1, use_container_width = True)
	col2.altair_chart(cp_2, use_container_width = True)

st.title("Create plots of frequencies or categories")

if bool(isinstance(st.session_state.dtm_pos, pd.DataFrame)) == True:
	tag_radio_tokens = st.radio("Select tags to display:", ("Parts-of-Speech", "DocuScope"), horizontal=True)

	if st.session_state.units == 'norm':
		if tag_radio_tokens == 'Parts-of-Speech':
			tag_type = st.radio("Select from general or specific tags", ("General", "Specific"), horizontal=True)
			if tag_type == 'General':
				df = st.session_state.dtm_simple
			else:
				df = st.session_state.dtm_pos
		else:
			df = st.session_state.dtm_ds
	
	else:
		if tag_radio_tokens == 'Parts-of-Speech':
			df = st.session_state.dtm_pos
		else:
			df = st.session_state.dtm_ds

	st.dataframe(df)
	
	if st.session_state.units == 'norm':
		de_norm = st.radio('Do you want to de-normalize the frequencies prior to download?', ('No', 'Yes'), horizontal = True)

	if st.button("Download"):
		if de_norm == 'Yes' and tag_radio_tokens == 'Parts-of-Speech':
			with st.spinner('Creating download link...'):
				output_df = df.copy()
				output_df = output_df.multiply(st.session_state.sums_pos, axis=0)
				output_df = output_df.divide(100, axis=0)
				output_df.index.name = 'doc_id'
				output_df.reset_index(inplace=True)
				towrite = BytesIO()
				downloaded_file = output_df.to_excel(towrite, encoding='utf-8', index=False, header=True)
				towrite.seek(0)  # reset pointer
				b64 = base64.b64encode(towrite.read()).decode()  # some strings
				st.success('Link generated!')
				linko= f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="tag_dfm.xlsx">Download Excel file</a>'
				st.markdown(linko, unsafe_allow_html=True)
		elif de_norm == 'Yes' and tag_radio_tokens == 'DocuScope':
			with st.spinner('Creating download link...'):
				output_df = df.copy()
				output_df = output_df.multiply(st.session_state.sums_ds, axis=0)
				output_df = output_df.divide(100, axis=0)
				output_df.index.name = 'doc_id'
				output_df.reset_index(inplace=True)
				towrite = BytesIO()
				downloaded_file = output_df.to_excel(towrite, encoding='utf-8', index=False, header=True)
				towrite.seek(0)  # reset pointer
				b64 = base64.b64encode(towrite.read()).decode()  # some strings
				st.success('Link generated!')
				linko= f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="tag_dfm.xlsx">Download Excel file</a>'
				st.markdown(linko, unsafe_allow_html=True)
		else:
			with st.spinner('Creating download link...'):
				output_df = df.copy()
				output_df.index.name = 'doc_id'
				output_df.reset_index(inplace=True)
				towrite = BytesIO()
				downloaded_file = output_df.to_excel(towrite, encoding='utf-8', index=False, header=True)
				towrite.seek(0)  # reset pointer
				b64 = base64.b64encode(towrite.read()).decode()  # some strings
				st.success('Link generated!')
				linko= f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="tag_dfm.xlsx">Download Excel file</a>'
				st.markdown(linko, unsafe_allow_html=True)

	
	st.markdown("""---""")
	
	if st.session_state.units == 'norm':
		cats = list(df.columns)
		
		st.markdown("#### Boxplots")
		box_vals = st.multiselect("Select variables for plotting:", (cats))
		if st.button("Boxplots of Frequencies"):
			if len(box_vals) == 0:
				st.markdown("Choose a variable to plot")
			elif len(box_vals) > 0:
				df_plot = df[box_vals]
				df_plot.index.name = 'doc_id'
				df_plot.reset_index(inplace=True)
				df_plot = pd.melt(df_plot,id_vars=['doc_id'],var_name='Tag', value_name='RF')
				df_plot['Median'] = df_plot.groupby(['Tag']).transform('median')
				df_plot.sort_values(by='Median', inplace=True, ignore_index=True, ascending=False)
				cols = df_plot['Tag'].drop_duplicates().tolist()
			
			
				base = alt.Chart(df_plot).mark_boxplot(ticks=True).encode(
    				x = alt.X('RF', title='Frequency (per 100 tokens)'),
    				y = alt.Y('Tag', sort=cols, title='')
					)
					
				st.altair_chart(base, use_container_width=True)
		
		if st.session_state.doccats != '':
			st.markdown('##### Add grouping variables')
			st.markdown('###### Group A')
			st.multiselect("Select categories for group A:", (sorted(set(st.session_state.doccats))), on_change = update_grpa, key='grpa')
			
			st.markdown('###### Group B')
			st.multiselect("Select categories for group B:", (sorted(set(st.session_state.doccats))), on_change = update_grpb, key='grpb')
			if st.button("Boxplots of Frequencies by Group"):
				grpa_list = [item + "_" for item in list(st.session_state.grpa)]
				grpb_list = [item + "_" for item in list(st.session_state.grpb)]
				df_plot = df[box_vals]
				df_plot.loc[df_plot.index.str.startswith(tuple(grpa_list)), 'Group'] = 'Group A'
				df_plot.loc[df_plot.index.str.startswith(tuple(grpb_list)), 'Group'] = 'Group B'
				df_plot = df_plot.dropna()
				
				df_plot.index.name = 'doc_id'
				df_plot.reset_index(inplace=True)
				df_plot = pd.melt(df_plot,id_vars=['doc_id', 'Group'],var_name='Tag', value_name='RF')
				df_plot['Median'] = df_plot.groupby(['Tag', 'Group']).transform('median')
				df_plot.sort_values(by=['Group', 'Median'], ascending=[False, True], inplace=True, ignore_index=True)
				plot = alt.Chart(df_plot).mark_boxplot(ticks=True).encode(
    				alt.X('RF', title='Frequency (per 100 tokens)'),
    				alt.Y('Group', title='', axis=alt.Axis(labels=False, ticks=False)),
    				alt.Color('Group'),
    					row=alt.Row('Tag', title='', header=alt.Header(orient='left', labelAngle=0, labelAlign='left'), sort=alt.SortField(field='Median', order='descending'))
						).configure_facet(
						spacing=10
						).configure_view(
						stroke=None
					)
				
				st.altair_chart(plot, use_container_width=False)
				

		st.markdown("""---""") 
		st.markdown("#### Scatterplots")
		xaxis = st.selectbox("Select variable for the x-axis", (cats))
		yaxis = st.selectbox("Select variable for the y-axis", (cats))

		if st.button("Scatterplot of Frequencies"):
			df_plot = df.copy()
			df_plot.index.name = 'doc_id'
			df_plot.reset_index(inplace=True)
			fig = px.scatter(df, x=xaxis, y=yaxis, template='plotly_white', hover_data=[df.index])
			fig.update_layout(paper_bgcolor='white', plot_bgcolor='white')
			fig.update_yaxes(zeroline=True, linecolor='black', rangemode="tozero")
			fig.update_xaxes(zeroline=True, linecolor='black', rangemode="tozero")

			#fig.update_layout(yaxis={'categoryorder':'total ascending'})
			#st.plotly_chart(fig, use_container_width=True)
			base = alt.Chart(df_plot).mark_circle().encode(
    			alt.X(xaxis),
    			alt.Y(yaxis),
    			tooltip=['doc_id:N']
			)
			
			st.altair_chart(base, use_container_width=False)
			
			cc = np.corrcoef(df[xaxis], df[yaxis])[0, 1]
			st.markdown(f"""Pearson's correlation coefficient: {cc.round(3)}
					""")

	st.markdown("""---""") 
	st.markdown("#### Principal Component Analysis")

	if st.button("PCA"):
		del st.session_state.pca_idx
		del st.session_state.pca
		del st.session_state.contrib
		st.session_state.pca_idx = 1
		st.session_state.pca = ''
		st.session_state.contrib = ''
		pca = PCA(n_components=len(df.columns))
		pca_result = pca.fit_transform(df.values)
		pca_df = pd.DataFrame(pca_result)
		pca_df.columns = ['PC' + str(col + 1) for col in pca_df.columns]
					
		sdev = pca_df.std(ddof=0)
		contrib = []
		for i in range(0, len(sdev)):
			coord = pca.components_[i] * sdev[i]
			polarity = np.divide(coord, abs(coord))
			coord = np.square(coord)
			coord = np.divide(coord, sum(coord))*100
			coord = np.multiply(coord, polarity)				
			contrib.append(coord)
		contrib_df =  pd.DataFrame(contrib).transpose()
		contrib_df.columns = ['PC' + str(col + 1) for col in contrib_df.columns]
		contrib_df['Tag'] = df.columns
		pca_df['Group'] = st.session_state.doccats
		pca_df['doc_id'] = list(df.index)		
		ve = np.array(pca.explained_variance_ratio_).tolist()
		st.session_state.variance = ve
		st.session_state.pca = pca_df
		st.session_state.contrib = contrib_df
		
	if bool(isinstance(st.session_state.pca, pd.DataFrame)) == True:
		st.selectbox("Select principal component to plot ", (list(range(1, len(df.columns)))), on_change=update_pca(st.session_state.pca, st.session_state.contrib), key='pca_idx')
		#st.multiselect("Select categories to highlight", (sorted(set(st.session_state.doccats))), on_change=update_pca(st.session_state.pca, st.session_state.contrib), key='pcacolors')
			
	
	st.markdown("""---""") 
	if st.button("Create New DTM"):
		del st.session_state.pca_idx
		st.session_state.dtm_pos = ''
		st.session_state.dtm_simple = ''
		st.session_state.dtm_ds = ''
		st.session_state.pca = ''
		st.session_state.contrib = ''
		st.session_state.variance = []
		st.session_state.pca_idx = 1
		st.experimental_rerun()
	

else:
	st.markdown("Use the menus to generate a document-term matrix for plotting and analysis.")
	
	dtm_type = st.radio("Select the type of dtm:", ("Normalized", "TF-IDF"), horizontal=True)
	if dtm_type == 'Normalized':
		scale = st.radio("Do you want to scale the variables?", ("No", "Yes"), horizontal=True)
	
	if st.button("Document-Term Matrix"):
		if st.session_state.corpus == '':
			st.write("It doesn't look like you've loaded a corpus yet.")
		else:
			with st.spinner('Generating dtm for plotting...'):
				tp = st.session_state.corpus
				#prep part-of-speech tag counts
				dtm_pos = ds.tags_dtm(tp, count_by='pos')
				dtm_pos.set_index('doc_id', inplace=True)
				sums_pos = np.array(dtm_pos.sum(axis=1))
				#prep docuscope tag counts
				dtm_ds = ds.tags_dtm(tp, count_by='ds')
				dtm_ds.set_index('doc_id', inplace=True)
				sums_ds = np.array(dtm_ds.sum(axis=1))
				#apply transformations
				if dtm_type == 'Normalized' and scale == 'No':
					#create dtm with simplified categories
					dtm_simple = dtm_pos.copy()
					dtm_simple.index.name = 'doc_id'
					dtm_simple.reset_index(inplace=True)
					#need index to maintain order
					dtm_simple['doc_order'] = dtm_simple.index
					dtm_simple = pd.melt(dtm_simple,id_vars=['doc_id', 'doc_order'],var_name='Tag', value_name='RF')
					dtm_simple['Tag'].replace('^NN\S*$', '#Noun', regex=True, inplace=True)
					dtm_simple['Tag'].replace('^VV\S*$', '#Verb', regex=True, inplace=True)
					dtm_simple['Tag'].replace('^JJ\S*$', '#Adjective', regex=True, inplace=True)
					dtm_simple['Tag'].replace('^R\S*$', '#Adverb', regex=True, inplace=True)
					dtm_simple['Tag'].replace('^P\S*$', '#Pronoun', regex=True, inplace=True)
					dtm_simple['Tag'].replace('^I\S*$', '#Preposition', regex=True, inplace=True)
					dtm_simple['Tag'].replace('^C\S*$', '#Conjunction', regex=True, inplace=True)
					dtm_simple = dtm_simple.loc[dtm_simple["Tag"].str.startswith('#', na=False)]
					dtm_simple['Tag'].replace('^#', '', regex=True, inplace=True)
					#sum tags
					dtm_simple = dtm_simple.groupby(['doc_id', 'doc_order', 'Tag'], as_index=False)['RF'].sum()
					dtm_simple.sort_values(by='doc_order', inplace=True, ignore_index=True)
					dtm_simple = dtm_simple.pivot_table(index=['doc_order', 'doc_id'], columns='Tag', values='RF')
					dtm_simple.reset_index(inplace=True)
					dtm_simple.drop('doc_order', axis=1, inplace=True)
					dtm_simple.set_index('doc_id', inplace=True)
					dtm_simple = dtm_simple.divide(sums_pos, axis=0)
					dtm_simple *= 100
					#create dtm for all pos categories
					dtm_pos = tf_proportions(dtm_pos)
					dtm_pos *= 100
					#and ds categories
					dtm_ds = tf_proportions(dtm_ds)
					dtm_ds *= 100
					units = 'norm'
					st.session_state.dtm_simple = dtm_simple
				elif dtm_type == 'Normalized' and scale == 'Yes':
					dtm_pos = tf_proportions(dtm_pos)
					dtm_ds = tf_proportions(dtm_ds)
					scaled_pos = StandardScaler().fit_transform(dtm_pos.values)
					scaled_ds = StandardScaler().fit_transform(dtm_ds.values)
					dtm_pos = pd.DataFrame(scaled_pos, index=dtm_pos.index, columns=dtm_pos.columns)
					dtm_ds = pd.DataFrame(scaled_ds, index=dtm_ds.index, columns=dtm_ds.columns)
					units = 'scaled'
				else:
					dtm_pos = tfidf(dtm_pos)
					dtm_ds = tfidf(dtm_ds)
					units = 'tfidf'
				dtm_ds.drop('Untagged', axis=1, inplace=True, errors='ignore')
				st.session_state.dtm_pos = dtm_pos
				st.session_state.dtm_ds = dtm_ds
				st.session_state.sums_pos = sums_pos
				st.session_state.sums_ds = sums_ds
				st.session_state.units = units
				st.success('DTM generated!')
				st.experimental_rerun()


				