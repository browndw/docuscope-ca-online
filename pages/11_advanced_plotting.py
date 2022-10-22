import streamlit as st

# NLP Pkgs
import docuscospacy.corpus_analysis as ds
from tmtoolkit.bow.dtm import dtm_to_dataframe
from tmtoolkit.bow.bow_stats import tf_proportions, tfidf

import pandas as pd
import plotly.express as px

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

if 'sums_pos' not in st.session_state:
	st.session_state.sums_pos = 0

if 'sums_ds' not in st.session_state:
	st.session_state.sums_ds = 0

if 'units' not in st.session_state:
	st.session_state.dtm_ds = ''

#prevent categories from being chosen in both multiselect
def update_grpa():
	if len(list(set(st.session_state.grpa) & set(st.session_state.grpb))) > 0:
		item = list(set(st.session_state.grpa) & set(st.session_state.grpb))
		st.session_state.grpa = list(set(list(st.session_state.grpa))^set(item))

def update_grpb():
	if len(list(set(st.session_state.grpa) & set(st.session_state.grpb))) > 0:
		item = list(set(st.session_state.grpa) & set(st.session_state.grpb))
		st.session_state.grpb = list(set(list(st.session_state.grpb))^set(item))

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
	
	st.markdown("""---""")
	
	if st.session_state.units == 'norm':
		cats = list(df.columns)
		
		st.markdown("#### Boxplots")
		box_vals = st.multiselect("Select variables for plotting:", (cats))
		if st.button("Boxplots of Frequencies"):
			df_plot = df[box_vals]
			df_plot.index.name = 'doc_id'
			df_plot.reset_index(inplace=True)
			df_plot = pd.melt(df_plot,id_vars=['doc_id'],var_name='Tag', value_name='RF')
			df_plot['Median'] = df_plot.groupby(['Tag']).transform('median')
			df_plot.sort_values(by='Median', inplace=True, ignore_index=True)
			fig = px.box(df_plot, x='RF', y='Tag', template='plotly_white', orientation='h', hover_data=['doc_id'])
			fig.update_layout(paper_bgcolor='white', plot_bgcolor='white')
			fig.update_yaxes(zeroline=True, linecolor='black')
			fig.update_xaxes(zeroline=True, linecolor='black', rangemode="tozero")
			#fig.update_layout(yaxis={'categoryorder':'total ascending'})
			st.plotly_chart(fig)
		
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
				fig = px.box(df_plot, x='RF', y='Tag', template='plotly_white', orientation='h', hover_data=['doc_id'], color="Group")
				fig.update_layout(paper_bgcolor='white', plot_bgcolor='white', legend={'traceorder':'reversed'})
				fig.update_yaxes(zeroline=True, linecolor='black')
				fig.update_xaxes(zeroline=True, linecolor='black', rangemode="tozero")
				#fig.update_layout(yaxis={'categoryorder':'total ascending'})
				st.plotly_chart(fig)


		st.markdown("""---""") 
		st.markdown("#### Scatterplots")
		xaxis = st.selectbox("Select variable for the x-axis", (cats))
		yaxis = st.selectbox("Select variable for the y-axis", (cats))

		if st.button("Scatterplot of Frequencies"):
			fig = px.scatter(df, x=xaxis, y=yaxis, template='plotly_white', hover_data=[df.index])
			fig.update_layout(paper_bgcolor='white', plot_bgcolor='white')
			fig.update_yaxes(zeroline=True, linecolor='black', rangemode="tozero")
			fig.update_xaxes(zeroline=True, linecolor='black', rangemode="tozero")

			#fig.update_layout(yaxis={'categoryorder':'total ascending'})
			st.plotly_chart(fig, use_container_width=True)
			cc = np.corrcoef(df[xaxis], df[yaxis])[0, 1]
			st.markdown(f"""Pearson's correlation coefficient: {cc.round(3)}
					""")

		st.markdown("""---""") 
		st.markdown("#### Principal Component Analysis")

		if st.button("PCA"):
			pca = PCA(n_components=2)
			pca_result = pca.fit_transform(df.values)
			pca_df = pd.DataFrame({'doc_id': list(df.index),'pca-one': pca_result[:,0], 'pca-two': pca_result[:,1]})

			fig = px.scatter(pca_df, x='pca-one', y='pca-two', template='plotly_white', hover_data=['doc_id'])
			fig.update_layout(paper_bgcolor='white', plot_bgcolor='white')
			#fig.update_yaxes(zeroline=True, linecolor='black', rangemode="tozero")
			#fig.update_xaxes(zeroline=True, linecolor='black', rangemode="tozero")

			#fig.update_layout(yaxis={'categoryorder':'total ascending'})
			st.plotly_chart(fig, use_container_width=True)
			ve = np.array(pca.explained_variance_ratio_*100).round(2).astype('str').tolist()
			st.markdown(f"""Explained variation per principal component: {', '.join(ve)}
					""")
			
	
	st.markdown("""---""") 
	if st.button("Create New DTM"):
		st.session_state.dtm_pos = ''
		st.session_state.dtm_simple = ''
		st.session_state.dtm_ds = ''
		st.experimental_rerun()
	
	#dtm_pos = dtm_pos.multiply(pos_sums, axis=0)

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
				st.session_state.dtm_pos = dtm_pos
				st.session_state.dtm_ds = dtm_ds
				st.session_state.sums_pos = sums_pos
				st.session_state.sums_ds = sums_ds
				st.session_state.units = units
				st.success('DTM generated!')
				st.experimental_rerun()


				