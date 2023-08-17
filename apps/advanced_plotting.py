# Copyright (C) 2023 David West Brown

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import base64
from io import BytesIO
import pathlib
from importlib.machinery import SourceFileLoader

# set paths
HERE = pathlib.Path(__file__).parents[1].resolve()
OPTIONS = str(HERE.joinpath("options.toml"))
IMPORTS = str(HERE.joinpath("utilities/handlers_imports.py"))

# import options
_imports = SourceFileLoader("handlers_imports", IMPORTS).load_module()
_options = _imports.import_options_general(OPTIONS)

modules = ['analysis', 'categories', 'handlers', 'messages', 'states', 'warnings', 'altair', 'streamlit', 'docuscospacy', 'numpy', 'pandas']
import_params = _imports.import_parameters(_options, modules)

for module in import_params.keys():
	object_name = module
	short_name = import_params[module][0]
	context_module_name = import_params[module][1]
	if not short_name:
		short_name = object_name
	if not context_module_name:
		globals()[short_name] = __import__(object_name)
	else:
		context_module = __import__(context_module_name, fromlist=[object_name])
		globals()[short_name] = getattr(context_module, object_name)


CATEGORY = _categories.OTHER
TITLE = "Advanced Plotting"
KEY_SORT = 9

def main():
	
	user_session = st.runtime.scriptrunner.script_run_context.get_script_run_ctx()
	user_session_id = user_session.session_id

	session = _handlers.load_session(user_session_id)
	
	if  bool(session['dtm']) == True:
		
		metadata_target = _handlers.load_metadata('target', user_session_id)
		
		st.sidebar.markdown("### Tagset")
		
		with st.sidebar.expander("About general tags"):
			st.markdown(_messages.message_general_tags)		

		tag_radio_tokens = st.sidebar.radio("Select tags to display:", ("Parts-of-Speech", "DocuScope"), on_change=_handlers.clear_plots(user_session_id), horizontal=True)
	
		if session['dtm']['units'] == 'norm':
			if tag_radio_tokens == 'Parts-of-Speech':
				tag_type = st.sidebar.radio("Select from general or specific tags", ("General", "Specific"), on_change=_handlers.clear_plots(user_session_id), horizontal=True)
				if tag_type == 'General':
					df = _handlers.load_table('dtm_simple', user_session_id)
				else:
					df = _handlers.load_table('dtm_pos', user_session_id)
			else:
				df = _handlers.load_table('dtm_ds', user_session_id)
				tag_type = None
		
		else:
			if tag_radio_tokens == 'Parts-of-Speech':
				df = _handlers.load_table('dtm_pos', user_session_id)
			else:
				df = _handlers.load_table('dtm_ds', user_session_id)
	
		st.dataframe(df)	
		
		st.markdown("""---""")
		
		if session['dtm']['units'] == 'norm':
			cats = list(df.columns)
			st.sidebar.markdown("---")
			st.sidebar.markdown("### Boxplots")
			box_vals = st.sidebar.multiselect("Select variables for plotting:", (cats))
			if st.sidebar.button("Boxplots of Frequencies"):
				#clear any pca data
				_handlers.update_session('pca', dict(), user_session_id)

				if len(box_vals) == 0:
					st.markdown(_warnings.warning_18, unsafe_allow_html=True)
				
				elif len(box_vals) > 0:
					df_plot, stats, cols = _analysis.boxplots(df, box_vals, tag_radio_tokens, tag_type, grp_a = None, grp_b = None)
						
					base = alt.Chart(df_plot).mark_boxplot(ticks=True).encode(
	    				x = alt.X('RF', title='Frequency (per 100 tokens)'),
	    				y = alt.Y('Tag', sort=cols, title='')
						)
						
					st.markdown(_messages.message_disable_full, unsafe_allow_html=True)
					st.altair_chart(base, use_container_width=True)
					
					st.markdown(_messages.message_stats_info(stats))				
			
			if session['has_meta'] == True:
				st.sidebar.markdown("---")
				st.sidebar.markdown('### Add grouping variables')
				st.sidebar.markdown("Select grouping variables from your metadata and click the button to generate boxplots of frequencies.")
				st.sidebar.markdown('#### Group A')
				st.session_state[user_session_id]['grpa'] = st.sidebar.multiselect("Select categories for group A:", (sorted(set(metadata_target['doccats']))), on_change = _handlers.update_grpa(user_session_id), key=f"grpa_{user_session_id}")
				
				st.sidebar.markdown('#### Group B')
				st.session_state[user_session_id]['grpb'] = st.sidebar.multiselect("Select categories for group B:", (sorted(set(metadata_target['doccats']))), on_change = _handlers.update_grpb(user_session_id), key=f"grpb_{user_session_id}")
				if st.sidebar.button("Boxplots of Frequencies by Group"):
					#clear any pca data
					_handlers.update_session('pca', dict(), user_session_id)

					if len(box_vals) == 0:
						st.markdown(_warnings.warning_19, unsafe_allow_html=True)
					
					elif len(box_vals) > 0:
						grpa_list = [item + "_" for item in list(st.session_state[user_session_id]['grpa'])]
						grpb_list = [item + "_" for item in list(st.session_state[user_session_id]['grpb'])]
						if len(grpb_list) == 0 or len(grpa_list) == 0:
							st.markdown(_warnings.warning_20, unsafe_allow_html=True)
						
						else:
							df_plot, stats = _analysis.boxplots(df, box_vals, tag_radio_tokens, tag_type, grp_a = grpa_list, grp_b = grpb_list)

							plot = alt.Chart(df_plot).mark_boxplot(ticks=True).encode(
			    				alt.X('RF', title='Frequency (per 100 tokens)'),
			    				alt.Y('Group', title='', axis=alt.Axis(labels=False, ticks=False)),
			    				alt.Color('Group'),
			    					row=alt.Row('Tag', title='', header=alt.Header(orient='left', labelAngle=0, labelAlign='left'), sort=alt.SortField(field='Median', order='descending'))
									).configure_facet(
									spacing=10
									).configure_view(
									stroke=None
								).configure_legend(orient='top')
							
							st.markdown(_messages.message_disable_full, unsafe_allow_html=True)
							st.altair_chart(plot, use_container_width=True)					
							
							st.markdown(_messages.message_group_info(grpa_list, grpb_list))
							
							st.markdown(_messages.message_stats_info(stats))
								
			st.sidebar.markdown("""---""") 
			st.sidebar.markdown("### Scatterplots")
			xaxis = st.sidebar.selectbox("Select variable for the x-axis", (cats))
			yaxis = st.sidebar.selectbox("Select variable for the y-axis", (cats))
	
			if st.sidebar.button("Scatterplot of Frequencies"):
				#clear any pca data
				_handlers.update_session('pca', dict(), user_session_id)

				df_plot = df.copy()
				df_plot.index.name = 'doc_id'
				df_plot.reset_index(inplace=True)
				
				x_label = xaxis + ' ' + '(per 100 tokens)'
				y_label = yaxis + ' ' + '(per 100 tokens)'
	
				base = alt.Chart(df_plot).mark_circle().encode(
	    			alt.X(xaxis, title=x_label),
	    			alt.Y(yaxis, title = y_label),
	    			tooltip=['doc_id:N']
				)
				
				st.altair_chart(base, use_container_width=True)
				
				cc_df, cc_r, cc_p = _analysis.correlation(df, xaxis, yaxis)
				
				st.markdown(_messages.message_correlation_info(cc_df, cc_r, cc_p))
	
		if session['dtm']['units'] != 'norm':

			st.sidebar.markdown("""---""") 
			st.sidebar.markdown("### Principal Component Analysis")
			st.sidebar.markdown("""
								Click the button to plot principal compenents.
								""")
		
			if st.sidebar.button("PCA"):
				_handlers.update_session('pca', dict(), user_session_id)
				if session['has_meta'] == True:
					grouping = metadata_target['doccats']
				else:
					grouping = []
	
				pca_df, contrib_df, ve = _analysis.pca_contributions(df, grouping)
				if bool(session['pca']) == False:
					_handlers.update_pca(pca_df, contrib_df, ve, 1, user_session_id)
					st.experimental_rerun()
				else:
					_handlers.update_pca(pca_df, contrib_df, ve, 1, user_session_id)
				
			if bool(session['pca']) == True:
				session['pca']['pca_idx'] = st.sidebar.selectbox("Select principal component to plot ", (list(range(1, len(df.columns)))))
				
				cp_1, cp_2, pca_x, pca_y, contrib_x, contrib_y, ve_1, ve_2 = _analysis.update_pca_plot(session['pca'])
				
				base = alt.Chart(session['pca']['pca']).mark_circle(size=50, opacity=.75).encode(
		    			alt.X(pca_x),
		    			alt.Y(pca_y),
		    			tooltip=['doc_id:N']
		    			)
		
				#zero axes
				line_y = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule().encode(y=alt.Y('y', title=pca_y))
				line_x = alt.Chart(pd.DataFrame({'x': [0]})).mark_rule().encode(x=alt.X('x', title=pca_x))
	
				if session['has_meta'] == True:
					groups = sorted(set(metadata_target['doccats']))
					# A dropdown filter
					group_dropdown = alt.binding_select(options=groups)
					group_select = alt.selection_single(fields=['Group'], bind=group_dropdown, name="Select")
					group_color_condition = alt.condition(group_select,
	                      alt.Color('Group:N', legend=None, scale=alt.Scale(range=['#133955'])),
	                      alt.value('lightgray'))
	                
					highlight_groups = base.add_selection(group_select).encode(color=group_color_condition)
					st.altair_chart(highlight_groups + line_y + line_x, use_container_width = True)
	
				else:
					st.altair_chart(base + line_y + line_x, use_container_width = True)
				
				st.markdown(_messages.message_variance_info(pca_x, pca_y, ve_1, ve_2))
				
				st.markdown(_messages.message_contribution_info(pca_x, pca_y, contrib_x, contrib_y))
				
				st.markdown("##### Variable contribution (by %) to principal component:")
				
				with st.expander("About variable contribution"):
					st.markdown(_messages.message_variable_contrib)

		
				col1,col2 = st.columns(2)
				col1.altair_chart(cp_1, use_container_width = True)
				col2.altair_chart(cp_2, use_container_width = True)


		st.sidebar.markdown("---")
		st.sidebar.markdown(_messages.message_download_dtm)
		
		if session['dtm']['units'] == 'norm': 
			de_norm = st.sidebar.radio('Do you want to de-normalize the frequencies prior to download?', ('No', 'Yes'), horizontal = True)
	
		if st.sidebar.button("Download"):
			if de_norm == 'Yes' and tag_radio_tokens == 'Parts-of-Speech':
				with st.sidebar:
					with st.spinner('Creating download link...'):
						output_df = df.copy()
						output_df = output_df.multiply(session['dtm']['sums_pos'], axis=0)
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
				with st.sidebar:
					with st.spinner('Creating download link...'):
						output_df = df.copy()
						output_df = output_df.multiply(session['dtm']['sums_ds'], axis=0)
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
				with st.sidebar:
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
					
		st.sidebar.markdown("""---""") 
		st.sidebar.markdown(_messages.message_reset_table)
		
		if st.sidebar.button("Create New DTM"):
			if 'grpa' in st.session_state[user_session_id]:
				del st.session_state[user_session_id]['grpa']
			if 'grpb' in st.session_state[user_session_id]:
				del st.session_state[user_session_id]['grpb']
			_handlers.update_session('dtm', dict(), user_session_id)
			_handlers.update_session('pca', dict(), user_session_id)
			st.session_state[user_session_id]['grpa'] = []
			st.session_state[user_session_id]['grpb'] = []
			st.experimental_rerun()
		
		st.sidebar.markdown("""---""") 

	else:
	
		st.markdown(_messages.message_plotting)
	
		dtm_type = st.sidebar.radio("Select the type of DTM:", ("Normalized", "TF-IDF"), horizontal=True)
		if dtm_type == 'Normalized':
			scale = st.sidebar.radio("Do you want to scale the variables?", ("No", "Yes"), horizontal=True)
		
		st.sidebar.markdown("---")
		st.sidebar.markdown(_messages.message_generate_table)
		if st.sidebar.button("Document-Term Matrix"):
			if session.get('target_path') == None:
				st.markdown(_warnings.warning_11, unsafe_allow_html=True)
			else:
				with st.sidebar:
					with st.spinner('Generating dtm for plotting...'):
						tp = _handlers.load_corpus_session('target', session, user_session_id)
						dtm_pos = ds.tags_dtm(tp, count_by='pos')
						dtm_pos.set_index('doc_id', inplace=True)
						sums_pos = np.array(dtm_pos.sum(axis=1))
						dtm_ds = ds.tags_dtm(tp, count_by='ds')
						dtm_ds.set_index('doc_id', inplace=True)
						sums_ds = np.array(dtm_ds.sum(axis=1))
						
						if dtm_type == 'Normalized' and scale == 'No':
							dtm_simple = _analysis.simplify_dtm(dtm_pos, sums_pos)					
							dtm_pos = _analysis.tf_proportions(dtm_pos, norm=True)
							dtm_ds  = _analysis.tf_proportions(dtm_ds, norm=True)
							units = 'norm'
							_handlers.save_table(dtm_simple, 'dtm_simple', user_session_id)
	
						elif dtm_type == 'Normalized' and scale == 'Yes':
							dtm_pos = _analysis.tf_proportions(dtm_pos, norm=False, scale=True)
							dtm_ds  = _analysis.tf_proportions(dtm_ds, norm=False, scale=True)
							units = 'scaled'
	
						else:
							dtm_pos = _analysis.tfidf(dtm_pos)
							dtm_ds = _analysis.tfidf(dtm_ds)
							units = 'tfidf'

					dtm_ds.drop('Untagged', axis=1, inplace=True, errors='ignore')
					_handlers.save_table(dtm_pos, 'dtm_pos', user_session_id)
					_handlers.save_table(dtm_ds, 'dtm_ds', user_session_id)
					_handlers.update_dtm(sums_pos, sums_ds, units, user_session_id)
					st.success('DTM generated!')
					st.experimental_rerun()
		
		st.sidebar.markdown("---")
				
if __name__ == "__main__":
    main()
    
				