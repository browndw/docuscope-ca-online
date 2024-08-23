# Copyright (C) 2024 David West Brown

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import altair as alt
import pandas as pd
import polars as pl
import streamlit as st

import categories as _categories
import states as _states
from utilities import analysis_functions as _analysis
from utilities import handlers_database as _handlers
from utilities import messages as _messages
from utilities import warnings as _warnings

CATEGORY = _categories.OTHER
TITLE = "Advanced Plotting"
KEY_SORT = 9

def main():
	
	user_session = st.runtime.scriptrunner.script_run_context.get_script_run_ctx()
	user_session_id = user_session.session_id

	if user_session_id not in st.session_state:
		st.session_state[user_session_id] = {}
	try:
		session = pl.DataFrame.to_dict(st.session_state[user_session_id]["session"], as_series=False)
	except:
		_handlers.init_session(user_session_id)
		session = pl.DataFrame.to_dict(st.session_state[user_session_id]["session"], as_series=False)
	
	try:
		metadata_target = _handlers.load_metadata('target', user_session_id)
	except:
		pass

	st.markdown(_messages.message_plotting)
		
	plot_type = st.radio("What kind of plot would you like to make?",
					["Boxplot", "Scatterplot", "PCA"],
					captions=[":package: Boxplots of normalized tag frequencies with grouping variables (if you've processed corpus metadata).",
					":sparkles: Scatterplots of normalized tag frequencies with grouping variables (if you've processed corpus metadata).",
					":triangular_ruler: Principal component analysis from scaled tag frequences with highlighting for groups (if you've processed corpus metadata)."], 
					horizontal=False,
					index=None)

	
	if  plot_type == "Boxplot"and session.get('has_target')[0] == True:
		
		_handlers.update_session('pca', False, user_session_id)
		st.sidebar.markdown("### Tagset")
		
		with st.sidebar.expander("About general tags"):
			st.markdown(_messages.message_general_tags)		

		tag_radio_tokens = st.sidebar.radio("Select tags to display:", ("Parts-of-Speech", "DocuScope"), on_change=_handlers.clear_plots, args=(user_session_id,), horizontal=True)

		if tag_radio_tokens == 'Parts-of-Speech':
			tag_type = st.sidebar.radio("Select from general or specific tags", ("General", "Specific"), on_change=_handlers.clear_plots, args=(user_session_id,), horizontal=True)
			if tag_type == 'General':
				df = st.session_state[user_session_id]["target"]["dtm_pos"]
				df = _analysis.dtm_simplify_pl(df)

			elif tag_type == 'Specific':
				df = st.session_state[user_session_id]["target"]["dtm_pos"]
		
		elif tag_radio_tokens == 'DocuScope':
			df = st.session_state[user_session_id]["target"]["dtm_ds"]
			tag_type = None
		
		st.sidebar.markdown("---")

		st.markdown("""---""")

		by_group = st.toggle("Plot using grouping variables.")
		
		if df.height == 0 or df is None:
			cats = []
		elif df.height > 0:
			to_drop = ['doc_id','Other','FU','Untagged']
			cats = sorted(list(df.drop([x for x in to_drop if x in df.columns]).columns))

		if by_group:
			if session['has_meta'][0] == False:
				st.markdown(_warnings.warning_21, unsafe_allow_html=True)
			
			else:
				st.sidebar.markdown("### Variables")
				box_vals = st.sidebar.multiselect("Select variables for plotting:", (cats))

				st.sidebar.markdown("---")
				st.sidebar.markdown('### Grouping variables')
				st.sidebar.markdown("Select grouping variables from your metadata and click the button to generate boxplots of frequencies.")
				st.sidebar.markdown('#### Group A')
				st.session_state[user_session_id]['grpa'] = st.sidebar.multiselect("Select categories for group A:", (sorted(set(metadata_target.get('doccats')[0]['cats']))), on_change = _handlers.update_grpa(user_session_id), key=f"grpa_{user_session_id}")
				
				st.sidebar.markdown('#### Group B')
				st.session_state[user_session_id]['grpb'] = st.sidebar.multiselect("Select categories for group B:", (sorted(set(metadata_target.get('doccats')[0]['cats']))), on_change = _handlers.update_grpb(user_session_id), key=f"grpb_{user_session_id}")
				
				if st.sidebar.button("Boxplots of Frequencies by Group"):
					#clear any pca data
					#_handlers.update_session('pca', dict())

					if len(box_vals) == 0:
						st.markdown(_warnings.warning_19, unsafe_allow_html=True)
					
					elif len(box_vals) > 0:
						grpa_list = list(st.session_state[user_session_id]['grpa'])
						grpb_list = list(st.session_state[user_session_id]['grpb'])
						
						if len(grpb_list) == 0 or len(grpa_list) == 0:
							st.markdown(_warnings.warning_20, unsafe_allow_html=True)
						
						else:
							df_plot = _analysis.dtm_weight_pl(df)
							df_plot = _analysis.boxplots_pl(df_plot, box_vals, grp_a = grpa_list, grp_b = grpb_list)

							plot = alt.Chart(df_plot.to_pandas()).mark_boxplot(ticks=True).encode(
								alt.X('RF', title='Frequency (per 100 tokens)'),
								alt.Y('Group', title='', axis=alt.Axis(labels=False, ticks=False)),
								alt.Color('Group', scale=alt.Scale(scheme='category10')),
									row=alt.Row('Tag', title='', header=alt.Header(orient='left', labelAngle=0, labelAlign='left'), sort=alt.SortField(field='Median', order='descending'))
									).configure_facet(
									spacing=10
									).configure_view(
									stroke=None
								).configure_legend(orient='top', direction='vertical')
							
							st.markdown(_messages.message_disable_full, unsafe_allow_html=True)
							st.altair_chart(plot)

							stats = (df_plot
									.group_by(["Group", "Tag"])
									.agg(
										pl.len().alias("count"),
										pl.col("RF").mean().alias("mean"),
										pl.col("RF").median().alias("median"),
										pl.col("RF").std().alias("std"),
										pl.col("RF").min().alias("min"),
										pl.col("RF").quantile(0.25).alias("25%"),
										pl.col("RF").quantile(0.5).alias("50%"),
										pl.col("RF").quantile(0.75).alias("75%"),
										pl.col("RF").max().alias("max")
										)
									.sort(["Tag", "Group"])
									)			
														
							st.markdown("##### Descriptive statistics:")
							st.dataframe(stats, hide_index=True)

				st.sidebar.markdown("---")
		
		else:
			st.sidebar.markdown("### Variables")
			box_vals = st.sidebar.multiselect("Select variables for plotting:", (cats))
			if st.sidebar.button("Boxplots of Frequencies"):
				#clear any pca data
				#_handlers.update_session('pca', dict())

				if len(box_vals) == 0:
					st.markdown(_warnings.warning_18, unsafe_allow_html=True)
				
				elif len(box_vals) > 0:
					df_plot = _analysis.dtm_weight_pl(df)
					df_plot = _analysis.boxplots_pl(df_plot, box_vals, grp_a = None, grp_b = None)
						
					base = alt.Chart(df_plot.to_pandas()).mark_boxplot(ticks=True).encode(
						alt.Color(scale=alt.Scale(scheme='category10')),
						x = alt.X('RF', title='Frequency (per 100 tokens)'),
						row=alt.Row('Tag', title='', header=alt.Header(orient='left', labelAngle=0, labelAlign='left'), sort=alt.SortField(field='Median', order='descending'))
						).configure_facet(
									spacing=10
									).configure_view(
									stroke=None
								)
						
					st.markdown(_messages.message_disable_full, unsafe_allow_html=True)
					st.altair_chart(base)

					stats = (df_plot
			  					.group_by(["Tag"])
								.agg(
									pl.len().alias("count"),
									pl.col("RF").mean().alias("mean"),
									pl.col("RF").median().alias("median"),
									pl.col("RF").std().alias("std"),
									pl.col("RF").min().alias("min"),
									pl.col("RF").quantile(0.25).alias("25%"),
									pl.col("RF").quantile(0.5).alias("50%"),
									pl.col("RF").quantile(0.75).alias("75%"),
									pl.col("RF").max().alias("max")
									)
								.sort("Tag")
								)
					
					st.markdown("##### Descriptive statistics:")
					st.dataframe(stats, hide_index=True)

		st.sidebar.markdown("---")
	
	elif plot_type == "Scatterplot" and session.get('has_target')[0] == True:

		_handlers.update_session('pca', False, user_session_id)		
		st.sidebar.markdown("### Tagset")
		
		with st.sidebar.expander("About general tags"):
			st.markdown(_messages.message_general_tags)		

		tag_radio_tokens = st.sidebar.radio("Select tags to display:", ("Parts-of-Speech", "DocuScope"), on_change=_handlers.clear_plots, args=(user_session_id,), horizontal=True)
		
		if tag_radio_tokens == 'Parts-of-Speech':
			tag_type = st.sidebar.radio("Select from general or specific tags", ("General", "Specific"), on_change=_handlers.clear_plots, args=(user_session_id,), horizontal=True)
			if tag_type == 'General':
				df = st.session_state[user_session_id]["target"]["dtm_pos"]
				df = _analysis.dtm_simplify_pl(df)

			elif tag_type == 'Specific':
				df = st.session_state[user_session_id]["target"]["dtm_pos"]
		
		elif tag_radio_tokens == 'DocuScope':
			df = st.session_state[user_session_id]["target"]["dtm_ds"]
			tag_type = None
		
		st.sidebar.markdown("---")
		
		st.markdown("""---""")
		by_group_highlight = st.toggle("Hightlight groups in scatterplots.")
		
		if df.height == 0 or df is None:
			cats = []
		elif df.height > 0:
			to_drop = ['doc_id','Other','FU','Untagged']
			cats = sorted(list(df.drop([x for x in to_drop if x in df.columns]).columns))

		if by_group_highlight:
			if session['has_meta'][0] == False:
				st.markdown(_warnings.warning_21, unsafe_allow_html=True)
			
			else:
				st.sidebar.markdown("### Variables")

				xaxis = st.sidebar.selectbox("Select variable for the x-axis", (cats))
				yaxis = st.sidebar.selectbox("Select variable for the y-axis", (cats))
	
				if st.sidebar.button("Scatterplot of Frequencies"):

					df_plot = _analysis.dtm_weight_pl(df).with_columns(pl.selectors.numeric().mul(100))
					df_plot = df_plot.with_columns(pl.col("doc_id").str.split_exact("_", 0).struct.rename_fields(["Group"]).alias("id")).unnest("id").to_pandas()

					x_label = xaxis + ' ' + '(per 100 tokens)'
					y_label = yaxis + ' ' + '(per 100 tokens)'

					base = alt.Chart(df_plot).mark_circle(size=50, opacity=.75).encode(
						alt.X(xaxis, title=x_label),
						alt.Y(yaxis, title = y_label),
						tooltip=['doc_id:N']
					)

					groups = sorted(set(metadata_target.get('doccats')[0]['cats']))
					# A dropdown filter
					group_dropdown = alt.binding_select(options=groups)
					group_select = alt.selection_single(fields=['Group'], bind=group_dropdown, name="Select")
					group_color_condition = alt.condition(group_select,
							alt.Color('Group:N', legend=None, scale=alt.Scale(range=['#133955'])),
							alt.value('lightgray'))
					
					highlight_groups = base.add_selection(group_select).encode(color=group_color_condition)
					st.altair_chart(highlight_groups)					
									
					cc_df, cc_r, cc_p = _analysis.correlation(df_plot, xaxis, yaxis)
					
					st.markdown(_messages.message_correlation_info(cc_df, cc_r, cc_p))

				st.sidebar.markdown("---")
		else:
			st.sidebar.markdown("### Variables")
			xaxis = st.sidebar.selectbox("Select variable for the x-axis", (cats))
			yaxis = st.sidebar.selectbox("Select variable for the y-axis", (cats))
	
			if st.sidebar.button("Scatterplot of Frequencies"):

				df_plot = _analysis.dtm_weight_pl(df).with_columns(pl.selectors.numeric().mul(100)).to_pandas()
				
				x_label = xaxis + ' ' + '(per 100 tokens)'
				y_label = yaxis + ' ' + '(per 100 tokens)'
	
				base = alt.Chart(df_plot).mark_circle().encode(
	    			alt.X(xaxis, title=x_label),
	    			alt.Y(yaxis, title = y_label),
	    			tooltip=['doc_id:N']
				)
				
				st.altair_chart(base)
				
				cc_df, cc_r, cc_p = _analysis.correlation(df_plot, xaxis, yaxis)
				
				st.markdown(_messages.message_correlation_info(cc_df, cc_r, cc_p))
			
		st.sidebar.markdown("---")
	
	elif plot_type == "PCA" and session.get('has_target')[0] == True:

		st.sidebar.markdown("### Tagset")
		
		with st.sidebar.expander("About general tags"):
			st.markdown(_messages.message_general_tags)		

		tag_radio_tokens = st.sidebar.radio("Select tags to display:", ("Parts-of-Speech", "DocuScope"), on_change=_handlers.clear_plots, args=(user_session_id,), horizontal=True)

		if tag_radio_tokens == 'Parts-of-Speech':
			tag_type = st.sidebar.radio("Select from general or specific tags", ("General", "Specific"), on_change=_handlers.clear_plots, args=(user_session_id,), horizontal=True)
			if tag_type == 'General':
				df = st.session_state[user_session_id]["target"]["dtm_pos"]
				df = _analysis.dtm_simplify_pl(df)
				df = _analysis.dtm_weight_pl(df, scheme="prop")
				df = _analysis.dtm_weight_pl(df, scheme="scale").to_pandas()

			elif tag_type == 'Specific':
				df = st.session_state[user_session_id]["target"]["dtm_pos"]
				df = _analysis.dtm_weight_pl(df, scheme="prop")
				df = _analysis.dtm_weight_pl(df, scheme="scale").to_pandas()
		
		elif tag_radio_tokens == 'DocuScope':
			df = st.session_state[user_session_id]["target"]["dtm_ds"]
			df = _analysis.dtm_weight_pl(df, scheme="prop")
			df = _analysis.dtm_weight_pl(df, scheme="scale").to_pandas()
			tag_type = None

			st.sidebar.markdown("""---""") 
			st.sidebar.markdown("### Principal Component Analysis")
			st.sidebar.markdown("""
								Click the button to plot principal compenents.
								""")
			
		st.markdown("---")

		if st.sidebar.button("PCA"):
			_handlers.update_session('pca', False, user_session_id)
			if session.get('has_meta')[0] == True:
				grouping = metadata_target.get('doccats')[0]['cats']
			else:
				grouping = []

			to_drop = ['Other','FU','Untagged']
			df = df.drop([x for x in to_drop if x in df.columns], axis=1)
			pca_df, contrib_df, ve = _analysis.pca_contributions(df, grouping)
			
			if "pca_df" not in st.session_state[user_session_id]["target"]:
				st.session_state[user_session_id]["target"]["pca_df"] = {}
			st.session_state[user_session_id]["target"]["pca_df"] = pca_df
			if "contrib_df" not in st.session_state[user_session_id]["target"]:
				st.session_state[user_session_id]["target"]["contrib_df"] = {}
			st.session_state[user_session_id]["target"]["contrib_df"] = contrib_df
			_handlers.update_metadata('target', 'variance', ve, user_session_id)
			_handlers.update_session('pca', True, user_session_id)
			st.rerun()	

		if session.get('pca')[0] == True:
			pca_df = st.session_state[user_session_id]["target"]["pca_df"]
			contrib_df = st.session_state[user_session_id]["target"]["contrib_df"]
			ve = metadata_target.get("variance")[0]['temp']
		
			st.session_state[user_session_id]['pca_idx'] = st.sidebar.selectbox("Select principal component to plot ", (list(range(1, len(df.columns)))))
			
			cp_1, cp_2, pca_x, pca_y, contrib_x, contrib_y, ve_1, ve_2 = _analysis.update_pca_plot(pca_df, contrib_df, ve, int(st.session_state[user_session_id]['pca_idx']))
			
			base = alt.Chart(pca_df).mark_circle(size=50, opacity=.75).encode(
					alt.X(pca_x),
					alt.Y(pca_y),
					tooltip=['doc_id:N']
					)

			#zero axes
			line_y = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule().encode(y=alt.Y('y', title=pca_y))
			line_x = alt.Chart(pd.DataFrame({'x': [0]})).mark_rule().encode(x=alt.X('x', title=pca_x))

			if session.get('has_meta')[0] == True:
				groups = sorted(set(metadata_target.get('doccats')[0]['cats']))
				# A dropdown filter
				group_dropdown = alt.binding_select(options=groups)
				group_select = alt.selection_single(fields=['Group'], bind=group_dropdown, name="Select")
				group_color_condition = alt.condition(group_select,
						alt.Color('Group:N', legend=None, scale=alt.Scale(range=['#133955'])),
						alt.value('lightgray'))
				
				highlight_groups = base.add_selection(group_select).encode(color=group_color_condition)
				st.altair_chart(highlight_groups + line_y + line_x)

			else:
				st.altair_chart(base + line_y + line_x)
			
			st.markdown(_messages.message_variance_info(pca_x, pca_y, ve_1, ve_2))
			
			st.markdown(_messages.message_contribution_info(pca_x, pca_y, contrib_x, contrib_y))
			
			st.markdown("##### Variable contribution (by %) to principal component:")
			
			with st.expander("About variable contribution"):
				st.markdown(_messages.message_variable_contrib)


			col1,col2 = st.columns(2)
			col1.altair_chart(cp_1, use_container_width = True)
			col2.altair_chart(cp_2, use_container_width = True)

		st.sidebar.markdown("---")
				
if __name__ == "__main__":
    main()
    
				