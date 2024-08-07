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
import pathlib
import polars as pl
import streamlit as st
import streamlit.components.v1 as components

import categories as _categories
import states as _states
from utilities import analysis_functions as _analysis
from utilities import handlers_database as _handlers
from utilities import messages as _messages
from utilities import warnings as _warnings

hex_highlights = ['#5fb7ca', '#e35be5', '#ffc701', '#fe5b05', '#cb7d60']
    
CATEGORY = _categories.OTHER
TITLE = "Single Documents"
KEY_SORT = 10

def main():

	user_session = st.runtime.scriptrunner.script_run_context.get_script_run_ctx()
	user_session_id = user_session.session_id

	if user_session_id not in st.session_state:
		st.session_state[user_session_id] = {}
	try:
		con = st.session_state[user_session_id]["ibis_conn"]
	except:
		con = _handlers.get_db_connection(user_session_id)
		_handlers.generate_temp(_states.STATES.items(), user_session_id, con)

	try:
		session = pl.DataFrame.to_dict(con.table("session").to_polars(), as_series=False)
	except:
		_handlers.init_session(con)
		session = pl.DataFrame.to_dict(con.table("session").to_polars(), as_series=False)

	if session.get('doc')[0] == True:
	
		_handlers.load_widget_state(pathlib.Path(__file__).stem, user_session_id)
		metadata_target = _handlers.load_metadata('target', con)

		st.sidebar.markdown("### Tagset")

		st.sidebar.markdown("Use the menus to select up to **5 tags** you would like to highlight.")
		
		with st.sidebar.expander("About general tags"):
			st.markdown(_messages.message_general_tags)		

		tag_radio = st.sidebar.radio("Select tags to display:", ("Parts-of-Speech", "DocuScope"), key = _handlers.persist("sd_radio", pathlib.Path(__file__).stem, user_session_id), horizontal=True)
	
		if tag_radio == 'Parts-of-Speech':
			tag_type = st.sidebar.radio("Select from general or specific tags", ("General", "Specific"), horizontal=True)
			if tag_type == 'General':
				tag_loc = con.table("doc_simple", database="target").to_polars()
				html_simple = ''.join(tag_loc.get_column("Text").to_list())
				doc_key = tag_loc.get_column("doc_id").unique().to_list()

				tag_list = st.sidebar.multiselect('Select tags to highlight', ['Adjective', 'Adverb', 'Conjunction', 'NounCommon', 'NounOther', 'Preposition', 'Pronoun', 'VerbBe', 'VerbLex', 'VerbOther'], on_change = _handlers.update_tags(html_simple, user_session_id), key=f"tags_{user_session_id}")
				tag_colors = hex_highlights[:len(tag_list)]
				tag_html = zip(tag_colors, tag_list)
				tag_html = list(map('">'.join, tag_html))
				tag_html = ['<span style="background-color: '+ item + '</span>' for item in tag_html]
				tag_html = '; '.join(tag_html)
				df = (tag_loc
					.filter(pl.col("Tag") != "Other")
					.group_by("Tag").len("AF")
					.with_columns(pl.col("AF").truediv(pl.sum("AF")).mul(100).alias("RF"))
					.sort(["AF", "Tag"], descending=[True, False])
					).to_pandas()
			else:
				tag_loc = con.table("doc_pos", database="target").to_polars()
				html_pos = ''.join(tag_loc.get_column("Text").to_list())
				doc_key = tag_loc.get_column("doc_id").unique().to_list()
				
				tag_list = st.sidebar.multiselect('Select tags to highlight', metadata_target.get('tags_pos')[0]['tags'], on_change = _handlers.update_tags(html_pos, user_session_id), key=f"tags_{user_session_id}")
				tag_colors = hex_highlights[:len(tag_list)]
				tag_html = zip(tag_colors, tag_list)
				tag_html = list(map('">'.join, tag_html))
				tag_html = ['<span style="background-color: '+ item + '</span>' for item in tag_html]
				tag_html = '; '.join(tag_html)
				df = (tag_loc
					.filter(pl.col("Tag") != "Y")
					.group_by("Tag").len("AF")
					.with_columns(pl.col("AF").truediv(pl.sum("AF")).mul(100).alias("RF"))
					.sort(["AF", "Tag"], descending=[True, False])
					).to_pandas()
		else:
			tag_loc = con.table("doc_ds", database="target").to_polars()
			html_ds = ''.join(tag_loc.get_column("Text").to_list())
			doc_key = tag_loc.get_column("doc_id").unique().to_list()[0]

			tag_list = st.sidebar.multiselect('Select tags to highlight', metadata_target.get('tags_ds')[0]['tags'], on_change = _handlers.update_tags(html_ds, user_session_id), key=f"tags_{user_session_id}")
			tag_colors = hex_highlights[:len(tag_list)]
			tag_html = zip(tag_colors, tag_list)
			tag_html = list(map('">'.join, tag_html))
			tag_html = ['<span style="background-color: '+ item + '</span>' for item in tag_html]
			tag_html = '; '.join(tag_html)
			df = (tag_loc
				.filter(pl.col("Tag") != "Untagged")
				.group_by("Tag").len("AF")
				.with_columns(pl.col("AF").truediv(pl.sum("AF")).mul(100).alias("RF"))
				.sort(["AF", "Tag"], descending=[True, False])
				).to_pandas()
		
		if len(tag_list) == 5:
			st.sidebar.markdown(':warning: You can hightlight a maximum of 5 tags.')

		st.sidebar.markdown("---")
		st.sidebar.markdown("### Plot tag locations")
		with st.sidebar.expander("Plot explanation"):
			st.write("""
					The plot(s) shows lines segment where tags occur in what might be called 'normalized text time.'
					For example, if you had a text 100 tokens long and a tag occurred at the 10th, 25th, and 60th token,
					the plot would show lines at 10%, 25%, and 60% along the x-axis.
					""")
		
		st.markdown(f"""
					###  {doc_key[0]}
					""")

		if st.sidebar.button("Tag Density Plot"):
			if len(tag_list) > 5:
				st.write(':no_entry_sign: You can only plot a maximum of 5 tags.')
			elif len(tag_list) == 0:
				st.write('There are no tags to plot.')
			else:
				plot_colors = tag_html.replace('<span style="background-color: ', '')
				plot_colors = plot_colors.replace('</span>', '')
				plot_colors = plot_colors.replace('">', '; ')
				plot_colors = plot_colors.split("; ")
				plot_colors = list(zip(plot_colors[1::2], plot_colors[::2]))			
				plot_colors = pd.DataFrame(plot_colors, columns=['Tag', 'Color'])
				plot_colors = plot_colors.sort_values(by=['Tag'])
				plot_colors = plot_colors['Color'].unique()
				
				df_plot = tag_loc.to_pandas()
				df_plot['X'] = (df_plot.index + 1)/(len(df_plot.index))
				df_plot = df_plot[df_plot['Tag'].isin(tag_list)]
				
				base = alt.Chart(df_plot, height={"step": 45}).mark_tick(size=35).encode(
					x=alt.X('X:Q', axis=alt.Axis(values=[0, .25, .5, .75, 1], format='%'), title=None),
					y=alt.Y('Tag:O', title = None, sort=tag_list)
					)
				
				lex_density = base.encode(
					color=alt.Color('Tag', scale=alt.Scale(range=plot_colors), legend=None),
				)

				st.altair_chart(lex_density, use_container_width=True)
		
		st.markdown(f"""
					##### Tags:  {tag_html}
					""", unsafe_allow_html=True)
						
		if 'html_str' not in st.session_state[user_session_id]:
			st.session_state[user_session_id]['html_str'] = ''
		
		components.html(st.session_state[user_session_id]['html_str'], height=500, scrolling=True)
		
		st.dataframe(df, hide_index=True)
		
		st.sidebar.markdown("---")
		with st.sidebar:
			download_file = _handlers.convert_to_word(st.session_state[user_session_id]['html_str'],
											 tag_html,
											 doc_key,
											 df)

			st.download_button(
    			label="Download to Word",
    			data=download_file,
    			file_name="document_tags.docx",
   					 mime="docx",
					)
		
		st.sidebar.markdown("---")
		
		st.sidebar.markdown("### Reset document")
		st.sidebar.markdown("""
							Click the button to explore a new document.
							""")
		if st.sidebar.button("Select a new document"):
			_TAGS = f"tags_{user_session_id}"
			try:
				con.drop_table("doc_simple", database="target")
			except:
				pass
			try:
				con.drop_table("doc_pos", database="target")
			except:
				pass
			try:
				con.drop_table("doc_ds", database="target")
			except:
				pass
			_handlers.update_session('doc', False, con)
			if _TAGS in st.session_state:
				del st.session_state[_TAGS]
			st.rerun()

		st.sidebar.markdown("---")
			
	else:
		
		st.markdown(_messages.message_single_document)
		
		try:
			metadata_target = _handlers.load_metadata('target', con)
		except:
			pass
		
		st.sidebar.markdown("### Choose document")
		st.sidebar.write("Use the menus to select the tags you would like to highlight.")		

		if session.get('has_target')[0] == True:
			doc_key = st.sidebar.selectbox("Select document to view:", (sorted(metadata_target.get('docids')[0]['ids'])))
		else:
			doc_key = st.sidebar.selectbox("Select document to view:", (['No documents to view']))

		if st.sidebar.button("Process Document"):
			if session.get('has_target')[0] == False:
				st.markdown(_warnings.warning_11, unsafe_allow_html=True)
			else:
				tok_pl = con.table("ds_tokens", database="target").to_pyarrow_batches(chunk_size=5000)
				tok_pl = pl.from_arrow(tok_pl)

				doc_pos, doc_simple, doc_ds = _analysis.html_build_pl(tok_pl, doc_key)
				
				con.create_table("doc_pos", obj=doc_pos, database="target", overwrite=True)
				con.create_table("doc_simple", obj=doc_simple, database="target", overwrite=True)
				con.create_table("doc_ds", obj=doc_ds, database="target", overwrite=True)
				_handlers.update_session('doc', True, con)
				
				st.rerun()

if __name__ == "__main__":
    main()