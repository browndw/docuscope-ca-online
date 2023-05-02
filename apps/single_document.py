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

modules = ['analysis', 'categories', 'handlers', 'messages', 'states', 'warnings', 'streamlit', 'altair', 'docx', 'docuscospacy', 'pandas']
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
    
hex_highlights = ['#5fb7ca', '#e35be5', '#ffc701', '#fe5b05', '#cb7d60']
    
CATEGORY = _categories.OTHER
TITLE = "Single Documents"
KEY_SORT = 10

def main():

	session = _handlers.load_session()	

	if bool(session['doc']) == True:
	
		_handlers.load_widget_state(pathlib.Path(__file__).stem)
		metadata_target = _handlers.load_metadata('target')

		st.sidebar.markdown("### Tagset")

		st.sidebar.markdown("Use the menus to select up to **5 tags** you would like to highlight.")
		
		with st.sidebar.expander("About general tags"):
			st.markdown(_messages.message_general_tags)		

		tag_radio = st.sidebar.radio("Select tags to display:", ("Parts-of-Speech", "DocuScope"), key = _handlers.persist("sd_radio", pathlib.Path(__file__).stem), horizontal=True)
	
		if tag_radio == 'Parts-of-Speech':
			tag_type = st.sidebar.radio("Select from general or specific tags", ("General", "Specific"), horizontal=True)
			if tag_type == 'General':
				tag_list = st.sidebar.multiselect('Select tags to highlight', ['Noun', 'Verb', 'Adjective', 'Adverb', 'Pronoun', 'Preposition', 'Conjunction'], on_change = _handlers.update_tags(session['doc']['html_simple']), key='tags')
				tag_colors = hex_highlights[:len(tag_list)]
				tag_html = zip(tag_colors, tag_list)
				tag_html = list(map('">'.join, tag_html))
				tag_html = ['<span style="background-color: '+ item + '</span>' for item in tag_html]
				tag_html = '; '.join(tag_html)
				tag_loc = _handlers.load_table('doc_simple')
				df = session['doc']['dc_simple']
			else:
				tag_list = st.sidebar.multiselect('Select tags to highlight', metadata_target['tags_pos'], on_change = _handlers.update_tags(session['doc']['html_pos']), key='tags')
				tag_colors = hex_highlights[:len(tag_list)]
				tag_html = zip(tag_colors, tag_list)
				tag_html = list(map('">'.join, tag_html))
				tag_html = ['<span style="background-color: '+ item + '</span>' for item in tag_html]
				tag_html = '; '.join(tag_html)
				tag_loc = _handlers.load_table('doc_pos')
				df = session['doc']['dc_pos']
		else:
			tag_list = st.sidebar.multiselect('Select tags to highlight', metadata_target['tags_ds'], on_change = _handlers.update_tags(session['doc']['html_ds']), key='tags')
			tag_colors = hex_highlights[:len(tag_list)]
			tag_html = zip(tag_colors, tag_list)
			tag_html = list(map('">'.join, tag_html))
			tag_html = ['<span style="background-color: '+ item + '</span>' for item in tag_html]
			tag_html = '; '.join(tag_html)
			tag_loc = _handlers.load_table('doc_ds')
			df = session['doc']['dc_ds']
		
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
					###  {session['doc']['doc_key']}
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
				
				df_plot = tag_loc.copy()
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
						
		if 'html_str' not in st.session_state:
			st.session_state['html_str'] = ''
		
		st.components.v1.html(st.session_state.html_str, height=500, scrolling=True)
		
		st.dataframe(df)
		
		st.sidebar.markdown("---")
		st.sidebar.markdown(_messages.message_download_dtm)
		
		if st.sidebar.button("Download"):
			with st.sidebar:
				with st.spinner('Creating download link...'):
					doc_html = st.session_state.html_str.split('</style>')
					style_sheet_str = doc_html[0] + '</style>'
					html_str = doc_html[1]
					doc_html = '<!DOCTYPE html><html><head>' + style_sheet_str + '</head><body>' + tag_html + '<br><br>' + html_str + '</body></html>'
					downloaded_file = docx.Document()
					downloaded_file.add_heading(session['doc']['doc_key'])
					downloaded_file.add_heading('Table of tag frequencies:', 3)
					#add counts table
					df['RF'] = df.RF.round(2)
					t = downloaded_file.add_table(df.shape[0]+1, df.shape[1])
					# add the header rows.
					for j in range(df.shape[-1]):
						t.cell(0,j).text = df.columns[j]
					# add the rest of the data frame
					for i in range(df.shape[0]):
						for j in range(df.shape[-1]):
							t.cell(i+1,j).text = str(df.values[i,j])
					t.style = 'LightShading-Accent1'
					downloaded_file.add_heading('Highlighted tags:', 3)
					#add html
					_analysis.add_alt_chunk(downloaded_file, doc_html)
					towrite = BytesIO()
					downloaded_file.save(towrite)
					towrite.seek(0)  # reset pointer
					b64 = base64.b64encode(towrite.read()).decode()
					st.success('Link generated!')
					linko= f'<a href="data:vnd.openxmlformats-officedocument.wordprocessingml.document;base64,{b64}" download="document_tags.docx">Download Word file</a>'
					st.markdown(linko, unsafe_allow_html=True)
		
		st.sidebar.markdown("---")
		
		st.sidebar.markdown("### Reset document")
		st.sidebar.markdown("""
							Click the button to explore a new document.
							""")
		if st.sidebar.button("Select a new document"):
			_handlers.clear_table('doc_simple')
			_handlers.clear_table('doc_pos')
			_handlers.clear_table('doc_ds')
			_handlers.update_session('doc', dict())
			if 'tags' in st.session_state:
				del st.session_state['tags']
			st.experimental_rerun()

		st.sidebar.markdown("---")
			
	else:
		
		st.markdown(_messages.message_single_document)
		
		try:
			metadata_target = _handlers.load_metadata('target')
		except:
			metadata_target = {}
		
		st.sidebar.markdown("### Choose document")
		st.sidebar.write("Use the menus to select the tags you would like to highlight.")		

		if 'docids' in metadata_target.keys():
			doc_key = st.sidebar.selectbox("Select document to view:", (sorted(metadata_target['docids'])))
		else:
			doc_key = st.sidebar.selectbox("Select document to view:", (['No documents to view']))

		if st.sidebar.button("Process Document"):
			if session.get('target_path') == None:
				st.markdown(_warnings.warning_11, unsafe_allow_html=True)
			else:
				tp = _handlers.load_corpus_session('target', session)
				doc_pos = ds.tag_ruler(tp, doc_key, count_by='pos')
				doc_simple = _analysis.simplify_span(doc_pos)
				doc_ds = ds.tag_ruler(tp, doc_key, count_by='ds')

				doc_tokens = len(doc_pos.index)
				doc_words = len(doc_pos[doc_pos.Tag != 'Y'])
				dc_pos = _analysis.doc_counts(doc_pos, doc_words, count_by='pos')
				dc_simple = _analysis.simplify_counts(dc_pos, doc_words)
				dc_ds = _analysis.doc_counts(doc_ds, doc_tokens, count_by='ds')

				html_pos = _analysis.html_build(tp, doc_key, count_by='pos')
				html_simple = _analysis.html_simplify(html_pos)
				html_ds = _analysis.html_build(tp, doc_key, count_by='ds')
				
				_handlers.save_table(doc_pos, 'doc_pos')
				_handlers.save_table(doc_simple, 'doc_simple')
				_handlers.save_table(doc_ds, 'doc_ds')

				_handlers.update_doc(dc_pos, dc_simple, dc_ds, html_pos, html_simple, html_ds, doc_key)
				
				st.experimental_rerun()

if __name__ == "__main__":
    main()