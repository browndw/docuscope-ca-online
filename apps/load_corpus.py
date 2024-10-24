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

from collections import Counter
from lingua import Language, LanguageDetectorBuilder
import pathlib
import polars as pl
import spacy
import streamlit as st

import categories as _categories
import states as _states
from utilities import analysis_functions as _analysis
from utilities import handlers_database as _handlers
from utilities import handlers_imports as _imports
from utilities import messages as _messages
from utilities import process_corpus as _process
from utilities import warnings as _warnings

# set paths
HERE = pathlib.Path(__file__).parents[1].resolve()
MODEL_LARGE = str(HERE.joinpath("models/en_docusco_spacy"))
MODEL_SMALL = str(HERE.joinpath("models/en_docusco_spacy_cd"))
OPTIONS = str(HERE.joinpath("options.toml"))

# import options
_options = _imports.import_options_general(OPTIONS)

# set options
ENABLE_DETECT = _options['global']['check_language']
DESKTOP = _options['global']['desktop_mode']
CHECK_SIZE = _options['global']['check_size']

if CHECK_SIZE == True:
	MAX_TEXT = _options['global']['max_bytes_text']
	MAX_POLARS = _options['global']['max_bytes_polars']

CATEGORY = _categories.CORPUS_LOAD
TITLE = "Manage Corpus Data"
KEY_SORT = 1

@st.cache_resource(show_spinner=False)
def load_detector():
	detector = LanguageDetectorBuilder.from_all_languages().with_low_accuracy_mode().build()
	return(detector)

@st.cache_resource(show_spinner=False)
def load_models():
	large_model = spacy.load(MODEL_LARGE)
	small_model = spacy.load(MODEL_SMALL)
	models = {"Large Dictionary": large_model, "Common Dictionary": small_model}
	return models

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
			
	if 'warning' not in st.session_state[user_session_id]:
		st.session_state[user_session_id]['warning'] = 0

	if session.get('has_target')[0] == True:
		metadata_target = st.session_state[user_session_id]['metadata_target'].to_dict()

		if session.get('has_reference')[0] == True:
			metadata_reference = st.session_state[user_session_id]['metadata_reference'].to_dict()
		
		st.markdown(_messages.message_target_info(metadata_target))
				
		if st.session_state[user_session_id]['warning'] == 40:
			st.markdown(_warnings._40_excluded_files_target(st.session_state[user_session_id]['exceptions']))
		
		with st.expander("Documents:"):
			st.write(metadata_target.get('docids')[0]['ids'])
		
		if session.get('has_meta')[0] == True:
			st.markdown('##### Target corpus metadata:')
			with st.expander("Counts of document categories:"):
				st.write(Counter(metadata_target.get('doccats')[0]['cats']))
		
		else:
			st.sidebar.markdown('### Target corpus metadata:')
			load_cats = st.sidebar.radio("Do you have categories in your file names to process?", ("No", "Yes"), horizontal=True)
			if load_cats == 'Yes':
				if st.sidebar.button("Process Document Metadata"):
					with st.spinner('Processing metadata...'):
						doc_cats = _process.get_doc_cats(metadata_target.get('docids')[0]['ids'])
						if len(set(doc_cats)) > 1 and len(set(doc_cats)) < 21:
							_handlers.update_metadata('target', 'doccats', doc_cats, user_session_id)
							_handlers.update_session('has_meta', True, user_session_id)
							st.success('Processing complete!')
							st.rerun()
						elif len(doc_cats) != 0:
							st.sidebar.markdown(_warnings._50_categories_number, unsafe_allow_html=True)
						else:
							st.sidebar.markdown(_warnings._51_categories_format, unsafe_allow_html=True)
			st.sidebar.markdown("---")
				
		if session.get('has_reference')[0] == True:
			metadata_reference = _handlers.load_metadata('reference', user_session_id)
			
			st.markdown(_messages.message_reference_info(metadata_reference))
			
			if st.session_state[user_session_id]['warning'] == 41:
				st.markdown(_warnings._41_excluded_files_reference(st.session_state[user_session_id]['ref_exceptions']))
	
			with st.expander("Documents in reference corpus:"):
				st.write(metadata_reference.get('docids')[0]['ids'])
				
		else:
			st.markdown("---")
			st.markdown('##### Reference corpus:')
			load_ref = st.radio("Would you like to load a reference corpus?", ("No", "Yes"), horizontal=True)
						
			st.markdown("---")
			if load_ref == 'Yes':
									
				ref_corpus_source = st.radio("What kind of reference corpus would you like to prepare?",
						["Internal", "External", "New"],
						captions=[":globe_with_meridians: Load a pre-processed corpus from the interface. (Note that only MICUSP and ELSEVIER can be compared.)",
						":desktop_computer: Load a pre-processed corpus from your computer.",
						":hatching_chick: Process a new corpus from plain text files."], 
						horizontal=False,
						index=None)
				st.markdown("---")
				
				if ref_corpus_source == 'Internal':
					st.markdown(_messages.message_select_reference)
					st.sidebar.markdown("Use the button to load a previously processed corpus.")
					saved_corpora, saved_ref = _handlers.find_saved_reference(metadata_target.get('model')[0], session.get('target_db')[0])
					to_load = st.sidebar.selectbox('Select a saved corpus to load:', (sorted(saved_ref)))			
					if st.sidebar.button("Load Saved Corpus"):
						corp_path = saved_corpora.get(to_load)
						_handlers.load_corpus_internal(corp_path, user_session_id, corpus_type="reference")
						_handlers.init_metadata_reference(user_session_id)
						_handlers.update_session('reference_db', str(corp_path), user_session_id)
						_handlers.update_session('has_reference', True, user_session_id)
						st.rerun()
					st.sidebar.markdown("---")

				if ref_corpus_source == 'External':
					
					st.markdown(_messages.message_load_target_external)

					with st.form("ref-file-form", clear_on_submit=True):
						ref_file = st.file_uploader("Upload your reference corpus", type=["parquet"], accept_multiple_files=False)
						submitted = st.form_submit_button("UPLOAD REFERENCE")
						
						if submitted:
							st.session_state[user_session_id]['warning'] = 0
							
						try:
							tok_pl = pl.read_parquet(ref_file)
							is_valid, dup_ref, corpus_size = _process.check_corpus_pl(tok_pl, check_size=True, check_ref=True, target_docs=metadata_target.get('docids')[0]['ids'])
							if 'ready_to_process' not in st.session_state[user_session_id]:
								st.session_state[user_session_id]['ready_to_process'] = False
							st.session_state[user_session_id]['ready_to_process'] = False
						except:
							is_valid = False
							dup_ref = []
							corpus_size = 0
							if 'ready_to_process' not in st.session_state[user_session_id]:
								st.session_state[user_session_id]['ready_to_process'] = False
							st.session_state[user_session_id]['ready_to_process'] = False
																							
					if is_valid == False and ref_file is not None:
						st.markdown(_warnings._12_polars_format, unsafe_allow_html=True)
												
					if CHECK_SIZE == True:
						if corpus_size > MAX_POLARS:
							st.markdown(_warnings._30_corpus_size, unsafe_allow_html=True)

					if len(dup_ref) > 0:
						st.markdown("---")
						st.markdown(_warnings._21_reference_duplicates(sorted(dup_ref)), unsafe_allow_html=True)
					
					if CHECK_SIZE == True:
						if is_valid == True and tok_pl.is_empty() == False and len(dup_ref) == 0 and corpus_size <= MAX_POLARS:
							st.markdown(f"""```
							Your reference corpus is ready to be processed! Use the button on the sidebar.
							""")
							st.session_state[user_session_id]['ready_to_process'] = True
				
					if CHECK_SIZE == False:
						if is_valid == True and tok_pl.is_empty() == False and len(dup_ref) == 0:
							st.markdown(f"""```
							Your reference corpus is ready to be processed! Use the button on the sidebar.
							""")
							st.session_state[user_session_id]['ready_to_process'] = True

					if st.session_state[user_session_id]['ready_to_process'] == True:
						st.sidebar.markdown("### Process Reference")
						st.sidebar.markdown("Once you have selected a file, use the button to process your corpus.")

						if st.sidebar.button("Load Reference Corpus"):
							ds_tokens = tok_pl
							ft_pos, ft_ds = _analysis.frequency_tables_pl(ds_tokens)
							tt_pos, tt_ds = _analysis.tag_tables_pl(ds_tokens)
							dtm_pos, dtm_ds = _analysis.dtm_pl(ds_tokens)

							_handlers.load_corpus_new(
								ds_tokens,
								dtm_ds,
								dtm_pos,
								ft_ds,
								ft_pos,
								tt_ds,
								tt_pos,
								user_session_id,
								'reference'
								)
							_handlers.init_metadata_reference(user_session_id)
							_handlers.update_session('has_reference', True, user_session_id)
							st.sidebar.markdown("---")
							st.rerun()
					
						st.sidebar.markdown("---")

				if ref_corpus_source == 'New':

					if st.session_state[user_session_id]['warning'] == 11:
						st.markdown(_warnings._11_corrupt_files_reference, unsafe_allow_html=True)
						st.session_state[user_session_id]['warning'] = 0
						st.markdown("---")
					
					if st.session_state[user_session_id]['warning'] == 21:
						st.markdown(_warnings._21_reference_duplicates(st.session_state[user_session_id]['ref_exceptions']))
						st.session_state[user_session_id]['warning'] = 0
						st.markdown("---")

					st.markdown(_messages.message_load_reference)

					with st.form("ref-form", clear_on_submit=True):
						ref_files = st.file_uploader("Upload your reference corpus", type=["txt"], accept_multiple_files=True, key='reffiles')
						submitted = st.form_submit_button("UPLOAD REFERENCE")

						if submitted:
							st.session_state[user_session_id]['warning'] = 0
				
						if CHECK_SIZE == True:
							dup_ids, dup_ref, corpus_size = _process.check_corpus(ref_files, check_size=True, check_ref=True, target_docs=metadata_target.get('docids')[0]['ids'])
							if 'ready_to_process' not in st.session_state[user_session_id]:
								st.session_state[user_session_id]['ready_to_process'] = False
							st.session_state[user_session_id]['ready_to_process'] = False
					
						if CHECK_SIZE == False:
							dup_ids, dup_ref = _process.check_corpus(ref_files, check_ref=True, target_docs=metadata_target.get('docids')[0]['ids'])
							corpus_size = 0
							if 'ready_to_process' not in st.session_state[user_session_id]:
								st.session_state[user_session_id]['ready_to_process'] = False
							st.session_state[user_session_id]['ready_to_process'] = False
												
					if CHECK_SIZE == True:
						if corpus_size > MAX_TEXT:
							st.markdown(_warnings._30_corpus_size, unsafe_allow_html=True)

					if len(dup_ids) > 0:
						st.markdown("---")
						st.markdown(_warnings._20_corpus_duplicates(sorted(dup_ids)), unsafe_allow_html=True)
					
					if len(dup_ref) > 0:
						st.markdown("---")
						st.markdown(_warnings._21_reference_duplicates(sorted(dup_ref)), unsafe_allow_html=True)
					
					if CHECK_SIZE == True:
						if len(ref_files) > 0 and len(dup_ids) == 0 and len(dup_ref) == 0 and corpus_size <= MAX_TEXT:
							st.markdown(f"""```
							{len(ref_files)} reference corpus files ready to be processed! Use the button on the sidebar.
							""")
							st.session_state[user_session_id]['ready_to_process'] = True
				
					if CHECK_SIZE == False:
						if len(ref_files) > 0 and len(dup_ids) == 0 and len(dup_ref) == 0:
							st.markdown(f"""```
							{len(ref_files)} reference corpus files ready to be processed! Use the button on the sidebar.
							""")
							st.session_state[user_session_id]['ready_to_process'] = True
	
					if st.session_state[user_session_id]['ready_to_process'] == True:
						st.sidebar.markdown("### Process Reference")
						st.sidebar.markdown("Click the button to process your reference corpus files.")
						if st.sidebar.button("Process Reference Corpus"):
							with st.sidebar:
								with st.spinner('Processing corpus data...'):
									models = load_models()
									selected_dict = metadata_target.get('model')[0]
									nlp = models[selected_dict]

									if ENABLE_DETECT == True:
										detector = load_detector()
										ref_corp, exceptions = _process.process_corpus_detect(ref_files, nlp, detector, Language.ENGLISH)
									
									if ENABLE_DETECT == False:
										ref_corp, exceptions = _process.process_corpus(ref_files, nlp)
								
								if len(exceptions) > 0 and bool(ref_corp) == False:
									st.session_state[user_session_id]['warning'] = 11
									exceptions = None
									st.rerun()
								
								elif len(exceptions) > 0 and bool(ref_corp) == True:
									st.session_state[user_session_id]['warning'] = 41
									st.session_state[user_session_id]['ref_exceptions'] = exceptions

									ds_tokens = _process.tokens_to_pl(ref_corp)
									ft_pos, ft_ds = _analysis.frequency_tables_pl(ds_tokens)
									tt_pos, tt_ds = _analysis.tag_tables_pl(ds_tokens)
									dtm_pos, dtm_ds = _analysis.dtm_pl(ds_tokens)

									_handlers.load_corpus_new(
										ds_tokens,
										dtm_ds,
										dtm_pos,
										ft_ds,
										ft_pos,
										tt_ds,
										tt_pos,
										user_session_id,
										'reference'
										)
									_handlers.init_metadata_reference(user_session_id)
									_handlers.update_session('has_reference', True, user_session_id)
									st.rerun()
								
								else:
									st.success('Processing complete!')
									st.session_state[user_session_id]['warning'] = 0
									
									ds_tokens = _process.tokens_to_pl(ref_corp)
									ft_pos, ft_ds = _analysis.frequency_tables_pl(ds_tokens)
									tt_pos, tt_ds = _analysis.tag_tables_pl(ds_tokens)
									dtm_pos, dtm_ds = _analysis.dtm_pl(ds_tokens)

									_handlers.load_corpus_new(
										ds_tokens,
										dtm_ds,
										dtm_pos,
										ft_ds,
										ft_pos,
										tt_ds,
										tt_pos,
										user_session_id,
										'reference'
										)
									_handlers.init_metadata_reference(user_session_id)
									_handlers.update_session('has_reference', True, user_session_id)
									st.rerun()
						
						st.sidebar.markdown("---")
		
		st.markdown("##### Download:")		
		download_corpus = st.toggle("Would you like to download any of your processed data?")

		if download_corpus == True:

			corpus_select = st.radio("Choose a corpus", ["Target", "Reference"], captions=["", "You can only download reference corpus data if you've processed one."])

			if corpus_select == "Target":

				data_select = st.radio("Choose the data to download", ["Corpus file only", "All of the processed data"], 
						   captions=["This is the option you want if you're planning to save your corpus for future analysis using this tool.", 
				   					"This is the option you want if you're planning to explore your data outside of the tool, in coding enviroments like R or Python. The data include the token file, frequency tables, and document-term-matrices."])
				
				if data_select == "Corpus file only":
					download_file = st.session_state[user_session_id]["target"]["ds_tokens"].to_pandas().to_parquet()
					
					st.markdown("---")
					st.markdown("#### Click the button to download your corpus file.")
					st.download_button(
						label="Download Corpus File",
						data=download_file,
						file_name="corpus.parquet",
							mime="parquet",
							)
			
				if data_select == "All of the processed data":

					format_select = st.radio("Select a file format", ["CSV", "PARQUET"], horizontal=True)

					if format_select == "CSV":

						download_file = _handlers.convert_corpus_to_zip(user_session_id, 'target', file_type='csv')

						st.markdown("---")
						st.markdown("#### Click the button to download your corpus files.")
						st.download_button(
							label="Download Corpus Files",
							data=download_file,
							file_name="corpus_files.zip",
									mime="application/zip",
							)
					
					if format_select == "PARQUET":

						download_file = _handlers.convert_corpus_to_zip(user_session_id, 'target')

						st.markdown("---")
						st.markdown("#### Click the button to download your corpus files.")
						st.download_button(
							label="Download Corpus Files",
							data=download_file,
							file_name="corpus_files.zip",
									mime="application/zip",
							)
	
			if corpus_select == "Reference":
				if session.get('has_reference')[0] == False:
					st.markdown(_warnings.warning_17, unsafe_allow_html=True)
				
				if session.get('has_reference')[0] == True:

					data_select = st.radio("Choose the data to download", ["Corpus file only", "All of the processed data"], 
							captions=["This is the option you want if you're planning to save your corpus for future analysis using this tool.", 
										"This is the option you want if you're planning to explore your data outside of the tool, in coding enviroments like R or Python. The data include the token file, frequency tables, and document-term-matrices."])
					
					if data_select == "Corpus file only":
						download_file = st.session_state[user_session_id]["reference"]["ds_tokens"].to_pandas().to_parquet()
						
						st.markdown("---")
						st.markdown("#### Click the button to download your corpus file.")
						st.download_button(
							label="Download Corpus File",
							data=download_file,
							file_name="corpus.parquet",
								mime="parquet",
								)
				
					if data_select == "All of the processed data":

						format_select = st.radio("Select a file format", ["CSV", "PARQUET"], horizontal=True)

						if format_select == "CSV":

							download_file = _handlers.convert_corpus_to_zip(user_session_id, 'reference', file_type='csv')

							st.markdown("---")
							st.markdown("#### Click the button to download your corpus files.")
							st.download_button(
								label="Download Corpus Files",
								data=download_file,
								file_name="corpus_files.zip",
										mime="application/zip",
								)
						
						if format_select == "PARQUET":

							download_file = _handlers.convert_corpus_to_zip(user_session_id, 'reference')

							st.markdown("---")
							st.markdown("#### Click the button to download your corpus files.")
							st.download_button(
								label="Download Corpus Files",
								data=download_file,
								file_name="corpus_files.zip",
										mime="application/zip",
								)
		st.markdown("---")

		st.sidebar.markdown('### Reset all tools and files:')
		st.sidebar.markdown(":warning: Using the **reset** button will cause all files, tables, and plots to be cleared.")
		if st.sidebar.button("Reset Corpus"):
			st.session_state[user_session_id] = {}
			_handlers.generate_temp(_states.STATES.items(), user_session_id)
			_handlers.init_session(user_session_id)
			st.rerun()
		st.sidebar.markdown("""---""")
		
	else:
	
		st.markdown("###  :dart: Load or process a target corpus")

		st.markdown(_messages.message_load)
		
		st.markdown("---")
		
		with st.expander("About internal corpora"):
				st.markdown(_messages.message_internal_corpora)
		
		with st.expander("About external corpora"):
				st.markdown(_messages.message_external_corpora)

		with st.expander("About new corpora"):
			st.markdown(_messages.message_naming)

		st.markdown("---")

		st.markdown("### Process a corpus:")
		
		corpus_source = st.radio("What kind of corpus would you like to prepare?",
						["Internal", "External", "New"],
						captions=[":globe_with_meridians: Load a pre-processed corpus from the interface.",
						":desktop_computer: Load a pre-processed corpus from your computer.",
						":hatching_chick: Process a new corpus from plain text files."], 
						horizontal=False,
						index=None)
		
		if corpus_source == 'Internal':
			st.markdown("---")
			st.markdown(_messages.message_load_target_internal)
			st.sidebar.markdown("Use the button to load a previously processed corpus.")
			from_model = st.sidebar.radio("Select data tagged with:", ("Large Dictionary", "Common Dictionary"), key='corpora_to_load')
			if from_model == 'Large Dictionary':
				saved_corpora = _handlers.find_saved('ld')
				to_load = st.sidebar.selectbox('Select a saved corpus to load:', (sorted(saved_corpora)))
			if from_model == 'Common Dictionary':
				saved_corpora = _handlers.find_saved('cd')
				to_load = st.sidebar.selectbox('Select a saved corpus to load:', (sorted(saved_corpora)))	
			if st.sidebar.button("Load Saved Corpus"):
				corp_path = saved_corpora.get(to_load)
				_handlers.load_corpus_internal(corp_path, user_session_id)
				_handlers.init_metadata_target(user_session_id)
				_handlers.update_session('target_db', str(corp_path), user_session_id)
				_handlers.update_session('has_target', True, user_session_id)
				st.rerun()
			st.sidebar.markdown("---")

		if corpus_source == 'External':
			st.markdown("---")
			
			st.markdown(_messages.message_load_target_external)

			with st.form("corpus-file-form", clear_on_submit=True):
				corp_file = st.file_uploader("Upload your target corpus", type=["parquet"], accept_multiple_files=False)
				submitted = st.form_submit_button("UPLOAD TARGET")
				
				if submitted:
					st.session_state[user_session_id]['warning'] = 0

				try:
					tok_pl = pl.read_parquet(corp_file)
					is_valid, corpus_size = _process.check_corpus_pl(tok_pl, check_size=True, check_ref=False)
					if 'ready_to_process' not in st.session_state[user_session_id]:
						st.session_state[user_session_id]['ready_to_process'] = False
					st.session_state[user_session_id]['ready_to_process'] = False
				except:
					is_valid = False
					corpus_size = 0
					if 'ready_to_process' not in st.session_state[user_session_id]:
						st.session_state[user_session_id]['ready_to_process'] = False
					st.session_state[user_session_id]['ready_to_process'] = False
																					
			if is_valid == False and corp_file is not None:
				st.markdown(_warnings._12_polars_format, unsafe_allow_html=True)
										
			if CHECK_SIZE == True:
				if corpus_size > MAX_POLARS:
					st.markdown(_warnings._30_corpus_size, unsafe_allow_html=True)
			
			if CHECK_SIZE == True:
				if is_valid == True and tok_pl.is_empty() == False and corpus_size <= MAX_POLARS:
					st.markdown(f"""```
					Your target corpus is ready to be processed! Use the button on the sidebar.
					""")
					st.session_state[user_session_id]['ready_to_process'] = True
		
			if CHECK_SIZE == False:
				if is_valid == True and tok_pl.is_empty() == False:
					st.markdown(f"""```
					Your target corpus is ready to be processed! Use the button on the sidebar.
					""")
					st.session_state[user_session_id]['ready_to_process'] = True

			if st.session_state[user_session_id]['ready_to_process'] == True:
				st.sidebar.markdown("### Process Target")
				st.sidebar.markdown("Once you have selected a file, use the button to process your corpus.")

				if st.sidebar.button("Load Target Corpus"):
					ds_tokens = tok_pl
					ft_pos, ft_ds = _analysis.frequency_tables_pl(ds_tokens)
					tt_pos, tt_ds = _analysis.tag_tables_pl(ds_tokens)
					dtm_pos, dtm_ds = _analysis.dtm_pl(ds_tokens)

					_handlers.load_corpus_new(
						ds_tokens,
						dtm_ds,
						dtm_pos,
						ft_ds,
						ft_pos,
						tt_ds,
						tt_pos,
						user_session_id,
						'target'
						)
					_handlers.init_metadata_target(user_session_id)
					_handlers.update_session('has_target', True, user_session_id)
					st.sidebar.markdown("---")
					st.rerun()
					
		if corpus_source == 'New':
			st.markdown("---")
			
			if st.session_state[user_session_id]['warning'] == 10:
				st.markdown(_warnings._10_corrupt_files_target, unsafe_allow_html=True)
				st.session_state[user_session_id]['warning'] = 0
				st.markdown("---")

			st.markdown(_messages.message_load_target_new)
							
			with st.form("corpus-form", clear_on_submit=True):
				corp_files = st.file_uploader("Upload your target corpus", type=["txt"], accept_multiple_files=True)
				submitted = st.form_submit_button("UPLOAD TARGET")

				if submitted:
					st.session_state[user_session_id]['warning'] = 0
				
				if CHECK_SIZE == True:
					dup_ids, corpus_size = _process.check_corpus(corp_files, check_size=True)
					if 'ready_to_process' not in st.session_state[user_session_id]:
						st.session_state[user_session_id]['ready_to_process'] = False
					st.session_state[user_session_id]['ready_to_process'] = False
					
				if CHECK_SIZE == False:
					dup_ids = _process.check_corpus(corp_files)
					corpus_size = 0
					if 'ready_to_process' not in st.session_state[user_session_id]:
						st.session_state[user_session_id]['ready_to_process'] = False
					st.session_state[user_session_id]['ready_to_process'] = False

			if CHECK_SIZE == True:
					if corpus_size > MAX_TEXT:
						st.markdown(_warnings._30_corpus_size, unsafe_allow_html=True)
			
			if len(dup_ids) > 0:
				st.markdown("---")
				st.markdown(_warnings._20_corpus_duplicates(sorted(dup_ids)), unsafe_allow_html=True)
				
			if CHECK_SIZE == True:
				if len(corp_files) > 0 and len(dup_ids) == 0 and corpus_size <= MAX_TEXT:
					st.markdown(f"""```
					{len(corp_files)} target corpus files ready to be processed! Use the button on the sidebar.
					""")
					st.session_state[user_session_id]['ready_to_process'] = True
					
			if CHECK_SIZE == False:
				if len(corp_files) > 0 and len(dup_ids) == 0:
					st.markdown(f"""```
					{len(corp_files)} target corpus files ready to be processed! Use the button on the sidebar.
					""")
					st.session_state[user_session_id]['ready_to_process'] = True	

			st.sidebar.markdown("### Models")
			models = load_models()
			selected_dict = st.sidebar.selectbox("Select a DocuScope model:", options=["Large Dictionary", "Common Dictionary"])
			nlp = models[selected_dict]
			st.session_state[user_session_id]['model'] = selected_dict
		
			with st.sidebar.expander("Which model do I choose?"):
				st.markdown(_messages.message_models)		
			
			st.sidebar.markdown("---")
				
			if st.session_state[user_session_id]['ready_to_process'] == True:
				st.sidebar.markdown("### Process Target")
				st.sidebar.markdown("Once you have selected your files, use the button to process your corpus.")
				if st.sidebar.button("Process Target"):
					with st.sidebar:
						with st.spinner('Processing corpus data...'):
							if ENABLE_DETECT == True:
								detector = load_detector()
								corp, exceptions = _process.process_corpus_detect(corp_files, nlp, detector, Language.ENGLISH)

							if ENABLE_DETECT == False:
								corp, exceptions = _process.process_corpus(corp_files, nlp)
						
						if len(exceptions) > 0 and bool(corp) == False:
							st.session_state[user_session_id]['warning'] = 10
							st.rerun()
						
						elif len(exceptions) > 0 and bool(corp) == True:
							st.session_state[user_session_id]['warning'] = 40
							st.session_state[user_session_id]['exceptions'] = exceptions

							ds_tokens = _process.tokens_to_pl(corp)
							ft_pos, ft_ds = _analysis.frequency_tables_pl(ds_tokens)
							tt_pos, tt_ds = _analysis.tag_tables_pl(ds_tokens)
							dtm_pos, dtm_ds = _analysis.dtm_pl(ds_tokens)

							_handlers.load_corpus_new(
								ds_tokens,
								dtm_ds,
								dtm_pos,
								ft_ds,
								ft_pos,
								tt_ds,
								tt_pos,
								user_session_id,
								'target'
								)
							_handlers.init_metadata_target(user_session_id)
							_handlers.update_session('has_target', True, user_session_id)
							st.rerun()
						
						else:
							st.success('Processing complete!')
							st.session_state[user_session_id]['warning'] = 0

							ds_tokens = _process.tokens_to_pl(corp)
							ft_pos, ft_ds = _analysis.frequency_tables_pl(ds_tokens)
							tt_pos, tt_ds = _analysis.tag_tables_pl(ds_tokens)
							dtm_pos, dtm_ds = _analysis.dtm_pl(ds_tokens)

							_handlers.load_corpus_new(
								ds_tokens,
								dtm_ds,
								dtm_pos,
								ft_ds,
								ft_pos,
								tt_ds,
								tt_pos,
								user_session_id,
								'target'
								)
							_handlers.init_metadata_target(user_session_id)
							_handlers.update_session('has_target', True, user_session_id)
							st.rerun()
				
				st.sidebar.markdown("---")
				
if __name__ == "__main__":
    main()
