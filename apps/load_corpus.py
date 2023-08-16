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

import pathlib
from collections import Counter
from importlib.machinery import SourceFileLoader

# set paths
HERE = pathlib.Path(__file__).parents[1].resolve()
MODEL_LARGE = str(HERE.joinpath("models/en_docusco_spacy"))
MODEL_SMALL = str(HERE.joinpath("models/en_docusco_spacy_fc"))
MODEL_DETECT = str(HERE.joinpath("models/lid.176.ftz"))
OPTIONS = str(HERE.joinpath("options.toml"))
IMPORTS = str(HERE.joinpath("utilities/handlers_imports.py"))

# import options
_imports = SourceFileLoader("handlers_imports", IMPORTS).load_module()
_options = _imports.import_options_general(OPTIONS)

# some modules have different import path requirements for electron
modules = ['categories', 'handlers', 'messages', 'process', 'states', 'warnings', 'streamlit', 'spacy']
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

# set options
ENABLE_DETECT = _options['global']['check_language']
ENABLE_SAVE = _options['global']['enable_save']
DESKTOP = _options['global']['desktop_mode']
CHECK_SIZE = _options['global']['check_size']

# fasttext is not available for Windows
if ENABLE_DETECT == True:
	try:
		import fasttext
	except:
		ENABLE_DETECT = False

if CHECK_SIZE == True:
	MAX_BYTES = _options['global']['max_bytes']

CATEGORY = _categories.CORPUS_LOAD
TITLE = "Manage Corpus Data"
KEY_SORT = 1

# desktop uses and older streamlit version with different caching function
if DESKTOP == True:
	@st.cache(show_spinner=False, allow_output_mutation=True, suppress_st_warning=True)
	def load_models():
		large_model = spacy.load(MODEL_LARGE)
		small_model = spacy.load(MODEL_SMALL)
		models = {"Large Dictionary": large_model, "Common Dictionary": small_model}
		return models

if DESKTOP == False:
	@st.cache_data(show_spinner=False)
	def load_models():
		large_model = spacy.load(MODEL_LARGE)
		small_model = spacy.load(MODEL_SMALL)
		models = {"Large Dictionary": large_model, "Common Dictionary": small_model}
		return models

def main():

	user_session = st.runtime.scriptrunner.script_run_context.get_script_run_ctx()
	user_session_id = user_session.session_id

	session = _handlers.load_session(user_session_id)
				
	if 'warning' not in st.session_state[user_session_id]:
		st.session_state[user_session_id]['warning'] = 0
	
	if session.get('target_path') is not None:
		metadata_target = _handlers.load_metadata('target', user_session_id)

		if session.get('has_reference') == True:
			metadata_reference = _handlers.load_metadata('reference', user_session_id)
		
		st.markdown(_messages.message_target_info(metadata_target))
				
		if st.session_state[user_session_id]['warning'] == 4:
			st.markdown(_warnings.warning_4(st.session_state[user_session_id]['exceptions']))
		
		with st.expander("Documents:"):
			st.write(sorted(metadata_target.get('docids')))
		
		if session.get('has_meta') == True:
			st.markdown('##### Target corpus metadata:')
			with st.expander("Counts of document categories:"):
				st.write(Counter(metadata_target.get('doccats')))
		
		else:
			st.sidebar.markdown('### Target corpus metadata:')
			load_cats = st.sidebar.radio("Do you have categories in your file names to process?", ("No", "Yes"), horizontal=True)
			if load_cats == 'Yes':
				if st.sidebar.button("Process Document Metadata"):
					with st.spinner('Processing metadata...'):
						doc_cats = _process.get_doc_cats(metadata_target.get('docids'))
						if len(set(doc_cats)) > 1 and len(set(doc_cats)) < 21:
							_handlers.update_metadata('target', 'doccats', doc_cats, user_session_id)
							_handlers.update_session('has_meta', True, user_session_id)
							st.success('Processing complete!')
							st.experimental_rerun()
						elif len(doc_cats) != 0:
							st.sidebar.markdown(_warnings.warning_8, unsafe_allow_html=True)
						else:
							st.sidebar.markdown(_warnings.warning_9, unsafe_allow_html=True)
			
			st.sidebar.markdown("""---""")
				
		if ENABLE_SAVE == True:
			if session.get('from_saved') == 'No' and session.get('is_saved') == 'No':
				st.markdown("---")
				save_target = st.radio("Would you like to save your target corpus for future analysis?", ("No", "Yes"), horizontal=True)
				if save_target == 'Yes':
					target_name = st.text_input("Name your target corpus (only letters, numbers, the hyphen or the underscore are allowed):")
					if st.button("Save Corpus"):
						with st.spinner('Saving corpus...'):
							if len(target_name) > 2 & len(target_name) < 15 and _handlers.check_name(target_name) == True:
								corp = _handlers.load_temp('target')
								tags_pos, tags_ds = _process.get_corpus_features(corp)
								model = _process.check_model(tags_ds)
								_handlers.save_corpus(corp, model, target_name)
								_handlers.update_session('is_saved', 'Yes', user_session_id)
								st.success('Corpus saved!')
								st.experimental_rerun()
							else:
								st.markdown(_warnings.warning_10, unsafe_allow_html=True)
		
		if session.get('has_reference') == True:
			metadata_reference = _handlers.load_metadata('reference', user_session_id)
			
			st.markdown(_messages.message_reference_info(metadata_reference))
			
			if st.session_state[user_session_id]['warning'] == 5:
				st.markdown(_warnings.warning_5(st.session_state[user_session_id]['ref_exceptions']))

			if st.session_state[user_session_id]['warning'] == 4:
				st.markdown(_warnings.warning_4(st.session_state[user_session_id]["ref_exceptions"]))
	
			with st.expander("Documents in reference corpus:"):
				st.write(sorted(metadata_reference.get('docids')))
				
		else:
			st.markdown("---")
			st.markdown('### Reference corpus:')
			load_ref = st.radio("Would you like to load a reference corpus?", ("No", "Yes"), horizontal=True)
			
			if st.session_state[user_session_id]['warning'] == 1:
				st.markdown(_warnings.warning_1, unsafe_allow_html=True)

			if st.session_state[user_session_id]['warning'] == 6:
				st.markdown(_warnings.warning_6, unsafe_allow_html=True)
			
			st.markdown("---")
			if load_ref == 'Yes':
									
				ref_from_saved = st.radio("Would you like to load your reference from a saved corpus?", ("No", "Yes"), horizontal=True)
				st.markdown("---")
				
				if ref_from_saved == 'Yes':
					st.markdown(_messages.message_select_reference)
					st.sidebar.markdown("Use the button to load a previously processed corpus.")
					saved_corpora, saved_ref = _handlers.find_saved_reference(metadata_target.get('model'), session.get('target_path'))
					to_load = st.sidebar.selectbox('Select a saved corpus to load:', (sorted(saved_ref)))			
					if st.sidebar.button("Load Saved Corpus"):
						corp_path = saved_corpora.get(to_load)
						ref_corp = _handlers.load_corpus_path(corp_path)
						ref_corp, exceptions = _process.check_reference(ref_corp, metadata_target.get('docids'))

						if len(exceptions) > 0 and bool(ref_corp) == False:
							st.session_state[user_session_id]['warning'] = 6
							st.experimental_rerun()
						
						elif len(exceptions) > 0 and bool(ref_corp) == True:
							st.session_state[user_session_id]['warning'] = 7
							st.session_state[user_session_id]['ref_exceptions'] = exceptions
							#get features
							tags_pos, tags_ds = _process.get_corpus_features(ref_corp)
							model = _process.check_model(tags_ds)
							#assign session states
							_handlers.init_metadata_reference(ref_corp, model, tags_pos, tags_ds, user_session_id)
							_handlers.update_session('reference_path', corp_path, user_session_id)
							_handlers.update_session('has_reference', True, user_session_id)
							st.experimental_rerun()
						
						else:
							st.success('Processing complete!')
							st.session_state[user_session_id]['warning'] = 0
							#get features
							tags_pos, tags_ds = _process.get_corpus_features(ref_corp)
							model = _process.check_model(tags_ds)
							#assign session states
							_handlers.init_metadata_reference(ref_corp, model, tags_pos, tags_ds, user_session_id)
							_handlers.update_session('reference_path', corp_path, user_session_id)
							_handlers.update_session('has_reference', True, user_session_id)
							st.experimental_rerun()
					st.sidebar.markdown("---")

				else:
					st.markdown(_messages.message_load_reference)
					with st.form("ref-form", clear_on_submit=True):
						ref_files = st.file_uploader("Upload your reference corpus", type=["txt"], accept_multiple_files=True, key='reffiles')
						submitted = st.form_submit_button("UPLOAD REFERENCE")
				
						if CHECK_SIZE == True:
							dup_ids, dup_ref, corpus_size = _process.check_corpus(ref_files, check_size=True, check_ref=True, target_docs=metadata_target.get('docids'))
							if 'ready_to_process' not in st.session_state[user_session_id]:
								st.session_state[user_session_id]['ready_to_process'] = False
							st.session_state[user_session_id]['ready_to_process'] = False
					
						if CHECK_SIZE == False:
							dup_ids, dup_ref = _process.check_corpus(ref_files, check_ref=True, target_docs=metadata_target.get('docids'))
							corpus_size = 0
							if 'ready_to_process' not in st.session_state[user_session_id]:
								st.session_state[user_session_id]['ready_to_process'] = False
							st.session_state[user_session_id]['ready_to_process'] = False
												
					if CHECK_SIZE == True:
						if corpus_size > MAX_BYTES:
							st.markdown(_warnings.warning_3, unsafe_allow_html=True)

					if len(dup_ids) > 0:
						st.markdown(_warnings.warning_2(sorted(dup_ids)), unsafe_allow_html=True)
					
					if len(dup_ref) > 0:
						st.markdown(_warnings.warning_5(sorted(dup_ref)), unsafe_allow_html=True)
					
					if CHECK_SIZE == True:
						if len(ref_files) > 0 and len(dup_ids) == 0 and len(dup_ref) == 0 and corpus_size <= MAX_BYTES:
							st.markdown(f"""```
							{len(ref_files)} reference corpus files ready to be processed! Use the button on the sidebar.
							""")
							st.session_state[user_session_id]['ready_to_process'] = True
					
					if CHECK_SIZE == False:
						if len(ref_files) > 0 and len(dup_ids) == 0 and len(dup_ref) == 0 :
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
									selected_dict = metadata_target.get('model')
									nlp = models[selected_dict]
									if ENABLE_DETECT == True:
										detector = _process.load_detector(MODEL_DETECT)
										ref_corp, exceptions = _process.process_corpus(ref_files, nlp, detector)
									if ENABLE_DETECT == False:
										ref_corp, exceptions = _process.process_corpus(ref_files, nlp)
								
								if len(exceptions) > 0 and bool(ref_corp) == False:
									st.session_state[user_session_id]['warning'] = 1
									st.experimental_rerun()
								
								elif len(exceptions) > 0 and bool(ref_corp) == True:
									st.session_state[user_session_id]['warning'] = 4
									st.session_state[user_session_id]['ref_exceptions'] = exceptions
									#get features
									tags_pos, tags_ds = _process.get_corpus_features(ref_corp)
									#assign session states
									_handlers.save_corpus_temp(ref_corp, 'reference')
									_handlers.init_metadata_reference(ref_corp, selected_dict, tags_pos, tags_ds, user_session_id)
									_handlers.update_session('has_reference', True, user_session_id)
									st.experimental_rerun()
								
								else:
									st.success('Processing complete!')
									st.session_state[user_session_id]['warning'] = 0
									#get features
									tags_pos, tags_ds = _process.get_corpus_features(ref_corp)
									#assign session states
									_handlers.save_corpus_temp(ref_corp, 'reference')
									_handlers.init_metadata_reference(ref_corp, selected_dict, tags_pos, tags_ds, user_session_id)
									_handlers.update_session('has_reference', True, user_session_id)
									st.experimental_rerun()
								
						st.sidebar.markdown("---")
		
		st.sidebar.markdown('### Reset all tools and files:')
		st.sidebar.markdown(":warning: Using the **reset** button will cause all files, tables, and plots to be cleared.")
		if st.sidebar.button("Reset Corpus"):
			st.session_state[user_session_id] = {}
			for key, value in _states.STATES.items():
				st.session_state[user_session_id][key] = value
			_handlers.clear_temp(user_session_id)
			_handlers.reset_session(user_session_id)
			st.experimental_rerun()
		st.sidebar.markdown("""---""")
		
	else:
	
		st.write(st.session_state[user_session_id])
		st.write(_handlers.data_path())
		st.markdown("###  :dart: Load or process a target corpus")

		st.markdown(_messages.message_load)
		
		st.markdown("---")
		
		with st.expander("About saved corpora"):
				st.markdown(_messages.message_saved_corpora)
		
		st.markdown("### Load a saved corpus:")
		
		from_saved = st.radio("Would you like to load a saved corpus?", ("No", "Yes"), horizontal=True)
		
		if from_saved == 'Yes':
			st.markdown("---")
			st.markdown(_messages.message_select_target)
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
				corp = _handlers.load_corpus_path(corp_path)
				tags_pos, tags_ds = _process.get_corpus_features(corp)
				model = _process.check_model(tags_ds)
				_handlers.init_metadata_target(corp, model, tags_pos, tags_ds, user_session_id)
				_handlers.update_session('target_path', corp_path, user_session_id)
				_handlers.update_session('from_saved', 'Yes', user_session_id)
				_handlers.update_session('is_saved', 'Yes', user_session_id)
				st.experimental_rerun()
			st.sidebar.markdown("---")
		
		if from_saved == 'No':
			st.markdown("---")
			st.markdown("#### Process a new corpus:")
			
			st.markdown(_messages.message_load_target)
			
			with st.expander("File preparation and file naming tips"):
				st.markdown(_messages.message_naming)
				
			if st.session_state[user_session_id]['warning'] == 1:
				st.markdown(_warnings.warning_1, unsafe_allow_html=True)
			
			with st.form("corpus-form", clear_on_submit=True):
				corp_files = st.file_uploader("Upload your target corpus", type=["txt"], accept_multiple_files=True)
				submitted = st.form_submit_button("UPLOAD TARGET")
				
				if CHECK_SIZE == True:
					dup_ids, corpus_size = _process.check_corpus(corp_files, check_size=True)
					if 'ready_to_process' not in st.session_state[user_session_id]:
						st.session_state[user_session_id]['ready_to_process'] = False
					st.session_state[user_session_id]['ready_to_process'] = False
					
				if CHECK_SIZE == False:
					dup_ids = _process.check_corpus(corp_files)
					corpus_size = 0
					if 'ready_to_process' not in st.session_state:
						st.session_state[user_session_id]['ready_to_process'] = False
					st.session_state[user_session_id]['ready_to_process'] = False

			if CHECK_SIZE == True:
					if corpus_size > MAX_BYTES:
						st.markdown(_warnings.warning_3, unsafe_allow_html=True)
			
			if len(dup_ids) > 0:
				st.markdown(_warnings.warning_2(sorted(dup_ids)), unsafe_allow_html=True)
				
			if CHECK_SIZE == True:
				if len(corp_files) > 0 and len(dup_ids) == 0 and corpus_size <= MAX_BYTES:
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
				if st.sidebar.button("Process Corpus"):
					with st.sidebar:
						with st.spinner('Processing corpus data...'):
							if ENABLE_DETECT == True:
								detector = _process.load_detector(MODEL_DETECT)
								corp, exceptions = _process.process_corpus(corp_files, nlp, detector)
							if ENABLE_DETECT == False:
								corp, exceptions = _process.process_corpus(corp_files, nlp)
						
						if len(exceptions) > 0 and bool(corp) == False:
							st.session_state[user_session_id]['warning'] = 1
							st.experimental_rerun()
						
						elif len(exceptions) > 0 and bool(corp) == True:
							st.session_state[user_session_id]['warning'] = 4
							st.session_state[user_session_id]['exceptions'] = exceptions
							#get features
							tags_pos, tags_ds = _process.get_corpus_features(corp)
							#assign session states
							_handlers.save_corpus_temp(corp, 'target')
							_handlers.init_metadata_target(corp, selected_dict, tags_pos, tags_ds, user_session_id)
							st.experimental_rerun()
						
						else:
							st.success('Processing complete!')
							st.session_state[user_session_id]['warning'] = 0
							#get features
							tags_pos, tags_ds = _process.get_corpus_features(corp)
							#assign session states
							_handlers.save_corpus_temp(corp, 'target')
							_handlers.init_metadata_target(corp, selected_dict, tags_pos, tags_ds, user_session_id)
							st.experimental_rerun()
				st.sidebar.markdown("---")
				
		if ENABLE_SAVE == True:
			st.sidebar.markdown('### Delete saved corpus:')
			st.sidebar.markdown(":ghost: Using the **delete** button will permanently remove the corpus and canot be undone.")
			affirm = st.sidebar.radio("Are you sure you want to delete a corpus:", ("No", "Yes"), horizontal=True)
			if affirm == 'Yes':
				del_from_model = st.sidebar.radio("Select data tagged with:", ("Large Dictionary", "Common Dictionary"), key='corpora_to_delete')
				if del_from_model == 'Large Dictionary':
					saved_corpora = _handlers.find_saved('ld')
					to_delete = st.sidebar.selectbox('Select a saved corpus to delete:', (sorted(saved_corpora)))
				if del_from_model == 'Common Dictionary':
					saved_corpora = _handlers.find_saved('cd')		
					to_delete = st.sidebar.selectbox('Select a saved corpus to delete:', (sorted(saved_corpora)))
				if st.sidebar.button("Delete Corpus"):
					path = pathlib.Path(saved_corpora.get(to_delete))
					path.unlink()
					st.experimental_rerun()
			st.sidebar.markdown("""---""")

if __name__ == "__main__":
    main()
