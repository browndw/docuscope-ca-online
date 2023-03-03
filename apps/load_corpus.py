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

import streamlit as st
import spacy
import fasttext

import re
import pathlib
from collections import Counter

import categories
import states as _states
from utilities import warnings
from utilities import process_functions

HERE = pathlib.Path(__file__).parents[1].resolve()
MODEL_LARGE = str(HERE.joinpath("models/en_docusco_spacy"))
MODEL_SMALL = str(HERE.joinpath("models/en_docusco_spacy_fc"))
MODEL_DETECT = str(HERE.joinpath("models/lid.176.ftz"))

CATEGORY = categories.CORPUS_LOAD
TITLE = "Mangage Corpus Data"
KEY_SORT = 1
MAX_BYTES = 20000000

@st.cache_data(show_spinner=False)
def load_models():
    large_model = spacy.load(MODEL_LARGE)
    small_model = spacy.load(MODEL_SMALL)
    models = {"Large Dictionary": large_model, "Common Dictionary": small_model}
    return models

def load_detector():
    fasttext.FastText.eprint = lambda x: None
    detector = fasttext.load_model(MODEL_DETECT)
    return(detector)

def main():
	# check states to prevent unlikely error
	for key, value in _states.STATES.items():
		if key not in st.session_state:
			setattr(st.session_state, key, value)

	if st.session_state.ndocs > 0:
		st.markdown(f"""##### Target corpus information:
		
		Number of tokens in corpus: {st.session_state.tokens}\n    Number of word tokens in corpus: {st.session_state.words}\n    Number of documents in corpus: {st.session_state.ndocs}
		""")
				
		if st.session_state.warning == 3:
			st.markdown(f"""###### The following documents were excluded from the corpus either because they are improperly encoded or have substantial segments not recognized as English (and, therefore, cannot be tagged properly):
			
			{', '.join(st.session_state.exceptions)}
			""")
		
		with st.expander("Documents:"):
			st.write(sorted(st.session_state.docids))
		
		if st.session_state.doccats != '':
			st.markdown('##### Target corpus metadata:')
			with st.expander("Counts of document categories:"):
				st.write(Counter(st.session_state.doccats))
		else:
			st.sidebar.markdown('##### Target corpus metadata:')
			load_cats = st.sidebar.radio("Do you have categories in your file names to process?", ("No", "Yes"), horizontal=True)
			if load_cats == 'Yes':
				if st.sidebar.button("Process Document Metadata"):
					with st.spinner('Processing metadata...'):
						if all(['_' in item for item in st.session_state.docids]):
							doc_cats = [re.sub(r"_\S+$", "", item, flags=re.UNICODE) for item in st.session_state.docids]
							if min([len(item) for item in doc_cats]) == 0:
								st.markdown(":no_entry_sign: Your categories don't seem to be formatted correctly. You can either proceed without assigning categories, or reset the corpus, fix your file names, and try again.")
							elif len(set(doc_cats)) > 1 and len(set(doc_cats)) < 21:
								st.session_state.doccats = doc_cats
								st.success('Processing complete!')
								st.experimental_rerun()
							else:
								st.markdown(warnings.warning_5, unsafe_allow_html=True)
						else:
							st.markdown(warnings.warning_6, unsafe_allow_html=True)
			
			st.sidebar.markdown("""---""")
		
		if st.session_state.reference != '':
			st.markdown(f"""##### Reference corpus information:
			
			Number of tokens in corpus: {st.session_state.ref_tokens}\n    Number of word tokens in corpus: {st.session_state.ref_words}\n    Number of documents in corpus: {st.session_state.ref_ndocs}
			""")
			
			if st.session_state.warning == 4:
				st.markdown(f"""###### The following documents were excluded from the corpus either because they are improperly encoded or have substantial segments not recognized as English (and, therefore, cannot be tagged properly):
			
				{', '.join(st.session_state.ref_exceptions)}
				""")

			
			with st.expander("Documents in reference corpus:"):
				st.write(sorted(st.session_state.ref_docids))
				
		else:
			st.markdown('### Reference corpus:')
			load_ref = st.radio("Would you like to load a reference corpus?", ("No", "Yes"), horizontal=True)
			if load_ref == 'Yes':
			
				if st.session_state.warning == 1:
					st.markdown(warnings.warning_1, unsafe_allow_html=True)
		
				with st.form("ref-form", clear_on_submit=True):
					ref_files = st.file_uploader("Upload your reference corpus", type=["txt"], accept_multiple_files=True, key='reffiles')
					submitted = st.form_submit_button("UPLOAD REFERENCE")
			
					if len(ref_files) > 0:
						all_files = []
						for file in ref_files:
							bytes_data = file.getvalue()
							file_size = len(bytes_data)
							all_files.append(file_size)
						corpus_size = sum(all_files)
						#check for duplicates
						doc_ids = [doc.name for doc in ref_files]
						doc_ids = [doc.replace(" ", "") for doc in doc_ids]
						if len(doc_ids) > len(set(doc_ids)):
							dup_ids = [x for x in doc_ids if doc_ids.count(x) >= 2]
							dup_ids = list(set(dup_ids))
						else:
							dup_ids = []
						
						dup_ref = list(set(st.session_state.docids).intersection(doc_ids))
				
					else:
						corpus_size = 0
						dup_ids = []
						dup_ref = []
		
				if corpus_size > MAX_BYTES:
					st.markdown(warnings.warning_3, unsafe_allow_html=True)
					st.write(corpus_size)
					st.markdown("---")
		
				if len(dup_ids) > 0:
					st.markdown(warnings.warning_2(sorted(dup_ids)), unsafe_allow_html=True)
				
				if len(dup_ref) > 0:
					st.markdown(warnings.warning_4(sorted(dup_ref)), unsafe_allow_html=True)

				if len(ref_files) > 0 and len(dup_ids) == 0 and len(dup_ref) == 0 and corpus_size <= MAX_BYTES:
					st.markdown(f"""```
					{len(ref_files)} reference corpus files ready to be processed! Use the button on the sidebar.
					""")

				if len(ref_files) > 0 and len(dup_ids) == 0 and len(dup_ref) == 0 and corpus_size <= MAX_BYTES:
					st.sidebar.markdown("### Process Reference")
					st.sidebar.markdown("Click the button to process your reference corpus files.")
					if st.sidebar.button("Process Reference Corpus"):
						with st.sidebar:
							with st.spinner('Processing corpus data...'):
								models = load_models()
								nlp = models[st.session_state.model]
								detector = load_detector()
								ref_corp, exceptions = process_functions.process_corpus(ref_files, detector, nlp)
							if len(exceptions) > 0 and bool(ref_corp) == False:
								st.session_state.warning = 1
								st.error('There was a problem proccessing your reference corpus.')
								st.experimental_rerun()
							elif len(exceptions) > 0 and bool(ref_corp) == True:
								st.warning('There was a problem proccessing your reference corpus.')
								st.session_state.warning = 4
								st.session_state.ref_exceptions = exceptions
								#get features
								tags_pos, tags_ds = process_functions.get_corpus_features(ref_corp)
								#assign session states
								st.session_state.ref_tokens = len(tags_pos)
								st.session_state.ref_words = len([x for x in tags_pos if not x.startswith('Y')])
								st.session_state.reference = ref_corp
								st.session_state.ref_docids = list(ref_corp.keys())
								st.session_state.ref_ndocs = len(list(ref_corp.keys()))
								st.experimental_rerun()
							else:
								st.success('Processing complete!')
								st.session_state.warning = 0
								#get features
								tags_pos, tags_ds = process_functions.get_corpus_features(ref_corp)
								#assign session states
								st.session_state.ref_tokens = len(tags_pos)
								st.session_state.ref_words = len([x for x in tags_pos if not x.startswith('Y')])
								st.session_state.reference = ref_corp
								st.session_state.ref_docids = list(ref_corp.keys())
								st.session_state.ref_ndocs = len(list(ref_corp.keys()))
								st.experimental_rerun()
								
					st.sidebar.markdown("---")
		
		st.sidebar.markdown('### Reset all tools and files:')
		st.sidebar.markdown(":warning: Using the **reset** button will cause all files, tables, and plots to be cleared.")
		if st.sidebar.button("Reset Corpus"):
			for key in st.session_state.keys():
				del st.session_state[key]
			for key, value in _states.STATES.items():
				if key not in st.session_state:
					setattr(st.session_state, key, value)
			st.experimental_rerun()
		st.sidebar.markdown("""---""")
	
	else:
	
		st.markdown("### Processing a target corpus :dart:")
		st.markdown(":warning: Be sure that all file names are unique.")
		
		if st.session_state.warning == 1:
			st.markdown(warnings.warning_1, unsafe_allow_html=True)
		
		with st.form("corpus-form", clear_on_submit=True):
			corp_files = st.file_uploader("Upload your target corpus", type=["txt"], accept_multiple_files=True)
			submitted = st.form_submit_button("UPLOAD TARGET")
			
			if len(corp_files) > 0:
				all_files = []
				for file in corp_files:
					bytes_data = file.getvalue()
					file_size = len(bytes_data)
					all_files.append(file_size)
				corpus_size = sum(all_files)
				#check for duplicates
				doc_ids = [doc.name for doc in corp_files]
				doc_ids = [doc.replace(" ", "") for doc in doc_ids]
				if len(doc_ids) > len(set(doc_ids)):
					dup_ids = [x for x in doc_ids if doc_ids.count(x) >= 2]
					dup_ids = list(set(dup_ids))
				else:
					dup_ids = []
				
			else:
				corpus_size = 0
				dup_ids = []
		
		if corpus_size > MAX_BYTES:
			st.markdown(warnings.warning_3, unsafe_allow_html=True)
			st.markdown("---")
		
		if len(dup_ids) > 0:
					st.markdown(warnings.warning_2(sorted(dup_ids)), unsafe_allow_html=True)

		if len(corp_files) > 0 and len(dup_ids) == 0 and corpus_size <= MAX_BYTES:
			st.markdown(f"""```
			{len(corp_files)} target corpus files ready to be processed! Use the button on the sidebar.
			""")

		st.markdown("""
					From this page you can load a corpus from a selection of text (**.txt**)
					files or reset a corpus once one has been processed.\n
					Once you have loaded a target corpus, you can add a reference corpus for comparison.
					Also note that you can encode metadata into your filenames, which can used for further analysis.
					(See naming tips.)\n
					The tool is designed to work with smallish corpora **(~3 million words or less)**.
					Processing times may vary, but you can expect the initial corpus processing to take roughly 1 minute for every 1 million words.
					""")
		
		with st.expander("File preparation and file naming tips"):
			st.markdown("""
					Files must be in a \*.txt format. If you are preparing files for the first time,
					it is recommended that you use a plain text editor (rather than an application like Word).
					Avoid using spaces in file names.
					Also, you needn't worry about preserving paragraph breaks, as those will be stripped out during processing.\n
					Metadata can be encoded at the beginning of a file name, before an underscore. For example: acad_01.txt, acad_02.txt, 
					blog_01.txt, blog_02.txt. These would allow you to compare **acad** vs. **blog** as categories.
					You can designate up to 20 categories.
					""")
				
		st.sidebar.markdown("### Models")
		models = load_models()
		selected_dict = st.sidebar.selectbox("Select a DocuScope Model", options=["Large Dictionary", "Common Dictionary"])
		nlp = models[selected_dict]
		st.session_state.model = selected_dict
	
		with st.sidebar.expander("Which model do I choose?"):
			st.markdown("""
					For detailed descriptions, see the tags tables available from the Help menu.
					But in short, the full dictionary has more categories and coverage than the common dictionary.
					""")		
		st.sidebar.markdown("---")
				
		if len(corp_files) > 0 and len(dup_ids) == 0 and corpus_size <= MAX_BYTES:
			st.sidebar.markdown("### Process Target")
			st.sidebar.markdown("Once you have selected your files, use the button to process your corpus.")
			if st.sidebar.button("Process Corpus"):
				with st.sidebar:
					with st.spinner('Processing corpus data...'):
						detector = load_detector()
						corp, exceptions = process_functions.process_corpus(corp_files, detector, nlp)
					if len(exceptions) > 0 and bool(corp) == False:
						st.session_state.warning = 1
						st.error('There was a problem proccessing your corpus.')
						st.experimental_rerun()
					elif len(exceptions) > 0 and bool(corp) == True:
						st.warning('There was a problem proccessing your corpus.')
						st.session_state.warning = 3
						st.session_state.exceptions = exceptions
						#get features
						tags_pos, tags_ds = process_functions.get_corpus_features(corp)
						#assign session states
						st.session_state.tokens = len(tags_pos)
						st.session_state.words = len([x for x in tags_pos if not x.startswith('Y')])
						st.session_state.corpus = corp
						st.session_state.docids = list(corp.keys())
						st.session_state.ndocs = len(list(corp.keys()))
						#tagsets
						tags_ds = set(tags_ds)
						tags_ds = sorted(set([re.sub(r'B-', '', i) for i in tags_ds]))
						tags_pos = set(tags_pos)
						tags_pos = sorted(set([re.sub(r'\d\d$', '', i) for i in tags_pos]))
						st.session_state.tags_ds = tags_ds
						st.session_state.tags_pos = tags_pos
						st.experimental_rerun()
					else:
						st.success('Processing complete!')
						st.session_state.warning = 0
						#get features
						tags_pos, tags_ds = process_functions.get_corpus_features(corp)
						#assign session states
						st.session_state.tokens = len(tags_pos)
						st.session_state.words = len([x for x in tags_pos if not x.startswith('Y')])
						st.session_state.corpus = corp
						st.session_state.docids = list(corp.keys())
						st.session_state.ndocs = len(list(corp.keys()))
						#tagsets
						tags_ds = set(tags_ds)
						tags_ds = sorted(set([re.sub(r'B-', '', i) for i in tags_ds]))
						tags_pos = set(tags_pos)
						tags_pos = sorted(set([re.sub(r'\d\d$', '', i) for i in tags_pos]))
						st.session_state.tags_ds = tags_ds
						st.session_state.tags_pos = tags_pos
						st.experimental_rerun()
			st.sidebar.markdown("---")

if __name__ == "__main__":
    main()
