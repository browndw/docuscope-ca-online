import streamlit as st

# NLP Pkgs
import spacy
import docuscospacy.corpus_analysis as ds

import re
import string

st.sidebar.markdown("## Load a corpus")

st.title("Load and manage your corpus")


if 'corpus' not in st.session_state:
	st.session_state.corpus = ''

if 'docids' not in st.session_state:
	st.session_state.docids = ''

if 'words' not in st.session_state:
	st.session_state.words = 0

if 'tokens' not in st.session_state:
	st.session_state.tokens = 0

if 'ndocs' not in st.session_state:
	st.session_state.ndocs = 0

nlp = spacy.load('en_docusco_spacy')

def pre_process(txt):
	txt = re.sub(r'\bits\b', 'it s', txt)
	txt = re.sub(r'\bIts\b', 'It s', txt)
	txt = " ".join(txt.split())
	return(txt)

def process_corpus(corp):
	doc_ids = [doc.name for doc in corp]
	if len(doc_ids) > len(set(doc_ids)):
		dup_ids = [x for x in doc_ids if doc_ids.count(x) >= 2]
		st.write("Your documents contain duplicate names: ", dup_ids)
	else:
		is_punct = re.compile("[{}]+\s*$".format(re.escape(string.punctuation)))
		is_digit = re.compile("\d[\d{}]*\s*$".format(re.escape(string.punctuation)))
		tp = {}
		for doc in corp:	
			doc_txt = doc.getvalue().decode("utf-8")
			doc_id = doc.name
			doc_txt = pre_process(doc_txt)
			doc_taged = nlp(doc_txt)
			token_list = [token.text for token in doc_taged]
			ws_list = [token.whitespace_ for token in doc_taged]
			token_list = list(map(''.join, zip(token_list, ws_list)))
			iob_list = [token.ent_iob_ for token in doc_taged]
			ent_list = [token.ent_type_ for token in doc_taged]
			iob_ent = list(map('-'.join, zip(iob_list, ent_list)))
			tag_list = [token.tag_ for token in doc_taged]
			tag_list = ['Y' if bool(is_punct.match(token_list[i])) else v for i, v in enumerate(tag_list)]
			tag_list = ['MC' if bool(is_digit.match(token_list[i])) and tag_list[i] != 'Y' else v for i, v in enumerate(tag_list)]
			tp.update({doc_id: (list(zip(token_list, tag_list, iob_ent)))})
		return tp

if st.session_state.ndocs > 0:
	st.write('Number of tokens in corpus: ', str(st.session_state.tokens))
	st.write('Number of word tokens in corpus: ', str(st.session_state.words))
	st.write('Number of documents in corpus: ', str(st.session_state.ndocs))
	with st.expander("📁 Documents:"):
		st.write(st.session_state.docids)
	
	st.markdown(":warning: Using the **reset** button will cause all files, tables, and plots to be cleared.")
	if st.button("Reset Corpus"):
		for key in st.session_state.keys():
			del st.session_state[key]
		st.experimental_singleton.clear()
		st.experimental_rerun()
else:

	st.markdown("From this page you can load a corpus from a selection of text (**.txt**) files or reset a corpus once one has been processed.")
	st.markdown(":warning: Be sure that all file names are unique.")

	corp_files = st.file_uploader("Upload your corpus", type=["txt"], accept_multiple_files=True)
	
	if len(corp_files) > 0:
		if st.button("Process Corpus"):
			with st.spinner('Processing corpus data...'):
				corp = process_corpus(corp_files)
			if corp == None:
				st.success('Fix or remove duplicate file names before processing corpus.')
			else:
				st.success('Processing complete!')
				tok = list(corp.values())
				tag_list = []
				for i in range(0,len(tok)):
					tags = [x[1] for x in tok[i]]
					tag_list.append(tags)
				tag_list = [x for xs in tag_list for x in xs]
				st.session_state.tokens = len(tag_list)
				st.session_state.words = len([x for x in tag_list if not x.startswith('Y')])
				st.session_state.corpus = corp
				st.session_state.docids = list(corp.keys())
				st.session_state.ndocs = len(list(corp.keys()))
				st.experimental_rerun()
