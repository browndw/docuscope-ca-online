import streamlit as st

# NLP Pkgs
import docuscospacy.corpus_analysis as ds

import pandas as pd
import plotly.express as px
from collections import Counter

import base64
from io import BytesIO

st.title("Explore single texts")

if 'corpus' not in st.session_state:
	st.session_state.corpus = ''
	
if 'docids' not in st.session_state:
	st.session_state.docids = ''

if 'dc_pos' not in st.session_state:
	st.session_state.dc_pos = ''

if 'dc_ds' not in st.session_state:
	st.session_state.dc_ds = ''

if 'html_pos' not in st.session_state:
	st.session_state.html_pos = ''

if 'html_ds' not in st.session_state:
	st.session_state.html_ds = ''
	
if 'doc_key' not in st.session_state:
	st.session_state.doc_key = ''


if 'count_4' not in st.session_state:
	st.session_state.count_4 = 0

def increment_counter():
	st.session_state.count_4 += 1

if st.session_state.count_4 % 2 == 0:
    idx = 0
else:
    idx = 1
    
def html_build(tok, key, count_by="tag"):
    df = ds.tag_ruler(tok=tok, key=key, count_by=count_by)
    df['ws'] = df['Token'].str.extract(r'(\s+)$')
    df['Token'] = df['Token'].str.replace(r'(\s+)$', '')
    df.Token[df['Tag'] != 'Untagged'] = df['Token'].str.replace(r'^(.*?)$', '\\1</span>')
    df = df.iloc[:,[1,0,4]]
    df.fillna('', inplace=True)
    df.Tag[df['Tag'] != 'Untagged'] = df['Tag'].str.replace(r'^(.*?)$', '<span class="\\1">')
    df.Tag[df['Tag'] == 'Untagged'] = df['Tag'].str.replace('Untagged', '')
    df['Text'] = df['Tag'] + df['Token'] + df['ws']
    doc = ''.join(df['Text'].tolist())
    return(doc)

def doc_counts(doc_span, n_tokens, count_by='pos'):
    if count_by=='pos':
        df = Counter(doc_span[doc_span.Tag != 'Y'].Tag)
        df = pd.DataFrame.from_dict(df, orient='index').reset_index()
        df = df.rename(columns={'index':'Tag', 0:'AF'})
        df['RF'] = df.AF/n_tokens*100
        df.sort_values(by=['AF', 'Tag'], ascending=[False, True], inplace=True)
        df.reset_index(drop=True, inplace=True)
    elif count_by=='ds':
        df = Counter(doc_span.Tag)
        df = pd.DataFrame.from_dict(df, orient='index').reset_index()
        df = df.rename(columns={'index':'Tag', 0:'AF'})
        df = df[df.Tag != 'Untagged']
        df['RF'] = df.AF/n_tokens*100
        df.sort_values(by=['AF', 'Tag'], ascending=[False, True], inplace=True)
        df.reset_index(drop=True, inplace=True)
    return(df)

html_highlights = [' { background-color:#FEB7B3; }', ' { background-color:#97E5D7; }', ' { background-color:#FCF1DD; }', ' { background-color:#FFD4B8; }', ' { background-color:#D2EBD8; }']

if bool(isinstance(st.session_state.dc_pos, pd.DataFrame)) == True:
	st.write("Use the menus to select the tags you would like to highlight.")
	
	tag_radio = st.radio("Select tags to display:", ("Parts-of-Speech", "DocuScope"), index=idx, on_change=increment_counter, horizontal=True)

	if tag_radio == 'Parts-of-Speech':
		tags = st.multiselect('Select tags to highlight', (st.session_state.tags_pos))
		html_str = st.session_state.html_pos
		df = st.session_state.dc_pos
	else:
		tags = st.multiselect('Select tags to highlight', (st.session_state.tags_ds))
		html_str = st.session_state.html_ds
		df = st.session_state.dc_ds
	
	col1, col2 = st.columns([1,1])
	with col1:
		if st.button("Highlight Tags"):
			if len(tags) > 5:
				st.write('You can only hightlight a maximum of 5 tags.')
			else:
				tags = ['.' + x for x in tags]
				highlights = html_highlights[:len(tags)]
				style_str = [''.join(x) for x in zip(tags, highlights)]
				style_str = ''.join(style_str)
				style_sheet_str = '<style>' + style_str + '</style>'
				html_str = style_sheet_str + html_str
	with col2:
		if st.button("Select a new document"):
			st.session_state.dc_pos = ''
			st.session_state.dc_ds = ''
			st.session_state.html_pos = ''
			st.session_state.html_ds = ''
			st.session_state.doc_key = ''
			st.experimental_rerun()

	st.write(st.session_state.doc_key)
	st.markdown(html_str, unsafe_allow_html=True)
	st.dataframe(df)
	



else:
	st.write("Use the menus to select the tags you would like to highlight.")
	
	doc_key = st.selectbox("Select document to view:", (st.session_state.docids))
	if st.button("Process Document"):
		if st.session_state.corpus == '':
			st.write("It doesn't look like you've loaded a corpus yet.")
		else:
			doc_pos = ds.tag_ruler(st.session_state.corpus, doc_key, count_by='pos')
			doc_ds = ds.tag_ruler(st.session_state.corpus, doc_key, count_by='ds')
			doc_tokens = len(doc_pos.index)
			doc_words = len(doc_pos[doc_pos.Tag != 'Y'])
			dc_pos = doc_counts(doc_pos, doc_words, count_by='pos')
			dc_ds = doc_counts(doc_ds, doc_tokens, count_by='ds')
			html_pos = html_build(st.session_state.corpus, doc_key, count_by='pos')
			html_ds = html_build(st.session_state.corpus, doc_key, count_by='ds')
			st.session_state.dc_pos = dc_pos
			st.session_state.dc_ds = dc_ds
			st.session_state.html_pos = html_pos
			st.session_state.html_ds = html_ds
			st.session_state.doc_key = doc_key
			st.experimental_rerun()
