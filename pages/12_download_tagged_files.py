import streamlit as st

# NLP Pkgs
import docuscospacy.corpus_analysis as ds

import base64
from io import BytesIO
import zipfile


if 'corpus' not in st.session_state:
	st.session_state.corpus = ''

if st.session_state.corpus != '':

	tp = st.session_state.corpus

	if st.button("Download Tagged Files"):
		with st.spinner('Creating download link...'):
			zip_buf = BytesIO()
			with zipfile.ZipFile(zip_buf, 'w', zipfile.ZIP_DEFLATED) as file_zip:
				for key in tp.keys():
					doc_id = key.replace(r'\.txt$', '')
					df = ds.tag_ruler(tp, key, count_by='pos')
					df['Token'] = df['Token'].str.strip()
					df['Token'] = df['Token'].str.replace(' ','_')
					df['Tag'] = df['Tag'].str.replace('Untagged','')
					df['Tag'] = df['Tag'].str.replace(r'^Y','')
					df['Tag'] = df['Tag'].str.replace(r'^FU','')
					df['Token'] = df['Token'] + '|' + df['Tag']
					df['Token'] = df['Token'].str.replace(r'\|$', '')
					doc = ' '.join(df['Token'])
					file_zip.writestr(doc_id + "_tagged"+".txt", doc)
    				
			zip_buf.seek(0)
			#pass it to front end for download
			b64 = base64.b64encode(zip_buf.read()).decode()
			del zip_buf
			st.success('Link generated!')
			href = f'<a href=\"data:file/zip;base64,{b64}\" download="tagged_files.zip">Download tagged files</a>'
			st.markdown(href, unsafe_allow_html=True)


else:
	st.write("It doesn't look like you've loaded a corpus yet.")