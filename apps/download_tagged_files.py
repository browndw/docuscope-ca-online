import streamlit as st
import docuscospacy.corpus_analysis as ds

import re
import base64
from io import BytesIO
import zipfile

import categories
import states as _states
	
CATEGORY = categories.OTHER
TITLE = "Download Tagged Files"
KEY_SORT = 11

def main():
	# check states to prevent unlikely error
	for key, value in _states.STATES.items():
		if key not in st.session_state:
			setattr(st.session_state, key, value)

	st.markdown("""Once a corpus has been processed, you can use this page to generate a **zipped folder of tagged text files**. 
	The tags are embbedd into the text after a vertical bar:
	
	At|II root|NN1 , every|AT1 hypothesis|NN1 is|VBZ a|AT1 claim|NN1 about|II the|AT relevance|NN1
	""")

	st.markdown("""Because the tags identify mutliword units, spaces that occur within a token are replaced with underscores:
	
	evidence|Reasoning and|SyntacticComplexity theory|AcademicTerms pertaining_to_the|Reasoning possibility_of|ConfidenceHedged sympatric|Description speciation|Description
	""")

	st.markdown("If you are planning to use the output to process the files in a tool like AntConc or in a coding environment, take note of these conventions and account for them accordingly.")
	
	if st.session_state.corpus != '':
	
		st.sidebar.markdown("### Tagset to embed")
		tag_radio = st.sidebar.radio("Select tagset:", ("Parts-of-Speech", "DocuScope"), horizontal=True)
	
		if tag_radio == 'Parts-of-Speech':
			tagset = 'pos'
		else:
			tagset = 'ds'
	
		tp = st.session_state.corpus
	
		if st.sidebar.button("Download Tagged Files"):
			with st.sidebar:
				with st.spinner('Creating download link...'):
					zip_buf = BytesIO()
					with zipfile.ZipFile(zip_buf, 'w', zipfile.ZIP_DEFLATED) as file_zip:
						for key in tp.keys():
							doc_id = re.sub(r'\.txt$', '', key)
							df = ds.tag_ruler(tp, key, count_by=tagset)
							df['Token'] = df['Token'].str.strip()
							df['Token'] = df['Token'].str.replace(' ','_')
							if tagset == 'ds':
								df['Tag'] = df['Tag'].str.replace('Untagged','')
							else:
								df['Tag'] = df['Tag'].str.replace(r'^Y','')
								df['Tag'] = df['Tag'].str.replace(r'^FU','')
							df['Token'] = df['Token'] + '|' + df['Tag']
							df['Token'] = df['Token'].str.replace(r'\|$', '')
							doc = ' '.join(df['Token'])
							file_zip.writestr(doc_id + "_tagged"+ ".txt", doc)
		    				
					zip_buf.seek(0)
					#pass it to front end for download
					b64 = base64.b64encode(zip_buf.read()).decode()
					del zip_buf
					st.success('Link generated!')
					href = f'<a href=\"data:file/zip;base64,{b64}\" download="tagged_files.zip">Download tagged files</a>'
					st.markdown(href, unsafe_allow_html=True)
	
	else:
		st.write(":neutral_face: It doesn't look like you've loaded a corpus yet.")

if __name__ == "__main__":
    main()
    