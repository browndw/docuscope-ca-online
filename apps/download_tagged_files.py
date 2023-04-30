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
import re
import zipfile
from importlib.machinery import SourceFileLoader

# set paths
HERE = pathlib.Path(__file__).parents[1].resolve()
OPTIONS = str(HERE.joinpath("options.toml"))
IMPORTS = str(HERE.joinpath("utilities/handlers_imports.py"))

# import options
_imports = SourceFileLoader("handlers_imports", IMPORTS).load_module()
_options = _imports.import_options_general(OPTIONS)

modules = ['categories', 'handlers', 'messages', 'states', 'warnings', 'streamlit', 'docuscospacy']
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

	
CATEGORY = _categories.OTHER
TITLE = "Download Tagged Files"
KEY_SORT = 11

def main():
	
	session = _handlers.load_session()
	st.markdown(_messages.message_download_tagged)
	
	st.sidebar.markdown("### Tagset to embed")
	tag_radio = st.sidebar.radio("Select tagset:", ("Parts-of-Speech", "DocuScope"), horizontal=True)

	if tag_radio == 'Parts-of-Speech':
		tagset = 'pos'
	else:
		tagset = 'ds'

	if st.sidebar.button("Download Tagged Files"):
		if session.get('target_path') == None:
			st.markdown(_warnings.warning_11, unsafe_allow_html=True)
		else:
			with st.sidebar:
				with st.spinner('Creating download link...'):
					try:
						tp = _handlers.load_corpus_session('target', session)
					except:
						st.markdown(_warnings.warning_11, unsafe_allow_html=True)
					
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
	
	st.sidebar.markdown("---")

if __name__ == "__main__":
    main()
    