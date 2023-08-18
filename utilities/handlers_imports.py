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

import tomli
import sys
import importlib.util

# Loading and processing requires the availability of specific modules that are platform dependent.
def import_options_load(options_path):
	try:
		with open(options_path, mode="rb") as fp:
			options = tomli.load(fp)
	except:
		options = {}
		options['global'] = {}
		options['global']['check_size'] = False
		options['global']['check_language'] = False
		options['global']['enable_save'] = False
		options['global']['desktop_mode'] = False
		options['global']['max_bytes'] = 0

	# language can't be checked on Windows so toggle off.
	if options['global']['check_language'] == True and (sys.platform == "win" or sys.platform == "cygwin"):
		options['global']['check_language'] = False
	# toggle off for desktop
	if options['global']['check_language'] == True and options['global']['desktop_mode'] == True:
		options['global']['check_language'] = False
	# check to be sure required package is installed
	if options['global']['check_language'] == True:
		fasttext_spec = importlib.util.find_spec('fasttext')
		if fasttext_spec is None:
			options['global']['check_language'] = False
	
	return(options)

def import_options_general(options_path):
	try:
		with open(options_path, mode="rb") as fp:
			options = tomli.load(fp)
	except:
		options = {}
		options['global'] = {}
		options['global']['check_size'] = False
		options['global']['check_language'] = False
		options['global']['enable_save'] = False
		options['global']['desktop_mode'] = False
		options['global']['max_bytes'] = 0

	return(options)
			
def import_parameters(options: dict, packages_to_import):

	ENABLE_SAVE = options['global']['enable_save']
	DESKTOP = options['global']['desktop_mode']
	ENABLE_DETECT = options['global']['check_language']
	
	short_names = {'altair': 'alt', 'numpy': 'np', 'pandas': 'pd', 'streamlit': 'st'}
	
	import_param = {}
	if DESKTOP == False:
		for package in packages_to_import:
			if package == 'categories' or package == 'states':
				short_name = '_' + package
				import_param[package] = [short_name, None]
			elif package == 'warnings' or package == 'messages':
				short_name = '_' + package
				import_param[package] = [short_name, 'utilities']
			elif package == 'apps':
				import_param[package] = ['_stable_apps', None]
			elif package == 'content':
				import_param['content_online'] = ['_content', 'utilities']
			elif package == 'analysis':
				import_param['analysis_functions'] = ['_analysis', 'utilities']
			elif ENABLE_SAVE == True and package == 'handlers':
				import_param['handlers_server'] = ['_handlers', 'utilities']
			elif ENABLE_SAVE == False and package == 'handlers':
				import_param['handlers_server'] = ['_handlers', 'utilities']
			elif ENABLE_DETECT == True and package == 'process':
				import_param['process_corpus_detect'] = ['_process', 'utilities']
			elif ENABLE_DETECT == False and package == 'process':
				import_param['process_corpus_nodetect'] = ['_process', 'utilities']
			elif package == 'docuscospacy':
				import_param['corpus_analysis'] = ['ds', 'docuscospacy']
			elif package == 'corpus_utils':
				import_param['corpus_utils'] = ['corpus_utils', 'docuscospacy']
			elif package == 'decomposition':
				import_param['decomposition'] = ['decomposition', 'sklearn']
			elif package in short_names.keys():
				import_param[package] = [short_names[package], None]
			else:
				import_param[package] = [None, None]

	if DESKTOP == True:
		for package in packages_to_import:
			if package == 'categories' or package == 'states':
				short_name = '_' + package
				import_param[package] = [short_name, 'docuscope._streamlit']
			elif package == 'warnings' or package == 'messages':
				short_name = '_' + package
				import_param[package] = [short_name, 'docuscope._streamlit.utilities']
			elif package == 'apps':
				import_param[package] = ['_stable_apps', 'docuscope._streamlit']
			elif package == 'utilities':
				import_param[package] = [None, 'docuscope._streamlit']
			elif package == 'content':
				import_param['content_desktop'] = ['_content', 'docuscope._streamlit.utilities']
			elif package == 'analysis':
				import_param['analysis_functions'] = ['_analysis', 'docuscope._streamlit.utilities']
			elif package == 'docuscospacy':
				import_param['ds'] = [None, 'docuscope._imports']
			elif ENABLE_SAVE == True and package == 'handlers':
				import_param['handlers_local'] = ['_handlers', 'docuscope._streamlit.utilities']
			elif ENABLE_SAVE == False and package == 'handlers':
				import_param['handlers_server'] = ['_handlers', 'docuscope._streamlit.utilities']
			elif package == 'process':
				import_param['process_corpus_nodetect'] = ['_process', 'docuscope._streamlit.utilities']
			elif package == 'decomposition':
				pass
			elif package in short_names.keys():
				import_param[package] = [short_names[package], 'docuscope._imports']
			else:
				import_param[package] = [None, 'docuscope._imports']
	
	return(import_param)



