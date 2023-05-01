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

import atexit
import glob
import os
import pathlib
import pickle
import re
import shutil
import tempfile
from importlib.machinery import SourceFileLoader


HERE = pathlib.Path(__file__).parents[1].resolve()
TEMP_DIR = HERE.joinpath("_temp")
CORPUS_DIR = HERE.joinpath("_corpora")
OPTIONS = str(HERE.joinpath("options.toml"))
IMPORTS = str(HERE.joinpath("utilities/handlers_imports.py"))

# import options
_imports = SourceFileLoader("handlers_imports", IMPORTS).load_module()
_options = _imports.import_options_general(OPTIONS)

modules = ['states', 'streamlit', 'pandas', 'blosc']
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

# Functions for handling states and files.
# Handling can be done either by storing temporary files locally or by storing data in session memory.
# Thus, functions are written in matching pairs, depending on how data is to be stored.
# These functions can be imported either from handlers_server.py or from handlers_local.py as _handlers
# When imported, then, a given function call is the same, taking the same arguments.
# And session data can be imported and stored as needed.

# Handling is set by the enable_save boolean value in options.toml

# Initialize session states.
# For local file handling, generate a temp folder to be deleted on close.

def generate_temp(states):
	for key, value in states:
		if key not in st.session_state:
			setattr(st.session_state, key, value)
	subdirs = len([x[0] for x in os.walk(TEMP_DIR)])
	# Create temp directory if one doesn't exist
	if subdirs < 2:
		DATA_DIR = tempfile.mkdtemp(dir=str(TEMP_DIR))
		init_session(TEMP_DIR, DATA_DIR)
		atexit.register(shutil.rmtree, DATA_DIR)
	# Check for unwanted temp directories and delete before generating temp       
	if subdirs > 2:
		DATA_DIR = tempfile.mkdtemp(dir=str(TEMP_DIR))
		init_session(TEMP_DIR, DATA_DIR)
		temps = [x[0] for x in os.walk(str(TEMP_DIR))]
		temps = set(temps)
		to_delete = temps.symmetric_difference(set([str(TEMP_DIR), DATA_DIR]))
		to_delete = list(to_delete)
		for dirs in to_delete:
			shutil.rmtree(dirs)
		atexit.register(shutil.rmtree, DATA_DIR)

def data_path():
	try:
		session_path = str(TEMP_DIR.joinpath('session.pkl'))
		with open(session_path, 'rb') as file:
			session = pickle.load(file)
		data_dir = session.get('data_dir')
		DATA_DIR = pathlib.Path(data_dir)
		if os.path.exists(DATA_DIR) == True:
			return(DATA_DIR)
		else:
			subdirs = len([x[0] for x in os.walk(str(TEMP_DIR))])
			if subdirs == 2:
				data_dir = [x[0] for x in os.walk(str(TEMP_DIR))]
				DATA_DIR = pathlib.Path(data_dir[1])
				update_session('data_dir', DATA_DIR)
				return(DATA_DIR)
			else:
				generate_temp()
				data_dir = [x[0] for x in os.walk(str(TEMP_DIR))]
				DATA_DIR = pathlib.Path(data_dir[1])
				update_session('data_dir', DATA_DIR)
				return(DATA_DIR)
	except:
		generate_temp()
		data_dir = [x[0] for x in os.walk(str(TEMP_DIR))]
		DATA_DIR = pathlib.Path(data_dir[1])
		update_session('data_dir', DATA_DIR)
		return(DATA_DIR)

def clear_temp():
	DATA_DIR = data_path()
	for filename in os.listdir(DATA_DIR):
		file_path = os.path.join(DATA_DIR, filename)
		try:
			if os.path.isfile(file_path) or os.path.islink(file_path):
				os.unlink(file_path)
			elif os.path.isdir(file_path):
				shutil.rmtree(file_path)
		except:
			pass
			
# Set session values.

def init_session(tempdir, datadir):
	session = {}
	session['data_dir'] = datadir
	session['target_path'] = None
	session['from_saved'] = 'No'
	session['is_saved'] = 'No'
	session['has_meta'] = False
	session['has_reference'] = False
	session['reference_path'] = None
	session['freq_table'] = False
	session['tags_table'] = False
	session['keyness_table'] = False
	session['ngrams'] = False
	session['kwic'] = False
	session['keyness_parts'] = {}
	session['dtm'] = {}
	session['pca'] = {}
	session['collocations'] = {}
	session['doc'] = {}
	file_path = str(tempdir.joinpath('session.pkl'))
	with open(file_path, 'wb') as file:
		pickle.dump(session, file)

# Functions for managing session values.

def load_session():
	session_path = str(TEMP_DIR.joinpath('session.pkl'))
	try:
		with open(session_path, 'rb') as file:
			session = pickle.load(file)
		return(session)
	except:
		generate_temp(_states.STATES.items())
	
def reset_session():
	DATA_DIR = data_path()
	session = {}
	session['data_dir'] = DATA_DIR
	session['target_path'] = None
	session['from_saved'] = 'No'
	session['is_saved'] = 'No'
	session['has_meta'] = False
	session['has_reference'] = False
	session['reference_path'] = None
	session['freq_table'] = False
	session['tags_table'] = False
	session['keyness_table'] = False
	session['ngrams'] = False
	session['kwic'] = False
	session['keyness_parts'] = {}
	session['dtm'] = {}
	session['pca'] = {}
	session['collocations'] = {}
	session['doc'] = {}
	file_path = str(TEMP_DIR.joinpath('session.pkl'))
	with open(file_path, 'wb') as file:
		pickle.dump(session, file)

def update_session(key, value):
	session_path = str(TEMP_DIR.joinpath('session.pkl'))
	with open(session_path, 'rb') as file:
		session = pickle.load(file)
	session[key] = value
	with open(session_path, 'wb') as file:
		pickle.dump(session, file)

# Functions for storing and managing corpus metadata

def init_metadata_target(corpus, model, tags_pos, tags_ds):
	DATA_DIR = data_path()
	temp_metadata_target = {}
	temp_metadata_target['tokens'] = len(tags_pos)
	temp_metadata_target['words'] = len([x for x in tags_pos if not x.startswith('Y')])
	temp_metadata_target['docids'] = list(corpus.keys())
	temp_metadata_target['ndocs'] = len(list(corpus.keys()))
	temp_metadata_target['model'] = model
	temp_metadata_target['doccats'] = None
	tagset_ds = set(tags_ds)
	tagset_ds = sorted(set([re.sub(r'B-', '', i) for i in tagset_ds]))
	tagset_pos = set(tags_pos)
	tagset_pos = sorted(set([re.sub(r'\d\d$', '', i) for i in tagset_pos]))
	temp_metadata_target['tags_ds'] = tagset_ds
	temp_metadata_target['tags_pos'] = tagset_pos
	file_name = 'temp_metadata_target.pkl'
	file_path = str(DATA_DIR.joinpath(file_name))
	with open(file_path, 'wb') as file:
		pickle.dump(temp_metadata_target, file)

def init_metadata_reference(corpus, model, tags_pos, tags_ds):
	DATA_DIR = data_path()
	temp_metadata_reference = {}
	temp_metadata_reference['tokens'] = len(tags_pos)
	temp_metadata_reference['words'] = len([x for x in tags_pos if not x.startswith('Y')])
	temp_metadata_reference['docids'] = list(corpus.keys())
	temp_metadata_reference['ndocs'] = len(list(corpus.keys()))
	temp_metadata_reference['model'] = model
	temp_metadata_reference['doccats'] = None
	tagset_ds = set(tags_ds)
	tagset_ds = sorted(set([re.sub(r'B-', '', i) for i in tagset_ds]))
	tagset_pos = set(tags_pos)
	tagset_pos = sorted(set([re.sub(r'\d\d$', '', i) for i in tagset_pos]))
	temp_metadata_reference['tags_ds'] = tagset_ds
	temp_metadata_reference['tags_pos'] = tagset_pos
	file_name = 'temp_metadata_reference.pkl'
	file_path = str(DATA_DIR.joinpath(file_name))
	with open(file_path, 'wb') as file:
		pickle.dump(temp_metadata_reference, file)

def load_metadata(corpus_type):
	DATA_DIR = data_path()
	file_name = 'temp_metadata_' + corpus_type + '.pkl'
	file_path = str(DATA_DIR.joinpath(file_name))
	with open(file_path, 'rb') as file:
		metadata = pickle.load(file)
	return(metadata)

def update_metadata(corpus_type, key, value):
	DATA_DIR = data_path()
	file_name = 'temp_metadata_' + corpus_type + '.pkl'
	file_path = str(DATA_DIR.joinpath(file_name))
	with open(file_path, 'rb') as file:
		metadata = pickle.load(file)
	metadata[key] = value
	with open(file_path, 'wb') as file:
		pickle.dump(metadata, file)

# Functions for handling corpora

def check_model(tagset):
	tags_to_check = tagset
	tags = ['Actors', 'Organization', 'Planning', 'Sentiment', 'Signposting', 'Stance']
	if any(tag in item for item in tags_to_check for tag in tags):
		model = 'Common Dictionary'
	else:
		model = 'Large Dictionary'
	return(model)

def load_corpus_path(file_path):
	try:
		with open(file_path, 'rb') as file:
			compressed_corpus = file.read()
		depressed_corpus = blosc.decompress(compressed_corpus)
		corpus = pickle.loads(depressed_corpus)
		return(corpus)
	except:
		pass

def load_corpus_session(corpus_id, session_data):
	corpus = corpus_id + '_path'
	corpus_path = str(session_data.get(corpus))
	corpus_name = 'temp_' + corpus_id
	if corpus_path == 'session':
		corpus = st.session_state.data[corpus_name]
		return(corpus)
	else:	
		try:
			with open(corpus_path, 'rb') as file:
				compressed_corpus = file.read()
			depressed_corpus = blosc.decompress(compressed_corpus)
			corpus = pickle.loads(depressed_corpus)
			return(corpus)
		except:
			st.session_state.session[corpus_name] = None

def load_temp(corpus_type):
	DATA_DIR = data_path()
	file_name = 'temp_' + corpus_type + '.dat'
	file_path = str(DATA_DIR.joinpath(file_name))
	with open(file_path, 'rb') as file:
		compressed_corpus = file.read()
	depressed_corpus = blosc.decompress(compressed_corpus)
	corpus = pickle.loads(depressed_corpus)
	return(corpus)

def save_corpus(corpus, model_name: str, corpus_name: str):
	model_dir = ''.join(word[0] for word in model_name.lower().split())
	file_name = corpus_name + '.dat'
	file_path = str(CORPUS_DIR.joinpath(model_dir, file_name))
	pickled_corpus = pickle.dumps(corpus)
	compressed_corpus = blosc.compress(pickled_corpus)
	with open(file_path, 'wb') as file:
		file.write(compressed_corpus)

def save_corpus_temp(corpus, corpus_type: str):
	DATA_DIR = data_path()
	file_name = 'temp_' + corpus_type + '.dat'
	file_path = str(DATA_DIR.joinpath(file_name))
	key = corpus_type + '_path'
	update_session(key, file_path)
	pickled_corpus = pickle.dumps(corpus)
	compressed_corpus = blosc.compress(pickled_corpus)
	with open(file_path, 'wb') as file:
		file.write(compressed_corpus)

def find_saved(model_type: str):
	SUB_DIR = CORPUS_DIR.joinpath(model_type)
	saved_paths = list(pathlib.Path(SUB_DIR).glob('*.dat'))
	saved_names = [os.path.splitext(os.path.basename(filename))[0] for filename in saved_paths]
	saved_corpora = {saved_names[i]: saved_paths[i] for i in range(len(saved_names))}
	return(saved_corpora)

def find_saved_reference(target_model, target_path):
	model_type = ''.join(word[0] for word in target_model.lower().split())
	SUB_DIR = CORPUS_DIR.joinpath(model_type)
	saved_paths = list(pathlib.Path(SUB_DIR).glob('*.dat'))
	saved_names = [os.path.splitext(os.path.basename(filename))[0] for filename in saved_paths]
	saved_corpora = {saved_names[i]: saved_paths[i] for i in range(len(saved_names))}
	saved_ref = {key:val for key, val in saved_corpora.items() if val != target_path}
	return(saved_corpora, saved_ref)

# Functions for handling data tables

def clear_table(table_id):
	DATA_DIR = data_path()
	file_name = 'temp_' + table_id + '.pkl'
	file_path = str(DATA_DIR.joinpath(file_name))
	try:
		if os.path.isfile(file_path) or os.path.islink(file_path):
			os.unlink(file_path)
	except:
		pass

def load_table(table_id):
	DATA_DIR = data_path()
	file_name = 'temp_' + table_id + '.pkl'
	file_path = str(DATA_DIR.joinpath(file_name))
	table = pd.read_pickle(file_path)  
	return(table)

def save_table(table, table_id: str):
	file_name = 'temp_' + table_id + '.pkl'
	DATA_DIR = data_path()
	file_path = str(DATA_DIR.joinpath(file_name))
	table.to_pickle(file_path)

# Functions for storing values associated with specific apps

def update_collocations(node_word, stat_mode, to_left, to_right):
	session_path = str(TEMP_DIR.joinpath('session.pkl'))
	with open(session_path, 'rb') as file:
		session = pickle.load(file)
	temp_coll = {}
	temp_coll['node']   = node_word
	temp_coll['stat']   = stat_mode
	temp_coll['span_l'] = to_left
	temp_coll['span_r'] = to_right
	session['collocations'].update(temp_coll)
	with open(session_path, 'wb') as file:
		pickle.dump(session, file)

def update_doc(dc_pos, dc_simple, dc_ds, html_pos, html_simple, html_ds, doc_key):
	session_path = str(TEMP_DIR.joinpath('session.pkl'))
	with open(session_path, 'rb') as file:
		session = pickle.load(file)
	temp_doc = {}
	temp_doc['dc_pos']      = dc_pos
	temp_doc['dc_simple']   = dc_simple
	temp_doc['dc_ds']       = dc_ds
	temp_doc['html_pos']    = html_pos
	temp_doc['html_simple'] = html_simple
	temp_doc['html_ds']     = html_ds
	temp_doc['doc_key']     = doc_key
	session['doc'].update(temp_doc)
	with open(session_path, 'wb') as file:
		pickle.dump(session, file)

def update_dtm(sums_pos, sums_ds, units):
	session_path = str(TEMP_DIR.joinpath('session.pkl'))
	with open(session_path, 'rb') as file:
		session = pickle.load(file)
	temp_dtm = {}
	temp_dtm['sums_pos'] = sums_pos
	temp_dtm['sums_ds']  = sums_ds
	temp_dtm['units']    = units
	session['dtm'].update(temp_dtm)
	with open(session_path, 'wb') as file:
		pickle.dump(session, file)

def update_keyness_parts(tar_words, ref_words, tar_tokens, ref_tokens, tar_ndocs, ref_ndocs, tar_cats, ref_cats):
	session_path = str(TEMP_DIR.joinpath('session.pkl'))
	with open(session_path, 'rb') as file:
		session = pickle.load(file)
	temp_kp = {}
	temp_kp['tar_words']  = tar_words
	temp_kp['ref_words']  = ref_words
	temp_kp['tar_tokens'] = tar_tokens
	temp_kp['ref_tokens'] = ref_tokens
	temp_kp['tar_ndocs']  = tar_ndocs
	temp_kp['ref_ndocs']  = ref_ndocs
	temp_kp['tar_cats']   = tar_cats
	temp_kp['ref_cats']   = ref_cats
	session['keyness_parts'].update(temp_kp)
	with open(session_path, 'wb') as file:
		pickle.dump(session, file)

def update_pca(pca, contrib, variance, pca_idx):
	session_path = str(TEMP_DIR.joinpath('session.pkl'))
	with open(session_path, 'rb') as file:
		session = pickle.load(file)
	temp_pca = {}
	temp_pca['pca']      = pca
	temp_pca['contrib']  = contrib
	temp_pca['variance'] = variance
	temp_pca['pca_idx']  = pca_idx
	session['pca'].update(temp_pca)
	with open(session_path, 'wb') as file:
		pickle.dump(session, file)

def update_tags(html_state):
	html_highlights = [' { background-color:#5fb7ca; }', ' { background-color:#e35be5; }', ' { background-color:#ffc701; }', ' { background-color:#fe5b05; }', ' { background-color:#cb7d60; }']
	if 'html_str' not in st.session_state:
		st.session_state['html_str'] = ''
	if 'tags' in st.session_state:
		tags = st.session_state['tags']
		if len(tags)>5:
			tags = tags[:5]
			st.session_state['tags'] = tags
	else:
		tags = []
	tags = ['.' + x for x in tags]
	highlights = html_highlights[:len(tags)]
	style_str = [''.join(x) for x in zip(tags, highlights)]
	style_str = ''.join(style_str)
	style_sheet_str = '<style>' + style_str + '</style>'
	st.session_state['html_str'] = style_sheet_str + html_state

# Convenience function called by widgets

def check_name(strg, search=re.compile(r'[^A-Za-z0-9_-]').search):
	return not bool(search(strg))

def clear_plots():
	update_session('pca', dict())
	if 'grpa' in st.session_state:
		st.session_state['grpa'] = []
	if 'grpb' in st.session_state:
		st.session_state['grpb'] = []

def persist(key: str, app_name: str):
	_PERSIST_STATE_KEY = f"{app_name}_PERSIST"
	if _PERSIST_STATE_KEY not in st.session_state.keys():
		st.session_state[_PERSIST_STATE_KEY] = {}
		st.session_state[_PERSIST_STATE_KEY][key] = None
    
	if key in st.session_state:
		st.session_state[_PERSIST_STATE_KEY][key] = st.session_state[key]
    	
	return key
	
def load_widget_state(app_name: str):
	_PERSIST_STATE_KEY = f"{app_name}_PERSIST"
	"""Load persistent widget state."""
	if _PERSIST_STATE_KEY in st.session_state:
		for key in st.session_state[_PERSIST_STATE_KEY]:
			if st.session_state[_PERSIST_STATE_KEY][key] is not None:
				if key not in st.session_state:
					st.session_state[key] = st.session_state[_PERSIST_STATE_KEY][key]

#prevent categories from being chosen in both multiselect
def update_grpa():
	if len(list(set(st.session_state.grpa) & set(st.session_state.grpb))) > 0:
		item = list(set(st.session_state.grpa) & set(st.session_state.grpb))
		st.session_state.grpa = list(set(list(st.session_state.grpa))^set(item))

def update_grpb():
	if len(list(set(st.session_state.grpa) & set(st.session_state.grpb))) > 0:
		item = list(set(st.session_state.grpa) & set(st.session_state.grpb))
		st.session_state.grpb = list(set(list(st.session_state.grpb))^set(item))

#prevent categories from being chosen in both multiselect
def update_tar():
	if len(list(set(st.session_state.tar) & set(st.session_state.ref))) > 0:
		item = list(set(st.session_state.tar) & set(st.session_state.ref))
		st.session_state.tar = list(set(list(st.session_state.tar))^set(item))

def update_ref():
	if len(list(set(st.session_state.tar) & set(st.session_state.ref))) > 0:
		item = list(set(st.session_state.tar) & set(st.session_state.ref))
		st.session_state.ref = list(set(list(st.session_state.ref))^set(item))

