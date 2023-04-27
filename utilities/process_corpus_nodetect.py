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

import math
import random
import re
import string
import unidecode

def pre_process(txt):
	txt = re.sub(r'\bits\b', 'it s', txt)
	txt = re.sub(r'\bIts\b', 'It s', txt)
	txt = " ".join(txt.split())
	return(txt)

def process_corpus(corp, nlp_model):
	doc_ids = [doc.name for doc in corp]
	doc_ids = [doc.replace(" ", "") for doc in doc_ids]
	is_punct = re.compile("[{}]+\s*$".format(re.escape(string.punctuation)))
	is_digit = re.compile("\d[\d{}]*\s*$".format(re.escape(string.punctuation)))
	tp = {}
	exceptions = []
	dup_ids = []
	for doc in corp:
		try:
			doc_txt = doc.getvalue().decode('utf-8')
		except:
			exceptions.append(doc.name)
		else:
			doc_txt = unidecode.unidecode(doc_txt)
			doc_id = doc.name.replace(" ", "")
			doc_txt = pre_process(doc_txt)
			doc_len = len(doc_txt)
			if doc_len > 1000000:
				n_chunks = math.ceil(doc_len/750000)
				chunk_idx = [math.ceil(i/n_chunks*doc_len) for i in range(1, n_chunks)]
				try:
					split_idx = [re.search('[\.\?!] [A-Z]', doc_txt[idx:]).span()[1] + (idx-1) for idx in chunk_idx]
				except:
					try:
						split_idx = [re.search(' ', doc_txt[idx:]).span()[0] + idx for idx in chunk_idx]
					except :
						exceptions.append(doc.name)
				else:
					split_idx.insert(0, 0)
					doc_chunks = [doc_txt[i:j] for i, j in zip(split_idx, split_idx[1:]+[None])]
					token_list = []
					tag_list = []
					iob_ent = []
					for chunk in doc_chunks:
						chunk_taged = nlp_model(chunk)
						token_chunk = [token.text for token in chunk_taged]
						ws_chunk = [token.whitespace_ for token in chunk_taged]
						token_chunk = list(map(''.join, zip(token_chunk, ws_chunk)))
						iob_chunk = [token.ent_iob_ for token in chunk_taged]
						ent_chunk = [token.ent_type_ for token in chunk_taged]
						iob_ent_chunk = list(map('-'.join, zip(iob_chunk, ent_chunk)))
						tag_chunk = [token.tag_ for token in chunk_taged]
						tag_chunk = ['Y' if bool(is_punct.match(token_chunk[i])) else v for i, v in enumerate(tag_chunk)]
						tag_chunk = ['MC' if bool(is_digit.match(token_chunk[i])) and tag_chunk[i] != 'Y' else v for i, v in enumerate(tag_chunk)]
						token_list.append(token_chunk)
						tag_list.append(tag_chunk)
						iob_ent.append(iob_ent_chunk)
					token_list = [item for sublist in token_list for item in sublist]
					tag_list = [item for sublist in tag_list for item in sublist]
					iob_ent = [item for sublist in iob_ent for item in sublist]
					tp.update({doc_id: (list(zip(token_list, tag_list, iob_ent)))})
			else:
				doc_taged = nlp_model(doc_txt)
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
	return tp, exceptions

def get_corpus_features(corpus):
	tok = list(corpus.values())
	tags_pos = []
	for i in range(0,len(tok)):
		tags = [x[1] for x in tok[i]]
		tags_pos.append(tags)
	tags_pos = [x for xs in tags_pos for x in xs]
	#get ds tags
	tags_ds = []
	for i in range(0,len(tok)):
		tags = [x[2] for x in tok[i]]
		tags_ds.append(tags)
	tags_ds = [x for xs in tags_ds for x in xs]
	tags_ds = [x for x in tags_ds if x.startswith('B-')]
	return tags_pos, tags_ds

def check_corpus(docs, check_size=False, check_ref=False, target_docs=None):
	if len(docs) > 0:
		all_files = []
		if check_size:
			for file in docs:
				bytes_data = file.getvalue()
				file_size = len(bytes_data)
				all_files.append(file_size)
			corpus_size = sum(all_files)
		#check for duplicates
		doc_ids = [doc.name for doc in docs]
		doc_ids = [doc.replace(" ", "") for doc in doc_ids]
		if len(doc_ids) > len(set(doc_ids)):
			dup_ids = [x for x in doc_ids if doc_ids.count(x) >= 2]
			dup_ids = list(set(dup_ids))
		else:
			dup_ids = []
		if check_ref and target_docs is not None:
			dup_docs = list(set(target_docs).intersection(doc_ids))
	else:
		corpus_size = 0
		dup_ids = []
		dup_docs = []
	if check_ref and check_size:
		return(dup_ids, dup_docs, corpus_size)
	elif check_ref and not check_size:
		return(dup_ids, dup_docs)
	elif check_size and not check_ref:
		return(dup_ids, corpus_size)
	else:
		return(dup_ids)

def check_model(tagset):
	tags_to_check = tagset
	tags = ['Actors', 'Organization', 'Planning', 'Sentiment', 'Signposting', 'Stance']
	if any(tag in item for item in tags_to_check for tag in tags):
		model = 'Common Dictionary'
	else:
		model = 'Large Dictionary'
	return(model)

def check_reference(corpus, target_docs):
	doc_ids = list(corpus.keys())
	dup_docs = list(set(target_docs).intersection(doc_ids))
	for dup in dup_docs:
		corpus.pop(dup, None)
	return corpus, dup_docs
