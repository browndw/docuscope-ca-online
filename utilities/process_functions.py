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

import re
import math
import random
import string
import unidecode

def pre_process(txt):
	txt = re.sub(r'\bits\b', 'it s', txt)
	txt = re.sub(r'\bIts\b', 'It s', txt)
	txt = " ".join(txt.split())
	return(txt)

def determine_if_file_is_english(text_str, detect_model):
	doc_len = len(text_str)
	this_D = {}
	if doc_len > 5000:
		idx_a = random.randint(0, doc_len - 1500)
		idx_b = random.randint(0, doc_len - 1500)
		idx_c = random.randint(0, doc_len - 1500)
		sample_a = text_str[idx_a:idx_a + 1000]
		sample_a = " ".join(sample_a.split())
		sample_b = text_str[idx_b:idx_b + 1000]
		sample_b = " ".join(sample_b.split())
		sample_c = text_str[idx_c:idx_c + 1000]
		sample_c = " ".join(sample_c.split())
		text_sample = [sample_a, sample_b, sample_c]
		#get prediction for each chunk
		for chunk in text_sample:  # Language predict each sampled chunk
			language_tuple = detect_model.predict(chunk)
			prediction = language_tuple[0][0].replace('__label__', '')
			value = language_tuple[1][0]
			if prediction not in this_D.keys():
				this_D[prediction] = 0
			this_D[prediction] += value
	else:
		text_str = " ".join(text_str.split())
		language_tuple = detect_model.predict(text_str)
		prediction = language_tuple[0][0].replace('__label__', '')
		value = language_tuple[1][0]
		if prediction not in this_D.keys():
			this_D[prediction] = 0
			this_D[prediction] += value
            
    # Find the max tallied confidence and the sum of all confidences.
	max_value = max(this_D.values())
	sum_of_values = sum(this_D.values())
	# calculate a relative confidence of the max confidence to all
	# confidence scores. Then find the key with the max confidence.
	confidence = max_value / sum_of_values
	max_key = [key for key in this_D.keys()
		if this_D[key] == max_value][0]

	# Only want to know if this is english or not.
	return max_key == 'en' and confidence > .9

def process_corpus(corp, detect_model, nlp_model):
	doc_ids = [doc.name for doc in corp]
	doc_ids = [doc.replace(" ", "") for doc in doc_ids]
	is_punct = re.compile("[{}]+\s*$".format(re.escape(string.punctuation)))
	is_digit = re.compile("\d[\d{}]*\s*$".format(re.escape(string.punctuation)))
	tp = {}
	exceptions = []
	dup_ids = []
	conf_eng = []
	for doc in corp:
		try:
			doc_txt = doc.getvalue().decode('utf-8')
		except:
			exceptions.append(doc.name)
		else:
			doc_txt = unidecode.unidecode(doc_txt)
			is_english = determine_if_file_is_english(doc_txt, detect_model)
			if is_english == False:
				exceptions.append(doc.name)
			else:
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
