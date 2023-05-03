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

import pathlib
import re
from collections import Counter
from functools import partial
from typing import Union, List, Callable, Optional
from importlib.machinery import SourceFileLoader

# set paths
HERE = pathlib.Path(__file__).parents[1].resolve()
OPTIONS = str(HERE.joinpath("options.toml"))
IMPORTS = str(HERE.joinpath("utilities/handlers_imports.py"))

# import options
_imports = SourceFileLoader("handlers_imports", IMPORTS).load_module()
_options = _imports.import_options_general(OPTIONS)

modules = ['altair', 'docx', 'numpy', 'docuscospacy', 'corpus_utils', 'pandas', 'scipy', 'sklearn', 'decomposition']
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


# https://github.com/WZBSocialScienceCenter/tmtoolkit/blob/master/tmtoolkit/tokenseq.py
def index_windows_around_matches(matches: np.ndarray, left: int, right: int,
                                 flatten: bool = False, remove_overlaps: bool = True) \
        -> Union[List[List[int]], np.ndarray]:
    if not isinstance(matches, np.ndarray) or matches.dtype != bool:
        raise ValueError('`matches` must be a boolean NumPy array')
    if not isinstance(left, int) or left < 0:
        raise ValueError('`left` must be an integer >= 0')
    if not isinstance(right, int) or right < 0:
        raise ValueError('`right` must be an integer >= 0')

    ind = np.where(matches)[0]
    nested_ind = list(map(lambda x: np.arange(x - left, x + right + 1), ind))

    if flatten:
        if not nested_ind:
            return np.array([], dtype=int)

        window_ind = np.concatenate(nested_ind)
        window_ind = window_ind[(window_ind >= 0) & (window_ind < len(matches))]

        if remove_overlaps:
            return np.sort(np.unique(window_ind))
        else:
            return window_ind
    else:
        return [w[(w >= 0) & (w < len(matches))] for w in nested_ind]

# https://github.com/WZBSocialScienceCenter/tmtoolkit/blob/master/tmtoolkit/tokenseq.py
def pmi(x: np.ndarray, y: np.ndarray, xy: np.ndarray, n_total: Optional[int] = None, logfn: Callable = np.log,
        k: int = 1, normalize: bool = False) -> np.ndarray:
    if not isinstance(k, int) or k < 1:
        raise ValueError('`k` must be a strictly positive integer')

    if k > 1 and normalize:
        raise ValueError('normalization is only implemented for standard PMI with `k=1`')

    if n_total is not None:
        if n_total < 1:
            raise ValueError('`n_total` must be strictly positive')
        x = x/n_total
        y = y/n_total
        xy = xy/n_total

    pmi_val = logfn(xy) - logfn(x * y)

    if k > 1:
        return pmi_val - (1-k) * logfn(xy)
    else:
        if normalize:
            return pmi_val / -logfn(xy)
        else:
            return pmi_val

npmi = partial(pmi, k=1, normalize=True)
pmi2 = partial(pmi, k=2, normalize=False)
pmi3 = partial(pmi, k=3, normalize=False)

def coll_table(tok, node_word, l_span=4, r_span=4, statistic='pmi', count_by='pos', node_tag=None, tag_ignore=False):
    tok = list(tok.values())
    stats = {'pmi', 'npmi', 'pmi2', 'pmi3'}
    if statistic not in stats:
        raise ValueError("results: statistic must be one of %r." % stats)
    if l_span < 0 or l_span > 9:
        raise ValueError("Span must be < " + str(0) + " and > " + str(9))
    if r_span < 0 or r_span > 9:
        raise ValueError("Span must be < " + str(0) + " and > " + str(9))
    if bool(tag_ignore) == True:
        node_tag = None
    if count_by == 'pos':
        tc = corpus_utils._merge_tags(tok)
    if count_by == 'ds':
        tc = corpus_utils._merge_ds(tok)
    in_span = []
    for i in range(0,len(tc)):
        tpf = tc[i]
        # create a boolean vector for node word
        if node_tag is None:
            v = [t[0] == node_word for t in tpf]
        else:
            v = [t[0] == node_word and t[1].startswith(node_tag) for t in tpf]
        if sum(v) > 0:
            # get indices within window around the node
            idx = list(index_windows_around_matches(np.array(v), left=l_span, right=r_span, flatten=False))
            node_idx = [i for i, x in enumerate(v) if x == True]
            # remove node word from collocates
            coll_idx = [np.setdiff1d(idx[i], node_idx[i]) for i in range(len(idx))]
            coll_idx = [x for xs in coll_idx for x in xs]
            coll = [tpf[i] for i in coll_idx]
        else:
            coll = []
        in_span.append(coll)
    in_span = [x for xs in in_span for x in xs]
    tc = [x for xs in tc for x in xs]
    df_total = pd.DataFrame(tc, columns=['Token', 'Tag'])
    if bool(tag_ignore) == True:
        df_total = df_total.drop(columns=['Tag'])
    if bool(tag_ignore) == True:
        df_total = df_total.groupby(['Token']).value_counts().to_frame('Freq Total').reset_index()
    else:
        df_total = df_total.groupby(['Token','Tag']).value_counts().to_frame('Freq Total').reset_index()
    df_span = pd.DataFrame(in_span, columns=['Token', 'Tag'])
    if bool(tag_ignore) == True:
        df_span = df_span.drop(columns=['Tag'])
    if bool(tag_ignore) == True:
        df_span = df_span.groupby(['Token']).value_counts().to_frame('Freq Span').reset_index()
    else:
        df_span = df_span.groupby(['Token','Tag']).value_counts().to_frame('Freq Span').reset_index()
    if node_tag is None:
        node_freq = sum(df_total[df_total['Token'] == node_word]['Freq Total'])
    else:
        node_freq = sum(df_total[(df_total['Token'] == node_word) & (df_total['Tag'].str.startswith(node_tag, na=False))]['Freq Total'])
    if bool(tag_ignore) == True:
        df = pd.merge(df_span, df_total, how='inner', on=['Token'])
    else:
        df = pd.merge(df_span, df_total, how='inner', on=['Token', 'Tag'])
    if statistic=='pmi':
        df['MI'] = pmi(node_freq, df['Freq Total'], df['Freq Span'], sum(df_total['Freq Total']), normalize=False)
    if statistic=='npmi':
        df['MI'] = pmi(node_freq, df['Freq Total'], df['Freq Span'], sum(df_total['Freq Total']), normalize=True)
    if statistic=='pmi2':
        df['MI'] = pmi2(node_freq, df['Freq Total'], df['Freq Span'], sum(df_total['Freq Total']))
    if statistic=='pmi3':
        df['MI'] = pmi3(node_freq, df['Freq Total'], df['Freq Span'], sum(df_total['Freq Total']))
    df.sort_values(by=['MI', 'Token'], ascending=[False, True], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return(df)

def ngrams_by_token(tok, node_word: str, node_position, span, n_tokens, search_type, count_by='pos'):
    span_l = node_position - 1
    span_r = span - span_l
    tok = list(tok.values())
    if count_by == 'pos':
        tc = corpus_utils._merge_tags(tok)
    if count_by == 'ds':
        tc = corpus_utils._merge_ds(tok)
    ngram_list = []
    for i in range(0,len(tc)):
        tp = tc[i]
        tpf = [t[0] for t in tp]
       # create a boolean vector for node word
        if search_type == "fixed":
            v = [t == node_word.lower() for t in tpf]
        elif search_type == "starts_with":
            v = [t.startswith(node_word.lower()) for t in tpf]
        elif search_type == "ends_with":
            v = [t.endswith(node_word.lower()) for t in tpf]
        elif search_type == "contains":
            v = [node_word.lower() in t.lower() for t in tpf]
        if sum(v) > 0:
            idx = list(index_windows_around_matches(np.array(v), left=span_l, right=span_r, flatten=False))
            start_idx = [min(x) for x in idx]
            end_idx = [max(x) for x in idx]
            in_span = []
            for i in range(len(idx)):
                span_tokens = [t for t in tp[start_idx[i]:end_idx[i]]]
                in_span.append(span_tokens)
                merge_with_tags = []
                for i in range(0,len(in_span)):
                    if len(in_span[i]) == span:
                        merge_with_tags.append(list('_tag_'.join(x) for x in in_span[i]))
                    merged_tokens = ['_token_'.join(x) for x in merge_with_tags]
            ngram_list.append(merged_tokens)
    # calculate ranges
    ngram_range = []
    for i in range(0,len(ngram_list)):
        ngram_range.append(list(set(ngram_list[i])))
    ngram_range = [x for xs in ngram_range for x in xs]
    ngram_range = Counter(ngram_range)
    ngram_range = sorted(ngram_range.items(), key=lambda pair: pair[0], reverse=False)
    # calculate counts
    ngram_count = [x for xs in ngram_list for x in xs]
    ngram_count = Counter(ngram_count)
    ngram_count = sorted(ngram_count.items(), key=lambda pair: pair[0], reverse=False)
    # build table
    ngrams = [x[0] for x in ngram_count]
    ngrams = [x.split('_token_') for x in ngrams]
    ngrams = [sum([x[i].split('_tag_') for i in range(span)], []) for x in ngrams]
    order = list(range(0, span*2, 2)) + list(range(1, span*2 + 1, 2))
    for l in reversed(range(len(ngrams))):
        ngrams[l] = [ngrams[l][j] for j in order]
    ngrams = np.array(ngrams)
    ngram_freq = np.array([x[1] for x in ngram_count])
    ngram_prop = np.array(ngram_freq)/n_tokens*1000000
    ngram_range = np.array([x[1] for x in ngram_range])/len(tok)*100
    counts = list(zip(ngrams.tolist(), ngram_freq.tolist(), ngram_prop.tolist(), ngram_range.tolist()))
    ngram_counts = list()
    for x in counts:
        tt = tuple()
        for y in x:
            if not type(y) == list:
                tt += (y,)
            else:
                tt += (*y,)
        ngram_counts.append(tt)
    df = pd.DataFrame(ngram_counts, columns=['Token' + str(i) for i in range (1, span+1)] + ['Tag' + str(i) for i in range (1, span+1)] + ['AF', 'RF', 'Range'])
    df.sort_values(by=['AF', 'Token1'], ascending=[False, True], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return(df)

def ngrams_by_tag(tok, tag: str, tag_position, span, n_tokens, count_by='pos'):
    span_l = tag_position - 1
    span_r = span - span_l
    tok = list(tok.values())
    if count_by == 'pos':
        tc = corpus_utils._merge_tags(tok)
    if count_by == 'ds':
        tc = corpus_utils._merge_ds(tok)
    ngram_list = []
    for i in range(0,len(tc)):
        tp = tc[i]
        tpf = [t[1] for t in tp]
       # create a boolean vector for tag
        v = [t == tag for t in tpf]
        if sum(v) > 0:
            idx = list(index_windows_around_matches(np.array(v), left=span_l, right=span_r, flatten=False))
            start_idx = [min(x) for x in idx]
            end_idx = [max(x) for x in idx]
            in_span = []
            for i in range(len(idx)):
                span_tokens = [t for t in tp[start_idx[i]:end_idx[i]]]
                in_span.append(span_tokens)
                merge_with_tags = []
                for i in range(0,len(in_span)):
                    if len(in_span[i]) == span:
                        merge_with_tags.append(list('_tag_'.join(x) for x in in_span[i]))
                    merged_tokens = ['_token_'.join(x) for x in merge_with_tags]
            ngram_list.append(merged_tokens)
    # calculate ranges
    ngram_range = []
    for i in range(0,len(ngram_list)):
        ngram_range.append(list(set(ngram_list[i])))
    ngram_range = [x for xs in ngram_range for x in xs]
    ngram_range = Counter(ngram_range)
    ngram_range = sorted(ngram_range.items(), key=lambda pair: pair[0], reverse=False)
    # calculate counts
    ngram_count = [x for xs in ngram_list for x in xs]
    ngram_count = Counter(ngram_count)
    ngram_count = sorted(ngram_count.items(), key=lambda pair: pair[0], reverse=False)
    # build table
    ngrams = [x[0] for x in ngram_count]
    ngrams = [x.split('_token_') for x in ngrams]
    ngrams = [sum([x[i].split('_tag_') for i in range(span)], []) for x in ngrams]
    order = list(range(0, span*2, 2)) + list(range(1, span*2 + 1, 2))
    for l in reversed(range(len(ngrams))):
        ngrams[l] = [ngrams[l][j] for j in order]
    ngrams = np.array(ngrams)
    ngram_freq = np.array([x[1] for x in ngram_count])
    ngram_prop = np.array(ngram_freq)/n_tokens*1000000
    ngram_range = np.array([x[1] for x in ngram_range])/len(tok)*100
    counts = list(zip(ngrams.tolist(), ngram_freq.tolist(), ngram_prop.tolist(), ngram_range.tolist()))
    ngram_counts = list()
    for x in counts:
        tt = tuple()
        for y in x:
            if not type(y) == list:
                tt += (y,)
            else:
                tt += (*y,)
        ngram_counts.append(tt)
    df = pd.DataFrame(ngram_counts, columns=['Token' + str(i) for i in range (1, span+1)] + ['Tag' + str(i) for i in range (1, span+1)] + ['AF', 'RF', 'Range'])
    df.sort_values(by=['AF', 'Token1'], ascending=[False, True], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return(df)

def split_corpus(tok, tar_list, ref_list):
	tar_docs = {key: value for key, value in tok.items() if key.startswith(tuple(tar_list))}
	ref_docs = {key: value for key, value in tok.items() if key.startswith(tuple(ref_list))}
	tar_ndocs = len(tar_docs)
	ref_ndocs = len(ref_docs)
	#get target counts
	tar_tok = list(tar_docs.values())
	tar_tags = []
	for i in range(0,len(tar_tok)):
		tags = [x[1] for x in tar_tok[i]]
		tar_tags.append(tags)
	tar_tags = [x for xs in tar_tags for x in xs]
	tar_tokens = len(tar_tags)
	tar_words = len([x for x in tar_tags if not x.startswith('Y')])
	
	#get reference counts
	ref_tok = list(ref_docs.values())
	ref_tags = []
	for i in range(0,len(ref_tok)):
		tags = [x[1] for x in ref_tok[i]]
		ref_tags.append(tags)
	ref_tags = [x for xs in ref_tags for x in xs]
	ref_tokens = len(ref_tags)
	ref_words = len([x for x in ref_tags if not x.startswith('Y')])
	return tar_docs, ref_docs, tar_words, ref_words, tar_tokens, ref_tokens, tar_ndocs, ref_ndocs

def kwic_st(tok, node_word, search_type, ignore_case=True):
	kwic = []
	for i in range(0,len(tok)):
		tpf = list(tok.values())[i]
		doc_id = list(tok.keys())[i]
		# create a boolean vector for node word
		if bool(ignore_case) == True and search_type == "fixed":
			v = [t[0].strip().lower() == node_word.lower() for t in tpf]
		elif bool(ignore_case) == False and search_type == "fixed":
			v = [t[0].strip() == node_word for t in tpf]
		elif bool(ignore_case) == True and search_type == "starts_with":
			v = [t[0].strip().lower().startswith(node_word.lower()) for t in tpf]
		elif bool(ignore_case) == False and search_type == "starts_with":
			v = [t[0].strip().startswith(node_word) for t in tpf]
		elif bool(ignore_case) == True and search_type == "ends_with":
			v = [t[0].strip().lower().endswith(node_word.lower()) for t in tpf]
		elif bool(ignore_case) == False and search_type == "ends_with":
			v = [t[0].strip().endswith(node_word) for t in tpf]
		elif bool(ignore_case) == True and search_type == "contains":
			v = [node_word.lower() in t[0].strip().lower() for t in tpf]
		elif bool(ignore_case) == False and search_type == "contains":
			v = [node_word in t[0].strip() for t in tpf]

		if sum(v) > 0:
			# get indices within window around the node
			idx = list(index_windows_around_matches(np.array(v), left=7, right=7, flatten=False))
			node_idx = [i for i, x in enumerate(v) if x == True]
			start_idx = [min(x) for x in idx]
			end_idx = [max(x) for x in idx]
			in_span = []
			for i in range(len(node_idx)):
				pre_node = "".join([t[0] for t in tpf[start_idx[i]:node_idx[i]]]).strip()
				post_node = "".join([t[0] for t in tpf[node_idx[i]+1:end_idx[i]]]).strip()
				node = tpf[node_idx[i]][0]
				in_span.append((doc_id, pre_node, node, post_node))
			kwic.append(in_span)
	kwic = [x for xs in kwic for x in xs]
	if len(kwic) > 0:
		df = pd.DataFrame(kwic)
		df.columns =['Doc ID', 'Pre-Node', 'Node', 'Post-Node']
	else:
		df = ''
	return(df)

# https://github.com/WZBSocialScienceCenter/tmtoolkit/blob/master/tmtoolkit/bow/bow_stats.py
def doc_frequencies(dtm, min_val=1, proportions=0):
    if dtm.ndim != 2:
        raise ValueError('`dtm` must be a 2D array/matrix')

    doc_freq = np.sum(dtm >= min_val, axis=0)

    if doc_freq.ndim != 1:
        doc_freq = doc_freq.A.flatten()

    if proportions == 1:
        return doc_freq / dtm.shape[0]
    elif proportions == 2:
        return np.log(doc_freq) - np.log(dtm.shape[0])
    else:
        return doc_freq

def doc_lengths(dtm):
    if dtm.ndim != 2:
        raise ValueError('`dtm` must be a 2D array/matrix')

    res = np.sum(dtm, axis=1)
    if res.ndim != 1:
        return res.A.flatten()
    else:
        return res

def tf_proportions(dtm, norm=False, scale=False):

    norm_factor = 1 / doc_lengths(dtm)[:, None]   # shape: Nx1

    res = dtm * norm_factor
    
    if norm == True:
        res *=100
    else:
        res
    if scale == True:
        scaled_res = res.select_dtypes(include='number').apply(scipy.stats.zscore)
        res = pd.DataFrame(scaled_res, index=res.index, columns=res.columns)
    else:
        res
    if isinstance(res, np.matrix):
        return res.A
    else:
        return res

def idf(dtm, smooth_log=1, smooth_df=1):
    if dtm.ndim != 2 or 0 in dtm.shape:
        raise ValueError('`dtm` must be a non-empty 2D array/matrix')

    n_docs = dtm.shape[0]
    df = doc_frequencies(dtm)

    if smooth_log == smooth_df == 1:      # log1p is faster than the equivalent log(1 + x)
        # log(1 + N/(1+df)) = log((1+df+N)/(1+df)) = log(1+df+N) - log(1+df) = log1p(df+N) - log1p(df)
        return np.log1p(df + n_docs) - np.log1p(df)
    else:
        # with s = smooth_log and t = smooth_df
        # log(s + N/(t+df)) = log((s(t+df)+N)/(t+df)) = log(s(t+df)+N) - log(t+df)
        return np.log(smooth_log * (smooth_df + df) + n_docs) - np.log(smooth_df + df)

def tfidf(dtm, tf_func=tf_proportions, idf_func=idf, **kwargs):
    if dtm.ndim != 2 or 0 in dtm.shape:
        raise ValueError('`dtm` must be a non-empty 2D array/matrix')

    if idf_func is idf:
        idf_opts = {}
        if 'smooth_log' in kwargs:
            idf_opts['smooth_log'] = kwargs.pop('smooth_log')
        if 'smooth_df' in kwargs:
            idf_opts['smooth_df'] = kwargs.pop('smooth_df')

        idf_vec = idf_func(dtm, **idf_opts)
    elif idf_func is idf_probabilistic and 'smooth' in kwargs:
        idf_vec = idf_func(dtm, smooth=kwargs.pop('smooth'))
    else:
        idf_vec = idf_func(dtm)

    tf_mat = tf_func(dtm, **kwargs)

    return tf_mat * idf_vec

def simplify_dtm(dtm, sums):
	dtm_simple = dtm.copy()
	dtm_simple.index.name = 'doc_id'
	dtm_simple.reset_index(inplace=True)
	#need index to maintain order
	dtm_simple['doc_order'] = dtm_simple.index
	dtm_simple = pd.melt(dtm_simple,id_vars=['doc_id', 'doc_order'],var_name='Tag', value_name='RF')
	dtm_simple['Tag'].replace('^NN\S*$', '#Noun', regex=True, inplace=True)
	dtm_simple['Tag'].replace('^VV\S*$', '#Verb', regex=True, inplace=True)
	dtm_simple['Tag'].replace('^J\S*$', '#Adjective', regex=True, inplace=True)
	dtm_simple['Tag'].replace('^R\S*$', '#Adverb', regex=True, inplace=True)
	dtm_simple['Tag'].replace('^P\S*$', '#Pronoun', regex=True, inplace=True)
	dtm_simple['Tag'].replace('^I\S*$', '#Preposition', regex=True, inplace=True)
	dtm_simple['Tag'].replace('^C\S*$', '#Conjunction', regex=True, inplace=True)
	dtm_simple = dtm_simple.loc[dtm_simple["Tag"].str.startswith('#', na=False)]
	dtm_simple['Tag'].replace('^#', '', regex=True, inplace=True)
	#sum tags
	dtm_simple = dtm_simple.groupby(['doc_id', 'doc_order', 'Tag'], as_index=False)['RF'].sum()
	dtm_simple.sort_values(by='doc_order', inplace=True, ignore_index=True)
	dtm_simple = dtm_simple.pivot_table(index=['doc_order', 'doc_id'], columns='Tag', values='RF')
	dtm_simple.reset_index(inplace=True)
	dtm_simple.drop('doc_order', axis=1, inplace=True)
	dtm_simple.set_index('doc_id', inplace=True)
	dtm_simple = dtm_simple.divide(sums, axis=0)
	dtm_simple *= 100
	return(dtm_simple)

def pca_contributions(df, doccats):
	n = min(len(df.index), len(df.columns))
	pca = sklearn.decomposition.PCA(n_components=n)
	pca_result = pca.fit_transform(df.values)
	pca_df = pd.DataFrame(pca_result)
	pca_df.columns = ['PC' + str(col + 1) for col in pca_df.columns]
						
	sdev = pca_df.std(ddof=0)
	contrib = []
	for i in range(0, len(sdev)):
		coord = pca.components_[i] * sdev[i]
		polarity = np.divide(coord, abs(coord))
		coord = np.square(coord)
		coord = np.divide(coord, sum(coord))*100
		coord = np.multiply(coord, polarity)				
		contrib.append(coord)
	contrib_df =  pd.DataFrame(contrib).transpose()
	contrib_df.columns = ['PC' + str(col + 1) for col in contrib_df.columns]
	contrib_df['Tag'] = df.columns
	if len(doccats) > 0:
		pca_df['Group'] = doccats
	pca_df['doc_id'] = list(df.index)		
	ve = np.array(pca.explained_variance_ratio_).tolist()
	return pca_df, contrib_df, ve

def update_pca_plot(pca: dict):
	coord_data = pca['pca']
	contrib_data = pca['contrib']
	variance = pca['variance']
	pca_idx = pca['pca_idx']
	pca_x = coord_data.columns[pca_idx - 1]
	pca_y = coord_data.columns[pca_idx]
	
	mean_x = contrib_data[pca_x].abs().mean()
	mean_y = contrib_data[pca_y].abs().mean()
	contrib_x = contrib_data[contrib_data[pca_x].abs() > mean_x]
	contrib_x[pca_x] = contrib_x[pca_x]
	contrib_x.sort_values(by=pca_x, ascending=False, inplace=True)
	contrib_x_values = contrib_x.loc[:,pca_x].tolist()
	contrib_x_values = ['%.2f' % x for x in contrib_x_values]
	contrib_x_values = [x + "%" for x in contrib_x_values]
	contrib_x_tags = contrib_x.loc[:,"Tag"].tolist()
	contrib_x = list(zip(contrib_x_tags, contrib_x_values))
	contrib_x = list(map(', '.join, contrib_x))
	contrib_x = '; '.join(contrib_x)
	
	contrib_y = contrib_data[contrib_data[pca_y].abs() > mean_y]
	contrib_y[pca_y] = contrib_y[pca_y]
	contrib_y.sort_values(by=pca_y, ascending=False, inplace=True)
	contrib_y_values = contrib_y.loc[:,pca_y].tolist()
	contrib_y_values = ['%.2f' % y for y in contrib_y_values]
	contrib_y_values = [y + "%" for y in contrib_y_values]
	contrib_y_tags = contrib_y.loc[:,"Tag"].tolist()
	contrib_y = list(zip(contrib_y_tags, contrib_y_values))
	contrib_y = list(map(', '.join, contrib_y))
	contrib_y = '; '.join(contrib_y)
	
	contrib_1 = contrib_data[contrib_data[pca_x].abs() > 0]
	contrib_1[pca_x] = contrib_1[pca_x].div(100)
	contrib_1.sort_values(by=pca_x, ascending=True, inplace=True)
	
	contrib_2 = contrib_data[contrib_data[pca_y].abs() > 0]
	contrib_2[pca_y] = contrib_2[pca_y].div(100)	
	contrib_2.sort_values(by=pca_y, ascending=True, inplace=True)
	
	ve_1 = "{:.2%}".format(variance[pca_idx - 1])
	ve_2 = "{:.2%}".format(variance[pca_idx])

	cp_1 = alt.Chart(contrib_1, height={"step": 24}).mark_bar(size=12).encode(
				x=alt.X(pca_x, axis=alt.Axis(format='%')), 
				y=alt.Y('Tag', sort='-x', title=None, axis=alt.Axis(labelLimit=200)),
	 			tooltip=[
				alt.Tooltip('Tag'),
     	  		 alt.Tooltip(pca_x, title="Cont.", format='.2%')
    			])
	
	cp_2 = alt.Chart(contrib_2, height={"step": 24}).mark_bar(size=12).encode(
				x=alt.X(pca_y, 
				axis=alt.Axis(format='%')), 
				y=alt.Y('Tag', sort='-x', title=None, axis=alt.Axis(labelLimit=200)),
				tooltip=[
				alt.Tooltip('Tag'),
				alt.Tooltip(pca_y, title="Cont.", format='.2%')
				])
		
	return(cp_1, cp_2, pca_x, pca_y, contrib_x, contrib_y, ve_1, ve_2)
	
def correlation(df, x, y):
	cc = scipy.stats.stats.pearsonr(df[x], df[y])
	cc_r = round(cc.statistic, 3)
	cc_p = round(cc.pvalue, 5)
	cc_df = len(df.index) - 2
	return cc_df, cc_r, cc_p

def boxplots(df, box_vals, tag_set, tag_type = None, grp_a = None, grp_b = None):

	df_plot = df[box_vals]
	
	if grp_a is None and grp_b is None:
		df_plot.index.name = 'doc_id'
		df_plot.reset_index(inplace=True)
	
		stats = df_plot.describe().T
		med = df_plot.median().rename("median")
		stats = stats.join(med)

	if grp_a is not None and grp_b is not None:
		df_plot.loc[df_plot.index.str.startswith(tuple(grp_a)), 'Group'] = 'Group A'
		df_plot.loc[df_plot.index.str.startswith(tuple(grp_b)), 'Group'] = 'Group B'
		df_plot = df_plot.dropna()							
		df_plot.index.name = 'doc_id'
		df_plot.reset_index(inplace=True)
	
		if tag_set == 'Parts-of-Speech' and tag_type == 'General':
			stats = df_plot.groupby('Group').describe().unstack(1).reset_index().pivot(index=['Tag', 'Group'], values=0, columns='level_1')
			med = df_plot.groupby('Group').median().unstack(1).reset_index()
			med = med.rename(columns={med.columns[2]: 'median'})
			med = med.sort_values(['Tag', 'Group'])
		
		else:
			stats = df_plot.groupby('Group').describe().unstack(1).reset_index().pivot(index=['level_0', 'Group'], values=0, columns='level_1')
			med = df_plot.groupby('Group').median().unstack(1).reset_index()
			med = med.rename(columns={med.columns[2]: 'median'})
			med = med.sort_values(['level_0', 'Group'])
	
		stats['median'] = med['median'].values
	
	stats = stats[['count', 'mean', 'median', 'std', 'min', '25%', '50%', '75%', 'max']]
	stats = stats.to_string(header=True, index_names=False)
	#preserve formatting for markdown
	stats = 'Tag' + stats
	stats = stats.replace('Tag   ', 'Tag')
	stats = stats.replace('\n', '\n    ')
		
	if grp_a is not None and grp_b is not None:
		df_plot = pd.melt(df_plot,id_vars=['doc_id', 'Group'],var_name='Tag', value_name='RF')
		df_plot['Median'] = df_plot.groupby(['Tag', 'Group']).transform('median')
		df_plot.sort_values(by=['Group', 'Median'], ascending=[False, True], inplace=True, ignore_index=True)
		return df_plot, stats

	else:
		df_plot = pd.melt(df_plot,id_vars=['doc_id'],var_name='Tag', value_name='RF')
		df_plot['Median'] = df_plot.groupby(['Tag']).transform('median')
		df_plot.sort_values(by='Median', inplace=True, ignore_index=True, ascending=False)
		cols = df_plot['Tag'].drop_duplicates().tolist()
		return df_plot, stats, cols

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

def simplify_counts(pos_counts, n_tokens):
        df = pos_counts.drop('RF', axis=1)
        df['Tag'].replace('^NN\S*$', '#Noun', regex=True, inplace=True)
        df['Tag'].replace('^VV\S*$', '#Verb', regex=True, inplace=True)
        df['Tag'].replace('^J\S*$', '#Adjective', regex=True, inplace=True)
        df['Tag'].replace('^R\S*$', '#Adverb', regex=True, inplace=True)
        df['Tag'].replace('^P\S*$', '#Pronoun', regex=True, inplace=True)
        df['Tag'].replace('^I\S*$', '#Preposition', regex=True, inplace=True)
        df['Tag'].replace('^C\S*$', '#Conjunction', regex=True, inplace=True)
        df = df.loc[df["Tag"].str.startswith('#', na=False)]
        df['Tag'].replace('^#', '', regex=True, inplace=True)
        df = df.groupby(['Tag'], as_index=False)['AF'].sum()
        df.sort_values(by='AF', inplace=True, ignore_index=True, ascending=False)
        df.reset_index(inplace=True)
        df['RF'] = df.AF/n_tokens*100
        return(df)
        
def simplify_span(pos_span):
        df = pos_span.copy()
        df['Tag'].replace('^NN\S*$', 'Noun', regex=True, inplace=True)
        df['Tag'].replace('^VV\S*$', 'Verb', regex=True, inplace=True)
        df['Tag'].replace('^J\S*$', 'Adjective', regex=True, inplace=True)
        df['Tag'].replace('^R\S*$', 'Adverb', regex=True, inplace=True)
        df['Tag'].replace('^P\S*$', 'Pronoun', regex=True, inplace=True)
        df['Tag'].replace('^I\S*$', 'Preposition', regex=True, inplace=True)
        df['Tag'].replace('^C\S*$', 'Conjunction', regex=True, inplace=True)
        return(df)

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

def html_simplify(html_str):
    html_str = re.sub(r'span class="NN\S*">', 'span class="Noun">', html_str)
    html_str = re.sub(r'span class="VV\S*">', 'span class="Verb">', html_str)
    html_str = re.sub(r'span class="J\S*">', 'span class="Adjective">', html_str)
    html_str = re.sub(r'span class="R\S*">', 'span class="Adverb">', html_str)
    html_str = re.sub(r'span class="P\S*">', 'span class="Pronoun">', html_str)
    html_str = re.sub(r'span class="I\S*">', 'span class="Preposition">', html_str)
    html_str = re.sub(r'span class="C\S*">', 'span class="Conjunction">', html_str)
    return(html_str)

def add_alt_chunk(doc: docx.Document, html: str):
    package = doc.part.package
    partname = package.next_partname('/word/altChunk%d.html')
    alt_part = docx.opc.part.Part(partname, 'text/html', html.encode(), package)
    r_id = doc.part.relate_to(alt_part, docx.opc.constants.RELATIONSHIP_TYPE.A_F_CHUNK)
    alt_chunk = docx.oxml.OxmlElement('w:altChunk')
    alt_chunk.set(docx.oxml.ns.qn('r:id'), r_id)
    doc.element.body.sectPr.addprevious(alt_chunk)
    