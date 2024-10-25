# Copyright (C) 2024 David West Brown

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Functions for generating content.
# Some populate descriptive content from corpora or sub-corpora.
# Others populate the results of statistical functions.


def message_version_info(ds_version, model_name, model_version):
	variance_info = f"""<p></p><span style="color:gray">
	DocuScope CAC version: {ds_version}; spaCy model <a href="https://huggingface.co/browndw/en_docusco_spacy/" target="_blank">{model_name}</a> version: {model_version}
	</span>
	"""
	return(variance_info)

def message_target_info(target_metadata):
	tokens_pos = target_metadata.get('tokens_pos')[0]
	tokens_ds = target_metadata.get('tokens_ds')[0]
	ndocs = target_metadata.get('ndocs')[0]
	target_info = f"""##### Target corpus information:
	
	Number of part-of-speech tokens in corpus: {tokens_pos}\n    Number of DocuScope tokens in corpus: {tokens_ds}\n    Number of documents in corpus: {ndocs}
	"""
	return(target_info)

def message_reference_info(reference_metadata):
	tokens_pos = reference_metadata.get('tokens_pos')[0]
	tokens_ds = reference_metadata.get('tokens_ds')[0]
	ndocs = reference_metadata.get('ndocs')[0]
	reference_info = f"""##### Reference corpus information:
	
	Number of part-of-speech tokens in corpus: {tokens_pos}\n    Number of DocuScope tokens in corpus: {tokens_ds}\n    Number of documents in corpus: {ndocs}
	"""
	return(reference_info)

def message_target_parts(keyness_parts):
	t_cats = keyness_parts[0]
	tokens_pos = keyness_parts[2]
	tokens_ds = keyness_parts[4]
	ndocs = keyness_parts[6]
	target_info = f"""##### Target corpus information:
	
	Document categories: {t_cats}\n    Part-of-speech tokens: {tokens_pos}\n    DocuScope tokens: {tokens_ds}\n    Documents: {ndocs}
	"""
	return(target_info)

def message_reference_parts(keyness_parts):
	r_cats = keyness_parts[1]
	tokens_pos = keyness_parts[3]
	tokens_ds = keyness_parts[5]
	ndocs = keyness_parts[7]
	reference_info = f"""##### Reference corpus information:
	
	Document categories: {r_cats}\n    Part-of-speech tokens: {tokens_pos}\n    DocuScope tokens: {tokens_ds}\n    Documents: {ndocs}
	"""
	return(reference_info)

def message_collocation_info(collocation_data):
	mi = str(collocation_data[1]).upper()
	span = collocation_data[2] + 'L - ' + collocation_data[3] + 'R'
	coll_info = f"""##### Collocate information:
	
	Association measure: {mi}\n    Span: {span}\n    Node word: {collocation_data[0]}
	"""
	return(coll_info)

def message_variance_info(pca_x, pca_y, ve_1, ve_2):
	variance_info = f"""##### Variance explained:
	
	{pca_x}: {ve_1}\n    {pca_y}: {ve_2}
	"""
	return(variance_info)

def message_contribution_info(pca_x, pca_y, contrib_x, contrib_y):
	contrib_info = f"""##### Variables with contribution > mean:
	
	{pca_x}: {contrib_x}\n    {pca_y}: {contrib_y}
	"""
	return(contrib_info)

def message_correlation_info(cc_df, cc_r, cc_p):
	corr_info = f"""##### Pearson's correlation coefficient:
	
	r({cc_df}) = {cc_r}, p-value = {cc_p}
	"""
	return(corr_info)

def message_stats_info(stats):
	stats_info = f"""##### Descriptive statistics:
	
	{stats}
	"""
	return(stats_info)

def message_group_info(grp_a, grp_b):
	grp_a = [s.strip('_') for s in grp_a]
	grp_a = ", ".join(str(x) for x in grp_a)
	grp_b = [s.strip('_') for s in grp_b]
	grp_b = ", ".join(str(x) for x in grp_b)
	group_info = f"""##### Grouping variables:
	
	Group A: {grp_a}\n    Group B: {grp_b}
	"""
	return(group_info)

# Static messages that populates the main containers of the apps.

message_load = """
	* From this page you can **load a saved corpus** or **process a new one** by selecting the desired (**.txt**) files. You can also reset your target corpus or manage any corpora you have saved.
	* Once you have loaded a target corpus, you can add a **reference corpus** for comparison. Also note that you can encode metadata into your filenames, which can used for further analysis. (See the **About new corpora** expander.)
	"""

message_load_target_internal = """
    :point_left: Select a saved corpus from the lists in the sidebar.\n
    :exclamation:  Note that corpora are organized by model with which they were tagged.
    """

message_load_target_external = """
    :point_down: Use the widget to select the corpus you'd like to load, either by browsing for them or dragging-and-dropping.\n
    :trackball: Once you've selected your file, click the **UPLOAD** button and a processing button will appear in the sidebar.
    """

message_load_target_new = """
    :point_down: Use the widget to select the files you'd like process, either by browsing for them or dragging-and-dropping.\n
    :trackball: Once you've selected your files, click the **UPLOAD** button and a processing button will appear in the sidebar.\n
    :file_folder: After processing, you will have the option to save your corpus to use for future analysis. This is not necessary, but may be useful if you plan to revisit the data.\n
    :exclamation: Don't forget to select your model from the sidebar if you are processing a new corpus.\n
    :exclamation: Be sure that all file names are unique.\n
    :alarm_clock: Processing times may vary, but you can expect the initial corpus processing to take roughly 1 minute for every 1 million words.
    """

message_load_reference = """
    * Use the widget to select the files you'd like process, either by browsing for them or dragging-and-dropping.
    * Once you've selected your files, click the **UPLOAD** button and a processing button will appear in the sidebar.\n
    :exclamation: Your reference will be tagged with the same model as your target corpus.\n
    :exclamation: Be sure that all file names are unique and that they don't share names with your target corpus.\n
    :alarm_clock: Processing times may vary, but you can expect the initial corpus processing to take roughly 1 minute for every 1 million words.
    """

message_select_reference = """
    :point_left: Select a saved corpus from the lists in the sidebar.\n
    :exclamation: Only corpora tagged with the same model as your target corpus will be available as a reference.
    """

message_tables = """
    :point_left: Use the button to generate a table.\n
    * After the table has been generated, you will be able to toggle between the tagsets.
    """

message_ngrams = """
	:point_down: N-grams/Clusters can be created using different options:\n
	* You can generate a table of the most common 1-, 2-, 3-, and 4-grams\n
	* You can input a word or string, specify whether that input should match a token completely or partially, and choose which tagset to return.
	* Alternatively, you can select a tag (like **NN1** or **AcademicTerms**) as the basis for your clusters.
	* For clusters, you must select their span and the slot where your chosen word or tag should appear (on the left, in the middle, or on the right).
	"""

message_collocations = """
	:point_left: Collocations can be created using different options:\n
	* You can input a word (without any spaces) and return collocates and their part-of-speech tags.
	* You can also adjust the span (to left or right) of your node word.
	* You can choose to **anchor** your node word by a tag (e.g. specifying *can* as a **modal verb** or as **hedged confidence**).
	* You can choose from among 4 different association measures.
	"""

message_kwic = """
	:point_left: Use this tool to generate Key Words in Context for a word or part of a word (like the ending *tion*).\n
	* Note that wildcard characters are **not needed**.
	* Instead specify if you want a word to start with, end with, or include a string.
	"""

message_keyness = """
	:point_left: Use the button to generate a table.\n
	* To use this tool, be sure that you have loaded a reference corpus.
	* Loading a reference can be done from **Manage Corpora**.
	"""

message_plotting = """
	:heavy_exclamation_mark: To use this page, you must first process a target corpus.\n
	:heavy_plus_sign: You can also increase your plotting options by processing target corpus metadata.
	"""

message_corpus_parts = """
	:point_left: Use the options to generate a table from subcorpora.\n
	* To use this tool, you must first process **metadata** from your file names.
	* Categories of interest can be placed at the beginning of file names before an underscore:
	```
	BIO_G0_02_1.txt, BIO_G0_03_1.txt, ENG_G0_16_1.txt, ENG_G0_21_1.txt, HIS_G0_02_1.txt, HIS_G0_03_1.txt
	```
	* Processing these names would yield the categories:
	```
	BIO, ENG, HIS
	```
	* Those categories could then be compared in any combination.\n
	:lock: Selecting of the same category as target and reference is prevented.
	"""

message_download_tagged = """
	Once a corpus has been processed, you can use this page to generate a **zipped folder of tagged text files**. 
	The tags are embbedd into the text after a vertical bar:
	````
	At|II root|NN1 , every|AT1 hypothesis|NN1 is|VBZ a|AT1 claim|NN1 about|II the|AT relevance|NN1
	````
	Because the tags identify mutliword units, spaces that occur within a token are replaced with underscores:
	````
	evidence|Reasoning and|SyntacticComplexity theory|AcademicTerms pertaining_to_the|Reasoning possibility_of|ConfidenceHedged sympatric|Description speciation|Description
	````
	If you are planning to use the output to process the files in a tool like AntConc or in a coding environment, take note of these conventions and account for them accordingly.
	"""

message_single_document = """
	:point_left: To use this page, first select a document. Then you can:\n
	* Choose up to 5 tags to highlight in the document.
	* Plot the location of those tags in the document.
	* Download highlighted text as a Word file.
	"""

# Static messages that populate the sidebars.

message_download = """
	### Download
    \nClick the button to download your data.
    """

message_download_dtm = """
	### Download
    \nClick the button to genenerate a download link.
    """

message_generate_table = """
	### Generate table
    \nUse the button to generate a table.
    For a table to be created, you must first load a target corpus.
    """

message_generate_plot = """
	### Generate Plot
    \nClick the button to genenerate a plot.
    You can use the checkboxes to plot selected rows.
    With no rows selected, all variables will be plotted.
    """

message_reset_table = """
	### Reset table
    \nUse the button to reset and create a new table.
    """

# Static messages that populate the expanders.

message_internal_corpora = """
	DocuScope CAC comes with some pre-processed corpus data to get you started.
	There is a sub-sample of the [Michigan Corpus of Upper-Level Student Papers (MICUSP)](https://elicorpora.info/main).
	The sub-sample contains 10 papers from 17 disciplines.
	This is called **MICUSP-mini** and is recommended for exploring, if you are new to the tool.\n\n
	There is also a parsed version of the full MICUSP corpus, as well as a corpus of published academic papers.
	The latter is named **ELSEVIER** and contains data from open access publications from 20 disciplines.
	You can see the metadata (as well as the full subject area names) on the [GitHub repository](https://github.com/browndw/corpus-tagger#elesevier-corpus).
	Information about mini version of the Human-AI Parallel Corpus (HAP-E) can be found on [huggingface](https://huggingface.co/datasets/browndw/human-ai-parallel-corpus-mini).\n\n
	If you are using the MICUSP data for academic work or for publication, [please cite it](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=U8wDvfIAAAAJ&citation_for_view=U8wDvfIAAAAJ:roLk4NBRz8UC).
	Likewise, the citation information for [HAP-E can be found here](https://scholar.google.com/scholar_lookup?arxiv_id=2410.16107).
	"""

message_external_corpora = """
	An external corpus is one that has already been processed by DocuScope CAC and downloaded to your computer.
	Most often, you would do this to save time. Processing is the most computationally intensive part of preparing the data.
	But you also might do this to easily share your data with others.
	Finally you might want to run you own experiments outside of the tool.
	The corpus file is in a specialized format called **parquet**.
	A parquet file can read into coding environments like Python and R using packages like **polars** and **arrow**.
"""

message_naming = """
    Files must be in a \*.txt format. If you are preparing files for the first time,
    it is recommended that you use a plain text editor (rather than an application like Word).
    Avoid using spaces in file names.
    Also, you needn't worry about preserving paragraph breaks, as those will be stripped out during processing.\n
    Metadata can be encoded at the beginning of a file name, before an underscore. For example: acad_01.txt, acad_02.txt, 
    blog_01.txt, blog_02.txt. These would allow you to compare **acad** vs. **blog** as categories.
    You can designate up to 20 categories.
	"""

message_models = """
	For detailed descriptions, see the tags tables available from the Help menu.
	But in short, the full dictionary has more categories and coverage than the common dictionary.
	"""

message_association_measures = """
	The most common statistic for measuring token associations is Pointwise Mutual Information (PMI),
	first developed by [Church and Hanks](https://aclanthology.org/J90-1003/).
	One potentially problematic characteristic of PMI is that it rewards (or generates high scores) for low frequency tokens.
	
	This can be handled by filtering for minimum frequencies and MI scores.
	Alternatively, [other measures have been proposed, which you can select from here.](https://en.wikipedia.org/wiki/Pointwise_mutual_information)
	"""

message_columns_collocations = """
	The **Freq Span** columns refers to the collocate's frequency within the given window,
	while **Freq Total** refers to its overall frequency in the corpus. 
	Note that is possible for a collocate to have a *higher* frequency within a window, than a total frequency.\n
	The **MI** column refers to the association measure selected when the table was generated
	(one of NPMI, PMI2, PMI3, or PMI).
	"""

message_columns_keyness = """
	The **LL** column refers to [log-likelihood](https://ucrel.lancs.ac.uk/llwizard.html),
	a hypothesis test measuring observed vs. expected frequencies.
	Note that a negative value means that the token is more frequent in the reference corpus than the target.\n
	**LR** refers to [Log-Ratio](http://cass.lancs.ac.uk/log-ratio-an-informal-introduction/), which is an [effect size](https://www.scribbr.com/statistics/effect-size/).
	And **PV** refers to the [p-value](https://scottbot.net/friends-dont-let-friends-calculate-p-values-without-fully-understanding-them/).\n
	The **AF** columns refer to the absolute frequencies in the target and reference.
	The **RF** columns refer to the relative frequencies (normalized **per million for tokens** and **per 100 for tags**).
	Note that for part-of-speech tags, tokens are normalized against word tokens,
	while DocuScope tags are normalized against counts of all tokens including punctuation.
	The **Range** column refers to the percentage of documents in which the token appears in your corpus.
	"""

message_filters = """
	Filters can be accessed by clicking on the three lines that appear while hovering over a column header.
	For text columns, you can filter by 'Equals', 'Starts with', 'Ends with', and 'Contains'.\n
	Rows can be selected before or after filtering using the checkboxes.
	(The checkbox in the header will select/deselect all rows.)\n
	If rows are selected and appear in new table below the main one,
	those selected rows will be available for download in an Excel file.
	If no rows are selected, the full table will be processed for downloading after clicking the Download button.
	"""

message_columns_tags = """
	The **AF** column refers to the absolute token frequency.
	The **RF** column refers to the relative token frequency (normalized **per 100 tokens**).
	Note that for part-of-speech tags, tokens are normalized against word tokens,
	while DocuScope tags are normalized against counts of all tokens including punctuation.
	The **Range** column refers to the percentage of documents in which the token appears in your corpus.
	"""

message_columns_tokens = """
	The **AF** column refers to the absolute token frequency.
	The **RF** column refers to the relative token frequency (normalized **per million tokens**).
	Note that for part-of-speech tags, tokens are normalized against word tokens,
	while DocuScope tags are normalized against counts of all tokens including punctuation.
	The **Range** column refers to the percentage of documents in which the token appears in your corpus.
	"""

message_anchor_tags = """
	You can choose to **anchor** at token to a specific tag.
	For example, if you wanted to disambiguate *can* as a noun (e.g., *can of soda*)
	from *can* as a modal verb, you could **anchor** the node word to a part-of-speech
	tag (like **Noun**, **Verb** or more specifically **VM**).
	
	For most cases, choosing an **anchor** tag isn't necessary.
	"""
message_span = """
	Associations are calculated by counting the observed frequency within a
	span around a node word and comparing that to the frequency that we would expect
	given its overall frequency in a corpus.
	
	You could adjust the span if, for example, you wanted look at the subjects of a verb. 
	For that, you would want to search only the left of the node word, setting the right span to 0. 
	For verb object, you would want to do the opposite.
	There could be cases when you want a narrower window or a wider one.
	"""

message_general_tags = """
	The conventions for aggregating tags follow those used by the [Corpus of Contemporary American English (COCA)](https://www.english-corpora.org/coca/).\n\n
	Nouns are identified by tags starting with **NN**, which means they are capturing **common nouns** not [proper nouns or pronouns](https://ucrel.lancs.ac.uk/claws7tags.html).\n\n
	Even more importantly verbs are identified by tags starting with **VV**.
	Those are assigned to **lexical verbs**.
	Modal verbs, for example, are identified by **VM**, and are not included in those counts.
	"""

message_variable_contrib = """
	The plots are a Python implementation of [fviz_contrib()](http://www.sthda.com/english/wiki/fviz-contrib-quick-visualization-of-row-column-contributions-r-software-and-data-mining), 
	an **R** function that is part of the **factoextra** package.
	"""


# Style option hack to disable full page view of plots.
# Some don't render correctly in the full-page view.

message_disable_full = """
	<style> button[title="View fullscreen"] {
	display: none;
	} </style>
	"""

