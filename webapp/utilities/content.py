# Copyright (C) 2025 David West Brown

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


def message_reference_info(reference_metadata):
    tokens_pos = reference_metadata.get('tokens_pos')[0]
    tokens_ds = reference_metadata.get('tokens_ds')[0]
    ndocs = reference_metadata.get('ndocs')[0]
    reference_info = f"""##### Reference corpus information:

    Number of part-of-speech tokens in corpus: {tokens_pos:,}
    \n    Number of DocuScope tokens in corpus: {tokens_ds:,}
    \n    Number of documents in corpus: {ndocs:,}
    """
    return reference_info


def message_target_parts(keyness_parts):
    t_cats = keyness_parts[0]
    tokens_pos = keyness_parts[2]
    tokens_ds = keyness_parts[4]
    ndocs = keyness_parts[6]
    target_info = f"""##### Target corpus information:

    Document categories: {t_cats}
    \n    Part-of-speech tokens: {tokens_pos}
    \n    DocuScope tokens: {tokens_ds}
    \n    Documents: {ndocs}
    """
    return target_info


def message_reference_parts(keyness_parts):
    r_cats = keyness_parts[1]
    tokens_pos = keyness_parts[3]
    tokens_ds = keyness_parts[5]
    ndocs = keyness_parts[7]
    reference_info = f"""##### Reference corpus information:

    Document categories: {r_cats}
    \n    Part-of-speech tokens: {tokens_pos}
    \n    DocuScope tokens: {tokens_ds}
    \n    Documents: {ndocs}
    """
    return reference_info


def message_collocation_info(collocation_data):
    mi = str(collocation_data[1]).upper()
    span = collocation_data[2] + 'L - ' + collocation_data[3] + 'R'
    coll_info = f"""##### Collocate information:

    Association measure: {mi}
    \n    Span: {span}
    \n    Node word: {collocation_data[0]}
    """
    return coll_info


def message_variance_info(pca_x: str,
                          pca_y: str,
                          ve_1: str,
                          ve_2: str) -> str:
    variance_info = f"""##### Variance explained:

    {pca_x}: {ve_1}\n    {pca_y}: {ve_2}
    """
    return variance_info


def message_contribution_info(pca_x: str,
                              pca_y: str,
                              contrib_x: str,
                              contrib_y: str) -> str:
    contrib_info = f"""##### Variables with contribution > mean:

    {pca_x}: {contrib_x}\n    {pca_y}: {contrib_y}
    """
    return contrib_info


def message_correlation_info(cc_df: str,
                             cc_r: float,
                             cc_p: float) -> str:
    corr_info = f"""##### Pearson's correlation coefficient:

    r({cc_df}) = {cc_r}, p-value = {cc_p}
    """
    return corr_info


def message_stats_info(stats):
    stats_info = f"""##### Descriptive statistics:

    {stats}
    """
    return stats_info


def message_group_info(grp_a: list[str],
                       grp_b: list[str]) -> str:
    grp_a = [s.strip('_') for s in grp_a]
    grp_a = ", ".join(str(x) for x in grp_a)
    grp_b = [s.strip('_') for s in grp_b]
    grp_b = ", ".join(str(x) for x in grp_b)
    group_info = f"""##### Grouping variables:

    Group A: {grp_a}\n    Group B: {grp_b}
    """
    return group_info


# Static messages that populates the main containers of the apps.


message_collocations = """
    ###### :material/checklist: \
    Collocations can be created using different options:

    * You can input a word (without any spaces) and return collocates
    and their part-of-speech tags.

    * You can also adjust the span (to left or right) of your node word.

    * You can choose to **anchor** your node word by a tag
    (e.g. specifying *can* as a **modal verb** or as **hedged confidence**).

    * You can choose from among 4 different association measures.
    """

message_kwic = """
    :point_left: Use this tool to generate Key Words in Context
    for a word or part of a word (like the ending *tion*).\n

    * Note that wildcard characters are **not needed**.

    * Instead specify if you want a word to start with, end with,
    or include a string.
    """

message_keyness = """
    :point_left: Use the button to generate a table.\n

    * To use this tool, be sure that you have loaded a reference corpus.

    * Loading a reference can be done from **Manage Corpora**.
    """

message_plotting = """
    :material/target: To use this page,
    you must first process a target corpus.\n

    :material/new_label: You can also increase your plotting options
    by processing target corpus metadata. This can be done from
    **Manage Corpus Data**.
    """

message_plotbot = """
    :point_left: Once you process a corpus
    and load the tables,
    the chat interface will become available.
    """  # noqa: E501

message_plotbot_home = """
    Plotbot is a chat assistant designed to **interatively** help you
    create and refine plots from your data.
    I **am not** designed to answer general questions.\n

    With that in mind, you can ask me to create specific plots:
    ```
    Create a bar plot of RF vs. Tag with gray as the bar color.
    ```
    Once I generate a plot, you can ask me to modify its features:
    ```
    Remove "Untagged" from the Tag column
    ```
    I will then revise the plot with the requested changes.\n

    If you want me to create a new plot, use the **Clear chat history** button
    in the sidebar and begin a new conversation.
    """  # noqa: E501

message_plotbot_tips = """
    * Use plot-related commands like "Make plot horizontal" or "Make bars blue".

    * Be specific in your requests. For example, you can ask me to
    create a bar plot of **RF** vs. **Tag** with **gray** as the color.

    * Similarly, be specific about the features you want to modify or format.
    For example, you can ask me to change the **y-axis** label to **Relative Frequency (per 100 tokens)**.

    * Pay attention to the code that I use to create a plot.
    I add comments to the code that I generate, which can help you
    craft your requests using specific terms or commands.
    """  # noqa: E501

message_plotbot_data = """
    * To make **Reference** data available,
    you must process a reference corpus.

    * To make **Keywords** data available,
    you must create keywords tables either by
    **Comparing Corpora** or **Comparing Corpus Parts**.

    * To make **Group** data available you first have
    to process metadata (available in Manage Corpora).
    """  # noqa: E501

message_plotbot_libraries = """
    To create plots, I can use the following libraries:
    * [Plotly express](https://plotly.com/python/plotly-express/)
    * [Matplotlib](https://matplotlib.org/)
    * [Seaborn](https://seaborn.pydata.org/)

    Each library has its own aesthetics and features.
    If you're unfamiliar with them, you should check out their documentation,
    as well as examples of their use.
    """  # noqa: E501

message_pandabot_home = """
    Pandabot is a chat assistant designed to help you
    explore and analyze tabular data (or data frames).\n

    Though I can make plots, I'm not an interative assistant like Plotbot.
    But I am more flexible in making:
    ```
    Create a bar plot of RF vs. Tag with gray as the bar color.
    ```
    Once I generate a plot, you can ask me to modify its features:
    ```
    Remove "Untagged" from the Tag column
    ```
    I will then revise the plot with the requested changes.\n

    If you want me to create a new plot, use the **Clear chat history** button
    in the sidebar and begin a new conversation.
    """  # noqa: E501

message_pandabot_tips = """
    * Use plot-related commands like "Make plot horizontal" or "Make bars blue".

    * Be specific in your requests. For example, you can ask me to
    create a bar plot of **RF** vs. **Tag** with **gray** as the color.

    * Similarly, be specific about the features you want to modify or format.
    For example, you can ask me to change the **y-axis** label to **Relative Frequency (per 100 tokens)**.

    * Pay attention to the code that I use to create a plot.
    I add comments to the code that I generate, which can help you
    craft your requests using specific terms or commands.
    """  # noqa: E501

message_single_document = """
    :point_left: To use this page, first select a document. Then you can:\n
    * Choose up to 5 tags to highlight in the document.
    * Plot the location of those tags in the document.
    * Download highlighted text as a Word file.
    """

# Static messages that populate the sidebars.

message_download = """
    Your data is ready!
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
    """  # noqa: E501


message_naming = """
    Files must be in a TXT (plain text) format.
    If you are preparing files for the first time,
    it is recommended that you use a plain text editor
    (rather than an application like Word).
    Avoid using spaces in file names.
    Also, you needn't worry about preserving paragraph breaks,
    as those will be stripped out during processing.\n

    Metadata can be encoded at the beginning of a file name,
    before an underscore. For example: acad_01.txt, acad_02.txt,
    blog_01.txt, blog_02.txt.
    These would allow you to compare **acad** vs. **blog** as categories.
    You can designate up to 20 categories.
    """  # noqa: E501

message_models = """
    For detailed descriptions, see the tags tables available from the Help menu.
    But in short, the full dictionary has more categories and coverage than the common dictionary.
    """  # noqa: E501

message_association_measures = """
    The most common statistic for measuring token associations is Pointwise Mutual Information (PMI),
    first developed by [Church and Hanks](https://aclanthology.org/J90-1003/).
    One potentially problematic characteristic of PMI is that it rewards (or generates high scores) for low frequency tokens.

    This can be handled by filtering for minimum frequencies and MI scores.
    Alternatively, [other measures have been proposed, which you can select from here.](https://en.wikipedia.org/wiki/Pointwise_mutual_information)
    """  # noqa: E501

message_columns_collocations = """
    The **Freq Span** columns refers to the collocate's frequency within the given window,
    while **Freq Total** refers to its overall frequency in the corpus.
    Note that is possible for a collocate to have a *higher* frequency within a window, than a total frequency.\n
    The **MI** column refers to the association measure selected when the table was generated
    (one of NPMI, PMI2, PMI3, or PMI).
    """  # noqa: E501

message_columns_keyness = """
    The **LL** column refers to [log-likelihood](https://ucrel.lancs.ac.uk/llwizard.html),
    a hypothesis test measuring observed vs. expected frequencies.
    Note that a negative value means that the token is more frequent
    in the reference corpus than the target.\n

    **LR** refers to
    [Log-Ratio](http://cass.lancs.ac.uk/log-ratio-an-informal-introduction/),
    which is an [effect size](https://www.scribbr.com/statistics/effect-size/).
    And **PV** refers to the
    [p-value](https://scottbot.net/friends-dont-let-friends-calculate-p-values-without-fully-understanding-them/).\n

    The **AF** columns refer to the absolute frequencies
    in the target and reference.
    The **RF** columns refer to the relative frequencies
    (normalized **per million for tokens** and **per 100 for tags**).
    Note that for part-of-speech tags,
    tokens are normalized against word tokens,
    while DocuScope tags are normalized against counts of all tokens
    including punctuation.
    The **Range** column refers to the percentage of documents
    in which the token appears in your corpus.
    """  # noqa: E501

message_filters = """
    Filters can be accessed by clicking on the three lines
    that appear while hovering over a column header.
    For text columns, you can filter by 'Equals', 'Starts with', 'Ends with', and 'Contains'.\n
    Rows can be selected before or after filtering using the checkboxes.
    (The checkbox in the header will select/deselect all rows.)\n
    If rows are selected and appear in new table below the main one,
    those selected rows will be available for download in an Excel file.
    If no rows are selected, the full table will be processed for downloading after clicking the Download button.
    """  # noqa: E501

message_columns_tags = """
    The **AF** column refers to the absolute token frequency.
    The **RF** column refers to the relative token frequency (normalized **per 100 tokens**).
    Note that for part-of-speech tags, tokens are normalized against word tokens,
    while DocuScope tags are normalized against counts of all tokens including punctuation.
    The **Range** column refers to the percentage of documents in which the token appears in your corpus.
    """  # noqa: E501

message_columns_tokens = """
    The **AF** column refers to the absolute token frequency.
    The **RF** column refers to the relative token frequency (normalized **per million tokens**).
    Note that for part-of-speech tags, tokens are normalized against word tokens,
    while DocuScope tags are normalized against counts of all tokens including punctuation.
    The **Range** column refers to the percentage of documents in which the token appears in your corpus.
    """  # noqa: E501

message_anchor_tags = """
    You can choose to **anchor** at token to a specific tag.
    For example, if you wanted to disambiguate *can* as a noun (e.g., *can of soda*)
    from *can* as a modal verb, you could **anchor** the node word to a part-of-speech
    tag (like **Noun**, **Verb** or more specifically **VM**).

    For most cases, choosing an **anchor** tag isn't necessary.
    """  # noqa: E501

message_span = """
    Associations are calculated by counting the observed frequency within a
    span around a node word and comparing that to the frequency that we would expect
    given its overall frequency in a corpus.

    You could adjust the span if, for example, you wanted look at the subjects of a verb.
    For that, you would want to search only the left of the node word, setting the right span to 0.
    For verb object, you would want to do the opposite.
    There could be cases when you want a narrower window or a wider one.
    """  # noqa: E501

message_general_tags = """
    The conventions for aggregating tags follow those used by the
    [Corpus of Contemporary American English (COCA)](https://www.english-corpora.org/coca/).\n\n

    Nouns are identified by tags starting with **NN**,
    which means they are capturing **common nouns** not
    [proper nouns or pronouns](https://ucrel.lancs.ac.uk/claws7tags.html).\n\n

    Even more importantly verbs are identified by tags starting with **VV**.
    Those are assigned to **lexical verbs**.
    Modal verbs, for example, are identified by **VM**,
    and are not included in those counts.
    """  # noqa: E501

message_variable_contrib = """
    The plots are a Python implementation of [fviz_contrib()](http://www.sthda.com/english/wiki/fviz-contrib-quick-visualization-of-row-column-contributions-r-software-and-data-mining),
    an **R** function that is part of the **factoextra** package.
    """  # noqa: E501

# Style option hack to disable full page view of plots.
# Some don't render correctly in the full-page view.

message_disable_full = """
    <style> button[title="View fullscreen"] {
    display: none;
    } </style>
    """
