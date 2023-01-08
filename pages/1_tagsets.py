import streamlit as st

st.title("Linguistically-informed and rhetorically-informed tagging")

st.markdown("[![User Guide](https://raw.githubusercontent.com/browndw/corpus-tagger/main/_static/user_guide.svg)](https://browndw.github.io/docuscope-docs/tagset_docuscope.html)")

st.markdown("""
	The application is designed to bridge traditional linguistically-informed corpus analysis with
	rhetorically-informed corpus analysis (or computational rhetoric). 
	Users can generate data that organize tokens into conventional lexical classes or 
	that organize tokens into rhetorical categories like **Narrative** or **Public Terms**.\n
	The application allows users to toggle easily between both allowing them to explore and
	find patterns they find have explanatory power.
	""")
st.markdown("---")

st.markdown("#### Toggling between tagsets")
st.markdown("""
		The application produces outputs of the data with tokens identified by parts-of-speech and DocuScope.
		For example, part of a frequency can appear like this:
		
		| Token     | Tag  | AF  | RF       | Range |
		|-----------|------|-----|----------|-------|
		| was       | VBDZ | 594 | 4421.15  | 92%   |
		| can       | VM   | 423 | 3148.39  | 94%   |
		| students  | NN2  | 149 | 1109.014 | 20%   |
		| community | NN1  | 141 | 1049.46  | 30%   |
		
		[CLAWS7 tagset](https://ucrel.lancs.ac.uk/claws7tags.html)
		
		Or like this:
		
		| Token        | Tag                    | AF  | RF      | Range |
		|--------------|------------------------|-----|---------|-------|
		| , but        | Metadiscourse Cohesive | 193 | 1259.53 | 68%   |
		| can be       | Confidence Hedged      | 118 | 770.07  | 62%   |
		| will be      | Future                 | 113 | 737.44  | 52%   |
		| participants | Character              | 103 | 672.18  | 14%   |
		
		[DocuScope tagset](https://docuscospacy.readthedocs.io/en/latest/docuscope.html#categories)
		
		Users can toggle between the tagsets within each tool.
		In this way, the tool invites users to explore linguistic structure and variation from multiple perspectives.
		""")

st.markdown("---")

st.markdown("#### Multi-word (or multi-token) units")
st.markdown("""
		Unlike most [tokenizers](https://en.wikipedia.org/wiki/Text_segmentation#Word_segmentation) and concordancers,
		this application aggregates multi-word sequences into a single token.
		This most commonly happens with DocuScope as it identifies many phrasal units (as in the table above).
		But this also occurs with part-of-speech tagging in tokens like:
		>*in spite of*, *for example*, *in addition to*
		""")

st.markdown("---")

st.markdown("#### The model vs. the tools")
st.markdown("""
		Importantly, this application uses neither the CLAWS7 tagger nor DocuScope.
		Rather, it relies on a [spaCy model](https://huggingface.co/browndw/en_docusco_spacy) trained on those tagsets.
		Users interested in the fully functional tools can find them here:
		* [CLAWS7](https://ucrel.lancs.ac.uk/claws/)
		* [DocuScope CA](https://cmu.flintbox.com/technologies/dcb2a164-b661-495d-a5b5-404871842268)
		If you are using this application for research, that distinction is important to make in your methodology.
		And those whose work upon which this application is based should always be appropriately cited.
		""")

st.markdown("""
		CLAWS should be attribed to [Leech et al.](https://aclanthology.org/C94-1103.pdf):\n\n
		>*Leech, G., Garside, R., & Bryant, M. (1994). CLAWS4: the tagging of the British National Corpus. In COLING 1994 Volume 1: The 15th International Conference on Computational Linguistics.*
		""")

st.markdown("""
		And DocuScope to [Kaufer and Ishizaki](https://www.igi-global.com/chapter/content/61054):\n\n
		>*Ishizaki, S., & Kaufer, D. (2012). Computer-aided rhetorical analysis. In Applied natural language processing: Identification, investigation and resolution (pp. 276-296). IGI Global.*
		""")

st.markdown("---")

st.markdown("""
	## DocuScope\n
	[DocuScope tagset](https://docuscospacy.readthedocs.io/en/latest/docuscope.html#categories)
	""")

st.markdown("""
	DocuScope is a dictionary-based tagger, developed by [David Kaufer and Suguru Ishizaki](https://www.igi-global.com/chapter/computer-aided-rhetorical-analysis/61054) at Carnegie Mellon University.\n
	It consists of an enormous lexicon organized into a 3-level taxonomy. 
	An analogue would be the [lexicons typically used in sentiment analysis](https://saifmohammad.com/WebPages/lexicons.html). 
	Those usually organize words and phrases into 2 categories (positive and negative) and work by matching strings over a corpus of texts.\n
	DocuScope works in the same basic way, but organizes its strings into many more categories and is orders of magnitude larger.
	A typical sentiment lexicon may match 3-5 thousand strings. 
	DocuScope matches 100s of millions. You can find a small, early version of the dictionary [here](https://github.com/docuscope/DocuScope-Dictionary-June-26-2012).
	""")

st.markdown("---")

st.markdown("""
	## CLAWS7\n
	[CLAWS7 tagset](https://ucrel.lancs.ac.uk/claws7tags.html)
	""")

st.markdown("""
	CLAWS7 is robust part-of-speech tagset developed at Lancaster University. [(Try it!)](http://ucrel-api.lancaster.ac.uk/claws/free.html)
	It is used in the [BYU family of corpora](https://www.english-corpora.org/), 
	a project headed by Mark Davies and in ongoing development.
	The popularity of those data sets partly motivated the choice to use this tagset -- as opposed to say the [Penn Treebank tagset](https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html).\n
	This allows users to compare their results directly to those from much larger corpora.
	The tradeoff, is that CLAWS7 has is large and relatively fine-grained, 
	which can be challenging for users with less familiarity to English syntax.
	""")

st.markdown("---")

st.markdown("""
	## Limitations and warnings
	""")

st.markdown("""
	[The model that produces the tags](https://huggingface.co/browndw/en_docusco_spacy) was trained on American English.
	How it would perform on other varieties is unknown at this point.
	The model was also trained on roughly 100,000,000 words. 
	There are plans for a more rigorously trained model, but the preparation of training data is time-consuming.\n
	Also note, that part-of-speech tagging is, on the whole, more accurate that the rhetorical tagging (92.50% vs. 74.87%).
	As with any tagging system, what is generated by the model may not match the reader experience at the token- or text-level.
	Their potential lies in what they can reveal at scale.
	""")
	