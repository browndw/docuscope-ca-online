import streamlit as st

with st.expander("About the app"):
	st.markdown("""
		The app is wrapper for the [docuscospacy](https://docuscospacy.readthedocs.io/en/latest/) Python package.
		The app and package are built around a specially trained [spaCy model](https://huggingface.co/browndw/en_docusco_spacy). 
		The model, package and app were created by David West Brown at Carnegie Mellon University.
		""")


st.title("DocuScope and Part-of-Speech tagging with spaCy")
st.markdown("""
			This application is designed for the analysis of corpora assisted by part-of-speech and rhetorical tagging.
			With the application users can:\n
			1. process smallis corpora (~2 million words)\n
			2. create frequency tables of words, phrases, and tags\n
			3. calculate associations around node words\n
			4. generate key word in context (KWIC) tables\n
			5. compare corpora or sub-corpora\n
			6. explore single texts\n
			7. practice advanced plotting
		""")

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

st.markdown("#### Multi-word (or multi-token) units")
st.markdown("""
		Unlike most [tokenizers](https://en.wikipedia.org/wiki/Text_segmentation#Word_segmentation) and concordancers,
		this application aggregates multi-word sequences into a single token.
		This most commonly happens with DocuScope as it identifies many phrasal units (as in the table above).
		But this also occurs with part-of-speech tagging in tokens like:
		>*in spite of*, *for example*, *in addition to*
		""")

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

