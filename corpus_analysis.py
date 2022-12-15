import streamlit as st

st.markdown("[![Made with DocuScope](https://raw.githubusercontent.com/browndw/corpus-tagger/main/_static/docuscope_badge.svg)](https://www.cmu.edu/dietrich/english/research-and-publications/docuscope.html) ![](https://raw.githubusercontent.com/browndw/corpus-tagger/main/_static/porpoise_badge.svg)")

with st.expander("About the app"):
	st.markdown("""
		The app is wrapper for the [docuscospacy](https://docuscospacy.readthedocs.io/en/latest/) Python package.
		The app and package are built around a specially trained [spaCy model](https://huggingface.co/browndw/en_docusco_spacy). 
		The model, package and app were created by David West Brown at Carnegie Mellon University.
		""")
st.markdown("[![User Guide](https://raw.githubusercontent.com/browndw/corpus-tagger/main/_static/user_guide.svg)](https://browndw.github.io/docuscope-docs/)")

st.title("DocuScope Corpus Analysis Online")
st.markdown("## DocuScope and part-of-speech tagging with spaCy")
st.markdown("""
			This application is designed for the analysis of corpora assisted by part-of-speech and rhetorical tagging.
			With the application users can:\n
			1. process small-ish corpora (~2 million words)\n
			2. create frequency tables of words, phrases, and tags\n
			3. calculate associations around node words\n
			4. generate key word in context (KWIC) tables\n
			5. compare corpora or sub-corpora\n
			6. explore single texts\n
			7. practice advanced plotting
		""")

st.markdown("---")

st.markdown("## Using the tool")
st.markdown("""
			If you've used a concordancer like [AntConc](https://www.laurenceanthony.net/software/antconc/) or WordSmith, using DocuScope CA should be relatively intuitive.
			You simply need to:\n
			1. create a corpus of plain text files;\n
			2. load the corpus into the tool (using the **load corpus** tab on the left);\n
			3. explore!
		""")

st.markdown("---")

st.markdown("### User guide")
st.markdown("""
			For detailed instructions on how to use the tool, consult [the documentation](https://browndw.github.io/docuscope-docs/).
		""")



