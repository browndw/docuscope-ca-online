import streamlit as st

st.sidebar.markdown("## More about DocuScope and CLAWS7")

st.title("More about the tagsets")

st.markdown("## What is DocuScope?\n")
st.markdown("DocuScope is a dictionary-based tagger, developed by [David Kaufer and Suguru Ishizaki](https://www.igi-global.com/chapter/computer-aided-rhetorical-analysis/61054) at Carnegie Mellon University.")
st.markdown("It consists of an enormous lexicon organized into a 3-level taxonomy. An analogue would be the [lexicons typically used in sentiment analysis](https://saifmohammad.com/WebPages/lexicons.html). Those usually organize words and phrases into 2 categories (positive and negative) and work by matching strings over a corpus of texts.")
st.markdown("DocuScope works in the same basic way, but organizes its strings into many more categories and is orders of magnitude larger. A typical sentiment lexicon may match 3-5 thousand strings. DocuScope matches 100s of millions. You can find a small, early version of the dictionary [here](https://github.com/docuscope/DocuScope-Dictionary-June-26-2012).")
st.markdown("It is also trained on the [CLAWS7](https://ucrel.lancs.ac.uk/claws7tags.html) part-of-speech tagset.")
st.markdown("NOTE: this demo is public - please don't enter confidential text")

		