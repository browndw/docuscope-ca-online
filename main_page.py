import streamlit as st

st.title("DocuScope and Part-of-Speech tagging with spaCy")

st.markdown("This application is designed for the analysis of small corpora assisted by part-of-speech and rhetorical tagging.")
st.markdown("With the application users can:\n1. process small corpora\n2. create frequency tables of words, phrases, and tags\n3. calculate associations around node words\n4. generate key word in context (KWIC), tables\n5. compare corpora or sub-corpora\n6. explore single texts\n7. practice advanced plotting")
st.markdown("It uses a trained spaCy model ([en_docusco_spacy](https://huggingface.co/browndw/en_docusco_spacy)) to identify DocuScope categories in text.")
st.markdown("It is also trained on the [CLAWS7](https://ucrel.lancs.ac.uk/claws7tags.html) part-of-speech tagset.")
st.markdown("NOTE: this demo is public - please don't enter confidential text")
