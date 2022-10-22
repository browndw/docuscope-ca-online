import streamlit as st

st.title("DocuScope and Part-of-Speech tagging with spaCy")

st.markdown("This application is designed for the analysis of small corpora assisted by part-of-speech and rhetorical tagging.")

st.markdown("With the application users can:\n1. process small corpora\n2. create frequency tables of words, phrases, and tags\n3. calculate associations around node words\n4. generate key word in context (KWIC) tables\n5. compare corpora or sub-corpora\n6. explore single texts\n7. practice advanced plotting")

st.markdown("#### The model")

st.markdown("Importantly, this tool uses neither the CLAWS7 tagger nor DocuScope. Rather, it relies on a [spaCy model](https://huggingface.co/browndw/en_docusco_spacy) trained on those tagsets.")

st.markdown("If you are using this tool for research, that distinction is imporant to make in your methodology.")

st.markdown("CLAWS should be attribed to [Leech et al.](https://aclanthology.org/C94-1103.pdf):\n\n*Leech, G., Garside, R., & Bryant, M. (1994). CLAWS4: the tagging of the British National Corpus. In COLING 1994 Volume 1: The 15th International Conference on Computational Linguistics.*")

st.markdown("And DocuScope to [Kaufer and Ishizaki](https://www.igi-global.com/chapter/content/61054):\n\n*Ishizaki, S., & Kaufer, D. (2012). Computer-aided rhetorical analysis. In Applied natural language processing: Identification, investigation and resolution (pp. 276-296). IGI Global.*")

