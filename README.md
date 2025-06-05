
# DocuScope Corpus Analysis & Concordancer

<div class="image" align="center">
    <img width="150" height="auto" src="https://github.com/browndw/corpus-tagger/raw/main/_static/docuscope-logo.png" alt="DocuScope logo">
    <br>
</div>

---
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://browndw-corpus-tagger-index-st-deploy-3f662p.streamlit.app/) [![](https://badge.fury.io/py/docuscospacy.svg)](https://badge.fury.io/py/docuscospacy) [![](https://readthedocs.org/projects/docuscospacy/badge/?version=latest)](https://browndw.github.io/docuscope-docs/) [![](https://zenodo.org/badge/512227318.svg)](https://zenodo.org/badge/latestdoi/512227318) [![Built with spaCy](https://img.shields.io/badge/made%20with%20❤%20and-spaCy-09a3d5.svg)](https://spacy.io) ![](https://raw.githubusercontent.com/browndw/corpus-tagger/main/_static/porpoise_badge.svg)

## DocuScope and Part-of-Speech tagging with spaCy

This application is designed for the analysis of small corpora assisted by part-of-speech and rhetorical tagging.

With the application users can:

1. process small corpora
2. create frequency tables of words, phrases, and tags
3. calculate associations around node words
4. generate key word in context (KWIC) tables
5. compare corpora or sub-corpora
6. explore single texts
7. practice advanced plotting


Also note that you can activate/deactivate options from the `options.toml` file.

When running locally:

- [ ] Clone this repository.
- [ ] Create a virtual environment.
- [ ] Navigate to the directory.
- [ ] Install the requirements.
- [ ] Set `desktop_mode` to `True`.

Then run:

```
streamlit run webapp/index.py
```

> [!IMPORTANT]
> Features can like `desktop_mode` can be activated/deactivated from the `options.toml` file. Their defaults are set at their most restrictive.