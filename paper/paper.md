---
title: 'DocuScope Corpus Analysis & Concordancer: A Streamlit Application for Rhetorical and Linguistic Text Analysis'
tags:
  - Python
  - corpus linguistics
  - natural language processing
  - rhetorical analysis
  - DocuScope
  - digital humanities
  - text analysis
  - streamlit
authors:
  - name: David West Brown
    orcid: 0000-0001-7745-6354
    corresponding: true
    affiliation: "1"
affiliations:
 - name: Carnegie Mellon University, Department of English
   index: 1
date: 19 October 2025
bibliography: paper.bib
---

<!-- markdownlint-disable MD025 -->

# Summary

DocuScope Corpus Analysis & Concordancer is a Streamlit application for corpus and rhetorical text analysis. It combines spaCy linguistic annotation with DocuScope rhetorical tagging and runs in either desktop or multi-user modes. A headless API and CLI allow scripted workflows without the web interface.

Version 0.4.1 of the software is archived on Zenodo (doi:10.5281/zenodo.17392153) [@brown2025docuscopeca].

# Statement of need

Corpus linguistics and computational text analysis are established methods in linguistics, writing studies, and digital humanities [@biber2011corpus; @mcenery2012corpus]. However, existing tools present researchers with a fragmented landscape that forces compromises between accessibility and analytical depth.

Established tools like AntConc [@anthony2005antconc] excel at concordancing, frequency analysis, and keyword identification but provide no part-of-speech or rhetorical annotation capabilities. Web-based platforms like Voyant Tools [@voyant2016] offer accessible text visualization and basic analysis but similarly lack linguistic tagging and rhetorical analysis features.

Code-centric frameworks (spaCy, NLTK) provide sophisticated linguistic processing but require substantial programming expertise and offer no built-in rhetorical analysis. Proprietary tools often combine features but lack transparency, reproducibility controls, and flexible deployment options.

The DocuScope rhetorical taxonomy [@kaufer2004power] addresses systematic rhetorical analysis, identifying functional language patterns beyond surface-level linguistic features. However, DocuScope's established implementations relied on rule-based string matching, limiting integration with modern NLP pipelines and restricting adoption outside specialized research groups with access to proprietary tools. This barrier is particularly problematic in educational contexts, where students and novice researchers need access to authentic corpus analysis without first mastering programming or command-line interfaces.

No existing tool combines: (1) DocuScope's hierarchical rhetorical tagging, (2) contemporary linguistic annotation (POS, lemmatization), (3) transparent provenance tracking, (4) flexible deployment (desktop, web, headless), (5) reproducible workflows, and (6) educational accessibility in a single package.

DocuScope CA addresses this gap by unifying rhetorical and linguistic analysis in a deployable system that serves both exploratory users (via Streamlit interface) and reproducible workflows (via API/CLI). The intuitive web interface enables students and novices to engage directly with real corpus data, conducting sophisticated analyses without programming prerequisites. Meanwhile, the provenance manifest ensures transparency, and multiple deployment options accommodate diverse institutional and individual needs.

# Software Design

DocuScope CA builds upon two decades of DocuScope rhetorical taxonomy development [@kaufer2004power; @kaufer2023docuscope] by modernizing the framework from rule-based string matching to trained spaCy models, dramatically expanding reach and accessibility. This work extends rather than replaces the existing DocuScope ecosystem: the rhetorical dictionaries and linguistic theory remain foundational, while the technical implementation enables integration with contemporary NLP infrastructure and open-source distribution.

Creating new software was necessary because existing corpus tools lacked the architectural capacity for this integration. AntConc lacks extensibility for custom trained models; Voyant Tools runs exclusively in browser contexts incompatible with multi-stage NLP pipelines; spaCy and NLTK require programming expertise that would exclude our educational audience. The design separates processing logic from presentation, enabling the same analytical core to serve interactive learners (Streamlit UI), reproducible research scripts (API/CLI), and diverse deployment contexts (desktop, web, container, hosted).

Key trade-offs included choosing Polars over Pandas (performance over ecosystem maturity), Streamlit over Flask/Django (rapid development over fine-grained control), and bundling pre-trained models (reviewer accessibility over package size), reflecting the dual mandate of research-grade performance and educational accessibility that existing tools sacrifice.

# Contribution and novelty

DocuScope CA makes several key technical contributions: (1) first open-source implementation integrating DocuScope rhetorical tagging with spaCy's linguistic pipeline through custom trained models; (2) dual desktop/multi-user deployment architecture separating processing logic from interface; (3) explicit provenance manifest capturing version, model, content hashes, and processing parameters for reproducible analysis; (4) compact API and CLI enabling scripted workflows alongside interactive exploration. Together, these enable accessible, reproducible corpus analysis combining rhetorical depth with modern NLP capabilities in ways not previously available through open-source tools.

# Implementation

Python 3.11 stack: spaCy [@spacy2020] plus a DocuScope extension for joint linguistic/rhetorical tagging; Polars for columnar data processing [@polars2023]; Streamlit for the interface [@streamlit2023]; Plotly for visualization; `docuscospacy` for tag generation. Parsing and metric computation are isolated from presentation so the same core serves both UI and scripted contexts. The API/CLI expose only the minimal surface required for ingestion, parsing, metrics, and export. Tests exercise parsing, session persistence, and analysis routines. All core functionality operates offline with bundled models under `webapp/_models/`; external API keys are required only for optional AI-assisted analysis features.

## Ecosystem

DocuScope CA operates within a broader ecosystem designed for textual analysis. The architecture centers on the `docuscospacy` Python package, which extends spaCy with DocuScope rhetorical tagging capabilities. Pre-trained models are distributed via HuggingFace Hub, built from curated training datasets also available on HuggingFace, ensuring transparent model provenance and reproducibility.

This ecosystem supports multiple deployment modes: the web application (this paper), a cross-platform desktop application, and headless API/CLI access. The web application prioritizes educational accessibility and collaborative research, while the desktop version serves individual researchers requiring offline capabilities.

The layered design separates processing logic from interface concerns. Core functions handle corpus ingestion, spaCy+DocuScope parsing, and metric computation, with results cached by content hash to avoid redundant processing.

# Usage and reproducibility

Users may run a hosted instance, local container, desktop build, or the headless API. A sample corpus and script (`paper/scripts/run_example.py`) generate deterministic token annotations, frequency and tag tables, and a manifest (version, model, hashes, counts). Programmatic example:

```python
from docuscope_ca import process_corpus
res = process_corpus('paper/data/test_corpus', metrics=('freq','tags'))
print(res.manifest['corpus']['total_tokens'])
```

Artifacts can be regenerated to validate results.

## Interactive workflow

The typical interactive workflow demonstrates how students and researchers can conduct sophisticated corpus analysis without programming knowledge: (a) select from built-in sample corpora or upload custom text collections; (b) process the corpus through the integrated spaCy+DocuScope pipeline to generate token-level linguistic and rhetorical annotations; (c) process metadata (encoded into file names); (d) explore frequency distributions across tokens, part-of-speech tags, and rhetorical categories; (e) apply filters, create visualizations, and export results for statistical analysis. This workflow supports exploratory discovery and hypothesis-driven research while maintaining provenance tracking.

![Landing page showing primary navigation menu and real-time processing status indicators for corpus analysis workflows.](figures/01_LandingWithNav.png){#fig:landing width=90%}

![Corpus management interface allowing users to select from internal sample datasets or upload custom text collections with automatic format detection.](figures/02_ManageCorpusData.png){#fig:manage width=90%}

![Token frequency analysis displaying sortable, filterable tables of word frequencies with part-of-speech and rhetorical tag annotations, ready for download.](figures/06_TokenFrequencies.png){#fig:freq width=90%}

![Advanced filtering interface enabling users to refine analysis by applying multiple criteria to focus on specific linguistic or rhetorical patterns of interest.](figures/07_TokenFrequenciesFilter.png){#fig:filters width=90%}

## Performance

Benchmark (50 docs; 132k words; Python 3.11.8; 8‑core, 24 GB RAM) achieved ~5.6 documents/s (~890k words/min steady state, ≈1.1 min per million words) excluding initial model load. Contributing factors: batched spaCy calls, vectorized Polars group‑bys, minimal intermediate serialization, and hash‑based avoidance of duplicate work.

# Research Impact Statement

DocuScope CA demonstrates realized impact through institutional deployment and ecosystem integration. The software serves approximately 500 first-year students per semester in a Carnegie Mellon writing course focused on data-driven analysis, with additional usage via a hosted enterprise instance (docuscope-ca.eberly.cmu.edu). Cross-platform desktop applications enable offline usage. The underlying `docuscospacy` package receives 150-200 monthly PyPI downloads, indicating broader adoption. Pre-trained models distributed via HuggingFace Hub (browndw/en_docusco_spacy) establish transparent provenance and facilitate ecosystem integration.

The software provides novel capability by unifying DocuScope rhetorical analysis—previously available only through proprietary tools—with modern open-source NLP infrastructure in an accessible package. This combination enables corpus-based rhetorical analysis at scale, as demonstrated in published research utilizing the framework [@brown2022stylistic; @wetzel2021computer] and the recent edited volume on DocuScope applications [@brown2023corpora].

Community-readiness is evidenced through comprehensive testing (unit, integration, performance, and UI tests), Apache 2.0 licensing, extensive documentation, multiple deployment modes, reproducible benchmark workflows, and formal citation metadata (CITATION.cff) with Zenodo archival (doi:10.5281/zenodo.17392153). The separation of processing logic from interface enables integration into diverse workflows, from classroom instruction to large-scale corpus studies, addressing methodological gaps in digital humanities and corpus linguistics that existing fragmented tools leave unresolved.

# AI Usage Disclosure

DocuScope CA was developed over a three-year period beginning in 2022, prior to the widespread availability of AI-assisted coding tools. The project has maintained public repositories since its inception, beginning with an initial GUI wrapper (DocuConc, https://github.com/browndw/DocuConc) before evolving to the current Streamlit-based architecture. The software architecture, core processing pipeline, and user interface were designed and implemented without generative AI assistance. The software itself includes optional AI-assisted analysis features (utilizing the OpenAI API) that users may enable for exploratory data analysis; these features are clearly documented as experimental and optional. This paper was written without the use of generative AI tools for content generation or authoring.

# Acknowledgements

I acknowledge the DocuScope team at Carnegie Mellon University for the rhetorical framework, the spaCy development team for NLP infrastructure, and the Streamlit team for the web framework.

# References

<!-- References will be automatically generated from paper.bib -->
