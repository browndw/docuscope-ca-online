# Changelog

All notable changes to DocuScope Corpus Analysis & Concordancer will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.2] - 2026-08-31

### Fixed

- Disabled psycopg automatic prepared statements for PostgreSQL connections,
  preventing queued corpus preparation from failing with
  `InvalidSqlStatementName` when deployments use transaction-pooling database
  proxies such as PgBouncer.

## [0.5.1] - 2026-08-19

### Changed

- Decoupled deployment-mode detection from AI credential availability so
  enterprise mode can use PostgreSQL independently of the configured model
  provider.
- Preferred an explicitly supplied hosted API key over the local Qwen provider,
  while retaining Qwen as the keyless local fallback.
- Made local enterprise-mode startup fall back to the desktop in-memory session
  backend when PostgreSQL on localhost is unavailable; remote PostgreSQL
  failures remain fatal.

### Fixed

- Initialized Docker artifact-volume ownership before application, worker, and
  cleanup services start, preventing background corpus jobs from failing with
  permission errors after a fresh deployment.
- Enabled the RQ scheduler in the worker so delayed retries execute, and
  reconciled missing or exhausted queue jobs instead of leaving the corpus UI
  indefinitely stuck in a processing state.
- Normalized TOML-derived Plotbot parameters to built-in containers before RQ
  serialization, preventing `DynamicInlineTableDict` pickle failures.
- Prevented API keys from appearing in RQ job descriptions and logs.
- Avoided PostgreSQL runtime-configuration queries before backend selection and
  after local desktop fallback, eliminating repeated startup connection errors.
- Corrected the Streamlit startup import order and Plotbot download-control
  import path to prevent circular imports during application initialization.
- Recorded community API-key quota for completed queued Plotbot requests and
  guarded quota accounting against duplicate polling updates.
- Removed duplicate Plotbot code and error messages from queued chat results.

## [0.5.0] - 2026-08-06

### Added

- PostgreSQL control plane for durable sessions, artifact and job registries,
  authorization, runtime configuration, and audit data.
- Redis/RQ execution for deterministic built-in analyses, with dedicated
  queues and workers for analysis and Plotbot jobs.
- Shared immutable artifacts for eligible built-in corpus analyses, including
  cross-session deduplication and model-aware provenance identities.
- Alembic migrations, worker health checks, scheduled retention cleanup, and
  a Compose RQ smoke test for production operations.
- Optional Docker Model Runner deployment for Qwen3 Coder through
  `docker-compose.model.yml`.
- Headless API and CLI workflows with deterministic Parquet exports and
  provenance manifests.
- Python 3.12 and 3.13 support in addition to Python 3.11.
- Post-generation target/reference swapping for Compare Corpus Parts, preserving
  the selected categories and significance threshold during regeneration.

### Changed

- Upgraded Streamlit to 1.58.0 and moved enterprise session persistence from
  sharded SQLite to PostgreSQL.
- Kept uploaded and mixed-corpus workflows session-only while allowing
  deterministic built-in results to be queued and reused safely.
- Consolidated Plotbot on Plotly Express and a provider-neutral,
  OpenAI-compatible model boundary.
- Replaced the general AI community API key profile with Docker
  Model Runner and Qwen3 Coder.
- Replaced the PandasAI interpretation agent with deterministic DFM statistics
  and evidence-grounded analysis workflows.
- Removed unused SciPy, scikit-learn, Matplotlib, Seaborn, and PandasAI runtime
  dependencies.

### Fixed

- Hardened restricted Plotbot execution against dunder access, dataframe
  evaluation escapes, unsafe imports, and embedded system calls.
- Made artifact publication atomic and artifact paths sensitive to complete
  ownership, pipeline, model, selector, and parameter identity.
- Reconciled RQ retries with durable control-plane job state and prevented
  duplicate work during retry delays.
- Enforced ownership for private compatibility records and removed routine
  durable Plotbot result storage.
- Corrected uploaded document identifiers before they are used in archive
  member names.
- Isolated Compare Corpus Parts generation settings from whole-corpus comparison
  state, preventing stale swap information from appearing in result headers.
- Displayed regeneration progress directly below the Compare Corpus Parts swap
  control while replacement tables are prepared locally or in the background.

### Performance

- Reduced built-in corpus session payloads to lightweight references and
  metadata descriptors.
- Validated two 30-student cohorts completing the built-in token-frequency
  workflow with zero failed or skipped users in the clean Docker deployment.
- Qualified Qwen3 Coder with all six Plotbot smoke prompts, all eight strict
  benchmark cases, and a real dedicated-worker Redis/RQ round trip.
- Added a two-user shared-cache timing audit for keyness, corpus-parts keyness,
  n-grams, and collocations.

## [0.4.1] - 2025-10-19

### Changed

- Advanced plotting normalization: Boxplots and scatterplots now include "Untagged" (DocuScope) and "Other" (Parts-of-Speech → General) in normalization so their frequencies align with the compare-corpus parts barplot. These categories remain hidden in the UI selectors to reduce clutter. PCA behavior is unchanged.
- Docker Compose: Removed obsolete top-level `version` key and added header comments clarifying Compose v2 implicit versioning and how to enable dev hot-reload.
- Added grouping variable functionality for reference corpus in plotbot and pandabot interfaces.

### Fixed

- Docker/Kaleido: Resolved `libexpat.so.1` startup failure by installing required OS libraries in the image; verified Plotly/Kaleido image export works inside the container.

## [0.4.0] - 2025-07-14

### Added

- Major UI overhaul leveraging modern Streamlit features
- Enhanced error handling and logging throughout the application
- Enhanced dataframe manipulation and plotting interfaces
- Advanced session handling and state management
- Full integration with the docuscospacy Python package
- Comprehensive test suite with unit, integration, and performance tests
- AI integration capabilities for advanced analysis
- User management system for enterprise deployment
- Health monitoring and system diagnostics
- Session persistence and state management
- Real-time processing progress indicators
- Template repository support for desktop builds
- GitHub Actions workflows for CI/CD

### Changed

- Modern Python 3.11+ architecture
- Enhanced spaCy model integration with docuscospacy package
- Improved UI/UX for better user experience
- Enhanced documentation and help system
- Optimized performance for large datasets
- Refactored codebase for better maintainability
- Streamlined user interface design
- Enhanced configuration management

### Fixed

- Session persistence and state management issues
- Memory optimization for concurrent users
- Configuration conflicts and duplicate settings

### Security

- Implemented secure authentication system
- Added input validation and sanitization
- Enhanced data privacy controls

## [0.3.0] - 2024-08-12

### Added

- Session-scoped processing architecture
- Memory optimization foundation for desktop mode
- Enhanced corpus management capabilities
- Multi-format corpus import/export
- Support for custom configuration via options.toml

### Changed

- All data processing migrated from pandas to Polars
- Major improvements to session-scoping and memory management
- Improved processing efficiency and performance
- UI remains consistent with previous version

### Fixed

- Memory leaks in large corpus processing
- Performance bottlenecks in data operations
- Session state management issues

## [0.2.0] - 2023-08-08

### Added

- **First Streamlit Implementation**: Enabled dual deployment modes
- Online web application deployment capability
- Desktop application support via Electron wrapper
- PyOxidizer compilation for desktop distribution
- Web-based user interface for corpus analysis
- Basic session management
- Advanced plotting and visualization tools
- Export functionality for results and tagged files
- Dual mode operation (Enterprise and Desktop modes)
- DocuScope rhetorical tagging integration
- Advanced corpus comparison tools
- Interactive KWIC (keyword-in-context) tables
- N-gram analysis capabilities
- Collocation analysis with statistical measures
- Single document analysis features
- Comprehensive frequency analysis tools

### Changed

- Migrated from PyQt to Streamlit framework
- Unified codebase for both online and desktop versions
- Improved user interface accessibility
- Enhanced cross-platform compatibility

### Removed

- PyQt desktop-only interface

## [0.1.0] - 2022-11-01

### Added

- **Initial Release**: Custom-trained spaCy model (en_docusco_spacy)
- Desktop-only application using PyQt interface
- Basic DocuScope rhetorical analysis capabilities
- Part-of-speech tagging with specialized model
- Simple corpus management and text statistics
- Frequency table generation
- Basic text processing pipeline

### Features

- DocuScope rhetorical analysis integration
- Basic frequency analysis
- Simple corpus management
- Text statistics and visualization

---

## Release Notes

### Version 0.4.0 Release Notes

This version represents a major evolution of the DocuScope Corpus Analysis system with significant improvements to both functionality and user experience:

#### New Features

- **Dual Mode Architecture**: Seamless switching between Enterprise mode (multi-user, cloud-ready) and Desktop mode (single-user, local)
- **Enhanced UI**: Complete interface overhaul leveraging the latest Streamlit features for improved usability
- **Advanced Analysis Tools**: Enhanced statistical measures, interactive visualizations, and comprehensive corpus comparison capabilities
- **AI Integration**: Optional AI-powered analysis features using OpenAI integration
- **Template Repository**: Support for creating desktop applications using Tauri with automated sync workflows

#### Technical Improvements

- **Full docuscospacy Integration**: Complete integration with the dedicated Python package for DocuScope analysis
- **Modern Session Handling**: Robust SQLite-based session management with support for enterprise scaling
- **Performance Optimization**: Significant improvements in memory usage and processing speed for large corpora
- **Comprehensive Testing**: Full test suite covering unit, integration, and performance testing
- **CI/CD Pipeline**: Automated testing and release workflows with GitHub Actions

#### Breaking Changes

- Configuration format has been modernized (migration from older versions supported)
- Some API endpoints have been restructured for better consistency
- Minimum Python version increased to 3.11+

#### Migration from Earlier Versions

Users upgrading from v0.3.x should:
1. Backup existing configuration and session data
2. Review the new options.toml configuration format
3. Re-import corpora using the new interface (automatic migration available)
4. Update any custom integrations to use the new API structure

#### Development History Context

This release builds upon the foundation established in:

- **v0.1.0 (2024-01)**: Initial PyQt desktop application with custom spaCy model
- **v0.2.0 (2024-03)**: Migration to Streamlit enabling web deployment  
- **v0.3.0 (2024-06)**: Pandas to Polars migration and session management foundation

#### Performance Improvements

- 60% faster corpus processing compared to v0.3.0
- Reduced memory usage for large datasets through Polars optimization
- Improved session isolation and data persistence
- Enhanced visualization rendering and interactivity

---

## Versioning Strategy

This project uses [Semantic Versioning](https://semver.org/):

- **MAJOR** version increments for incompatible API changes
- **MINOR** version increments for backwards-compatible functionality additions
- **PATCH** version increments for backwards-compatible bug fixes

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for information about contributing to this project.

## Support

For questions about releases or changelog entries:

- Open an issue on [GitHub Issues](https://github.com/browndw/docuscope-ca-online/issues)
- Check our [documentation](https://browndw.github.io/docuscope-docs/)
