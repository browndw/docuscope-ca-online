# Browser audits and load tests

The Artillery scenarios in `scenarios/` measure focused browser workflows under
load. The standalone Playwright audit in `scripts/page-audit.mjs` is a
single-user functional journey: it preserves one Streamlit websocket, prepares
corpora, visits every core page, exercises a page-specific interaction, and
writes a JSON report under `reports/`.

## Page audit profiles

Run both profiles:

```bash
npm run test:audit
```

Run one profile:

```bash
npm run test:audit:uploads
npm run test:audit:analysis
```

The profiles intentionally have different contracts:

- `uploads` processes text files from `test_data/tar_corpus` and
  `test_data/ref_corpus`. Those filenames expose one category per corpus. The
  expected metadata outcome is therefore a graceful 1-category rejection, and
  Compare Corpus Parts must display its no-metadata guidance rather than
  generate a comparison.
- `analysis` loads `A_MICUSP_mini` and `C_BAWE_mini`, processes MICUSP metadata,
  and exercises analytical workflows. Compare Corpus Parts explicitly compares
  BIO against ENG.

A page cannot pass with zero interactions. Reports include each page's actual
route, duration, interactions, displayed alerts, Streamlit exceptions, browser
errors, failed requests, and HTTP errors. Failed pages also produce screenshots
in `reports/page-audit-artifacts/`.

## Configuration

Environment variables:

- `LOAD_TEST_BASE_URL`: application URL; defaults to `http://127.0.0.1:8501`.
- `DOCUSCOPE_AUDIT_FIXTURE_LIMIT`: files loaded from each upload fixture corpus;
  defaults to `5`.
- `DOCUSCOPE_AUDIT_TIMEOUT_MS`: timeout for an interaction or expected result;
  defaults to `180000`.
- `DOCUSCOPE_AUDIT_HEADED=1`: run Chromium with a visible window.
- `LOAD_TEST_TARGET_CORPUS` and `LOAD_TEST_REFERENCE_CORPUS`: internal corpora
  used by the analysis profile.
- `DOCUSCOPE_AUDIT_TARGET_CATEGORY` and
  `DOCUSCOPE_AUDIT_REFERENCE_CATEGORY`: corpus parts selected by the analysis
  profile; defaults to `BIO` and `ENG`.

The audit does not submit AI prompts, mutate admin settings, or trigger browser
downloads. It validates the AI data-selection surface and download preparation
controls without requiring credentials or writing downloaded artifacts.

## Shared-cache audit

The shared-cache audit uses two independent browser contexts as two users. Both
load the same bundled target and reference corpora, then request identical
keyness, corpus-parts keyness, n-gram, and collocation tables:

```bash
npm run test:audit:shared-cache
```

For each table, user 1 must visibly enter background generation. User 2 must
avoid that queue path, render the table at least 500 ms sooner, and take no more
than 70% of user 1's time. The JSON report records both durations, time saved,
ratio, queue observations, and assertion outcomes. It is written to
`reports/shared-cache-audit-*.json`.

If user 1 does not enter the queue, the manifest was already warm. That scenario
is reported as `inconclusive` and the command exits nonzero rather than claiming
a cold-to-warm cache result. Use an unused parameter combination or a clean
artifact store for a conclusive rerun.

Shared artifacts are restricted to bundled corpora. Uploaded or mixed sources
remain session-local; `tests/unit/persistence/test_registry.py` verifies that
their identities are rejected, while `npm run test:audit:uploads` exercises the
uploaded-corpus browser path.

Shared-cache environment variables:

- `DOCUSCOPE_CACHE_AUDIT_TIMEOUT_MS`: per-step timeout; defaults to `180000`.
- `DOCUSCOPE_CACHE_AUDIT_MAX_WARM_RATIO`: largest accepted user-2/user-1 ratio;
  defaults to `0.7`.
- `DOCUSCOPE_CACHE_AUDIT_MIN_SAVED_MS`: minimum required time saved; defaults to
  `500`.
- `DOCUSCOPE_CACHE_AUDIT_TARGET_CATEGORY` and
  `DOCUSCOPE_CACHE_AUDIT_REFERENCE_CATEGORY`: corpus parts; defaults to `BIO`
  and `HIS`.
- `DOCUSCOPE_CACHE_AUDIT_NGRAM_SPAN`: n-gram span; defaults to `4`.
- `DOCUSCOPE_CACHE_AUDIT_COLLOCATION_NODE`: collocation node; defaults to
  `however`.
- `LOAD_TEST_TARGET_CORPUS` and `LOAD_TEST_REFERENCE_CORPUS`: bundled corpora;
  defaults to `A_MICUSP_mini` and `C_BAWE_mini`.
