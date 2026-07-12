export function beforeScenario(context, events, done) {
  context.vars.startedAt = Date.now();
  context.vars.baseUrl = process.env.LOAD_TEST_BASE_URL || 'http://127.0.0.1:8501';
  context.vars.targetCorpus = process.env.LOAD_TEST_TARGET_CORPUS || 'A_MICUSP_mini';
  context.vars.referenceCorpus = process.env.LOAD_TEST_REFERENCE_CORPUS || 'C_BAWE_mini';
  context.vars.dictionaryLabel = process.env.LOAD_TEST_DICTIONARY_LABEL || 'Large Dictionary';
  return done();
}

export function afterScenario(context, events, done) {
  const startedAt = context.vars.startedAt || Date.now();
  const elapsedMs = Date.now() - startedAt;
  events.emit('histogram', 'scenario.duration_ms', elapsedMs);
  events.emit('counter', 'scenario.completed', 1);
  return done();
}

async function recordStep(events, metricBaseName, step, action) {
  const startedAt = Date.now();
  try {
    const result = await action();
    const elapsedMs = Date.now() - startedAt;
    events.emit('histogram', `${metricBaseName}.${step}.duration_ms`, elapsedMs);
    events.emit('counter', `${metricBaseName}.${step}.completed`, 1);
    return result;
  } catch (error) {
    const elapsedMs = Date.now() - startedAt;
    events.emit('histogram', `${metricBaseName}.${step}.duration_ms`, elapsedMs);
    events.emit('counter', `${metricBaseName}.${step}.failed`, 1);
    throw error;
  }
}

function emitStepMetric(events, metricName, elapsedMs) {
  if (!events) {
    return;
  }

  events.emit('histogram', `${metricName}.duration_ms`, elapsedMs);
  events.emit('counter', `${metricName}.completed`, 1);
}

function containsName(text) {
  const escaped = text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  return new RegExp(escaped, 'i');
}

function normalizePathname(pathname) {
  if (!pathname) {
    return '/';
  }

  return pathname.endsWith('/') && pathname.length > 1
    ? pathname.slice(0, -1)
    : pathname;
}

async function waitForPathname(page, expectedPathname, timeout = 60000) {
  const normalizedExpectedPathname = normalizePathname(expectedPathname);
  await page.waitForFunction(
    (pathname) => {
      const current = window.location.pathname;
      const normalizedCurrent = current.endsWith('/') && current.length > 1
        ? current.slice(0, -1)
        : current;
      return normalizedCurrent === pathname;
    },
    normalizedExpectedPathname,
    { timeout },
  );
}

function debugEnabled() {
  return process.env.LOAD_TEST_DEBUG === '1';
}

function forcedNavMode() {
  return (process.env.LOAD_TEST_NAV_MODE || '').trim().toLowerCase();
}

function shouldTrace404s() {
  return debugEnabled() || forcedNavMode() === 'goto';
}

function attachHttpDiagnostics(page) {
  if (!shouldTrace404s() || page.__loadTest404TracerAttached) {
    return;
  }

  page.__loadTest404TracerAttached = true;
  page.__loadTestSeen404Urls = new Set();

  page.on('response', async (response) => {
    if (response.status() !== 404) {
      return;
    }

    const seen404Urls = page.__loadTestSeen404Urls;
    const url = response.url();
    if (!(seen404Urls instanceof Set) || seen404Urls.has(url) || seen404Urls.size >= 10) {
      return;
    }

    seen404Urls.add(url);
    const request = response.request();
    const resourceType = request.resourceType();
    console.log(`[load-tests] 404 url=${url} resource=${resourceType} page=${page.url()}`);
  });
}

function processTargetProbeMode() {
  return (process.env.DOCUSCOPE_PROCESS_TARGET_PROBE || 'full').trim().toLowerCase();
}

function emitRecordedStageTiming(events, stageName, elapsedMs) {
  if (!events || elapsedMs === null || elapsedMs === undefined) {
    return;
  }

  events.emit(
    'histogram',
    `streamlit.token_frequency_step.load_target_callback_${stageName}.duration_ms`,
    elapsedMs,
  );
  events.emit(
    'counter',
    `streamlit.token_frequency_step.load_target_callback_${stageName}.completed`,
    1,
  );
}

function isProcessTargetProbeMode() {
  return processTargetProbeMode() !== 'full';
}

async function debugLog(page, message) {
  if (!debugEnabled()) {
    return;
  }

  const title = await page.title().catch(() => 'unknown');
  console.log(`[load-tests] ${message} | url=${page.url()} | title=${title}`);
}

export async function recordPageMetrics(page, events = null) {
  const manageCorpusLink = page.getByRole('link', { name: /Manage Corpus Data/i }).first();
  const heading = page.getByRole('heading', { name: 'DocuScope', exact: true }).first();

  const landingShellStartedAt = Date.now();
  const landingReady = await Promise.any([
    heading.waitFor({ state: 'visible', timeout: 60000 }).then(() => 'heading'),
    manageCorpusLink.waitFor({ state: 'visible', timeout: 60000 }).then(() => 'link')
  ]).catch(() => null);

  if (!landingReady) {
    throw new Error('Landing page did not become ready within 60s');
  }

  emitStepMetric(
    events,
    'streamlit.token_frequency_step.landing_shell',
    Date.now() - landingShellStartedAt,
  );

  const landingControlsStartedAt = Date.now();
  await manageCorpusLink.waitFor({ state: 'visible', timeout: 60000 });
  emitStepMetric(
    events,
    'streamlit.token_frequency_step.landing_controls',
    Date.now() - landingControlsStartedAt,
  );
}

export async function visitHealthEndpoint(page) {
  const baseUrl = process.env.LOAD_TEST_BASE_URL || 'http://127.0.0.1:8501';
  const response = await page.request.get(`${baseUrl}/_stcore/health`);
  if (!response.ok()) {
    throw new Error(`Health endpoint returned ${response.status()}`);
  }

  const body = await response.text();
  if (!body.includes('ok')) {
    throw new Error(`Unexpected health response body: ${body}`);
  }
}

export async function homePageSmoke(page, vuContext, events) {
  const baseUrl = vuContext?.vars?.baseUrl || process.env.LOAD_TEST_BASE_URL || 'http://127.0.0.1:8501';
  attachHttpDiagnostics(page);
  await page.goto(baseUrl, { waitUntil: 'domcontentloaded', timeout: 30000 });
  await page.waitForLoadState('networkidle', { timeout: 30000 });
  await recordPageMetrics(page);
  await visitHealthEndpoint(page);
  events.emit('counter', 'streamlit.home.loaded', 1);
}

async function clickText(page, text, timeout = 30000) {
  const locator = page.getByText(containsName(text)).first();
  await locator.waitFor({ state: 'visible', timeout });
  await locator.click();
}

async function clickButton(page, name, timeout = 30000) {
  const locator = page.getByRole('button', { name: containsName(name) }).first();
  await locator.waitFor({ state: 'visible', timeout });
  await locator.click();
}

async function clickRadio(page, name, timeout = 30000) {
  const locator = page.getByRole('radio', { name: containsName(name) }).first();
  await locator.waitFor({ state: 'visible', timeout });
  await locator.click();
}

function classroomThinkMs(step, fallbackMs) {
  const envKey = `LOAD_TEST_CLASSROOM_${step.toUpperCase()}_THINK_MS`;
  const parsed = Number.parseInt(process.env[envKey] || '', 10);
  return Number.isFinite(parsed) && parsed >= 0 ? parsed : fallbackMs;
}

async function classroomThink(page, events, step, fallbackMs, jitterMs = 2000) {
  const baseMs = classroomThinkMs(step, fallbackMs);
  const jitter = jitterMs > 0 ? Math.floor(Math.random() * jitterMs) : 0;
  const elapsedMs = baseMs + jitter;
  await page.waitForTimeout(elapsedMs);
  emitStepMetric(events, `streamlit.classroom_token_frequency_step.think_${step}`, elapsedMs);
}

async function chooseComboboxOption(page, label, optionText, timeout = 60000, index = 0) {
  const combo = page.getByRole('combobox', { name: containsName(label) }).nth(index);
  await combo.waitFor({ state: 'visible', timeout });
  await combo.click();
  const option = page.getByRole('option', { name: containsName(optionText) }).first();
  await option.waitFor({ state: 'visible', timeout });
  await option.click();
}

async function expandNavigation(page) {
  const sidebarToggle = page.getByRole('button', { name: containsName('keyboard_double_arrow_right') }).first();
  if (await sidebarToggle.isVisible().catch(() => false)) {
    await sidebarToggle.click({ force: true }).catch(() => null);
    await page.waitForTimeout(500);
  }

  const navText = page.getByText('Navigation', { exact: true }).first();
  if (await navText.isVisible().catch(() => false)) {
    await navText.click({ force: true });
  }
}

async function clickSidebarPageLink(page, label, timeout = 60000) {
  const navLink = page.getByRole('link', { name: containsName(label) }).first();
  await navLink.waitFor({ state: 'visible', timeout });

  const mode = forcedNavMode();
  if (mode === 'goto') {
    const href = await navLink.getAttribute('href');
    if (!href) {
      throw new Error(`Sidebar link ${label} did not expose an href for forced goto navigation`);
    }

    const targetUrl = new URL(href, page.url()).toString();
    await page.goto(targetUrl, { waitUntil: 'domcontentloaded', timeout });
    return 'forced-goto';
  }

  if (mode === 'dom-click') {
    await navLink.evaluate((element) => {
      element.click();
    });
    return 'dom-click';
  }

  try {
    await navLink.click({ timeout: 10000 });
    return 'click';
  } catch (error) {
    await debugLog(page, `sidebar-link-force-fallback label=${label} error=${error?.name || 'unknown'}`);
  }

  try {
    await navLink.click({ force: true, timeout: 10000 });
    return 'force-click';
  } catch (error) {
    await debugLog(page, `sidebar-link-goto-fallback label=${label} error=${error?.name || 'unknown'}`);
  }

  const href = await navLink.getAttribute('href');
  if (!href) {
    throw new Error(`Sidebar link ${label} did not expose an href for fallback navigation`);
  }

  const targetUrl = new URL(href, page.url()).toString();
  await page.goto(targetUrl, { waitUntil: 'domcontentloaded', timeout });
  return 'goto';
}

async function goToManageCorpus(page, events = null) {
  const navStartedAt = Date.now();
  const pageLink = page.getByRole('link', { name: containsName('Manage Corpus Data') }).first();
  if (await pageLink.isVisible().catch(() => false)) {
    await pageLink.click();
    const navElapsedMs = Date.now() - navStartedAt;
    events?.emit('histogram', 'streamlit.token_frequency_step.go_manage_corpus_nav.duration_ms', navElapsedMs);
    events?.emit('counter', 'streamlit.token_frequency_step.go_manage_corpus_nav.completed', 1);

    const pathStartedAt = Date.now();
    await waitForPathname(page, '/load_corpus');
    emitStepMetric(
      events,
      'streamlit.token_frequency_step.go_manage_corpus_path',
      Date.now() - pathStartedAt,
    );

    const headingStartedAt = Date.now();
    await page.getByRole('heading', { name: 'Manage Corpus Data', exact: true }).first().waitFor({ state: 'visible', timeout: 60000 });
    emitStepMetric(
      events,
      'streamlit.token_frequency_step.go_manage_corpus_heading',
      Date.now() - headingStartedAt,
    );

    const readyStartedAt = Date.now();
    await page.getByRole('radiogroup', { name: containsName('What kind of corpus would you like to prepare?') }).first().waitFor({ state: 'visible', timeout: 60000 });
    const readyElapsedMs = Date.now() - readyStartedAt;
    events?.emit('histogram', 'streamlit.token_frequency_step.go_manage_corpus_ready.duration_ms', readyElapsedMs);
    events?.emit('counter', 'streamlit.token_frequency_step.go_manage_corpus_ready.completed', 1);
    return;
  }

  await expandNavigation(page);
  await clickSidebarPageLink(page, 'Manage Corpus Data');
  const navElapsedMs = Date.now() - navStartedAt;
  events?.emit('histogram', 'streamlit.token_frequency_step.go_manage_corpus_nav.duration_ms', navElapsedMs);
  events?.emit('counter', 'streamlit.token_frequency_step.go_manage_corpus_nav.completed', 1);

  const pathStartedAt = Date.now();
  await waitForPathname(page, '/load_corpus');
  emitStepMetric(
    events,
    'streamlit.token_frequency_step.go_manage_corpus_path',
    Date.now() - pathStartedAt,
  );

  const headingStartedAt = Date.now();
  await page.getByRole('heading', { name: 'Manage Corpus Data', exact: true }).first().waitFor({ state: 'visible', timeout: 60000 });
  emitStepMetric(
    events,
    'streamlit.token_frequency_step.go_manage_corpus_heading',
    Date.now() - headingStartedAt,
  );

  const readyStartedAt = Date.now();
  await page.getByRole('radiogroup', { name: containsName('What kind of corpus would you like to prepare?') }).first().waitFor({ state: 'visible', timeout: 60000 });
  const readyElapsedMs = Date.now() - readyStartedAt;
  events?.emit('histogram', 'streamlit.token_frequency_step.go_manage_corpus_ready.duration_ms', readyElapsedMs);
  events?.emit('counter', 'streamlit.token_frequency_step.go_manage_corpus_ready.completed', 1);
}

async function goToCompareCorpora(page) {
  await expandNavigation(page);
  const navMethod = await clickSidebarPageLink(page, 'Compare Corpora');
  await debugLog(page, `compare-corpora-nav-clicked method=${navMethod}`);
  await waitForPathname(page, '/compare_corpora').catch(async (error) => {
    const bodyText = await page.locator('body').innerText().catch(() => '');
    await debugLog(page, `compare-corpora-path-timeout body=${bodyText.slice(0, 500).replace(/\s+/g, ' ')}`);
    throw error;
  });
  await debugLog(page, 'compare-corpora-path-ready');
  const readySignal = await Promise.any([
    page.getByRole('heading', { name: containsName('Compare Corpora') }).first().waitFor({ state: 'visible', timeout: 60000 }).then(() => 'heading'),
    page.getByRole('button', { name: containsName('Keyness Table') }).first().waitFor({ state: 'visible', timeout: 60000 }).then(() => 'keyness-button'),
    page.getByText(/Use the button in the sidebar to generate keywords/i).first().waitFor({ state: 'visible', timeout: 60000 }).then(() => 'instructions')
  ]).catch(async (error) => {
    const bodyText = await page.locator('body').innerText().catch(() => '');
    await debugLog(page, `compare-corpora-ready-timeout body=${bodyText.slice(0, 500).replace(/\s+/g, ' ')}`);
    throw error;
  });
  await debugLog(page, `compare-corpora-ready-signal=${readySignal}`);
}

async function goToTokenFrequencies(page, events = null) {
  await expandNavigation(page);
  const navStartedAt = Date.now();
  const navMethod = await clickSidebarPageLink(page, 'Token Frequencies');
  const navElapsedMs = Date.now() - navStartedAt;
  events?.emit('histogram', 'streamlit.token_frequency_step.go_token_frequencies_nav.duration_ms', navElapsedMs);
  events?.emit('counter', 'streamlit.token_frequency_step.go_token_frequencies_nav.completed', 1);

  const pathStartedAt = Date.now();
  await waitForPathname(page, '/token_frequencies');
  emitStepMetric(
    events,
    'streamlit.token_frequency_step.go_token_frequencies_path',
    Date.now() - pathStartedAt,
  );

  const heading = page.getByRole('heading', { name: 'Token Frequencies', exact: true }).first();

  const headingAttachedStartedAt = Date.now();
  void heading.waitFor({ state: 'attached', timeout: 60000 }).then(() => {
    emitStepMetric(
      events,
      'streamlit.token_frequency_step.go_token_frequencies_heading_attached',
      Date.now() - headingAttachedStartedAt,
    );
  }).catch(() => null);

  const headingVisibleStartedAt = Date.now();
  void heading.waitFor({ state: 'visible', timeout: 60000 }).then(() => {
    emitStepMetric(
      events,
      'streamlit.token_frequency_step.go_token_frequencies_heading_visible',
      Date.now() - headingVisibleStartedAt,
    );
  }).catch(() => null);

  const readyStartedAt = Date.now();
  const readySignal = await Promise.any([
    page.getByRole('button', { name: containsName('Frequency Table') }).first().waitFor({ state: 'visible', timeout: 60000 }).then(() => 'frequency-button'),
    page.getByText(/No target corpus loaded\./i).first().waitFor({ state: 'visible', timeout: 60000 }).then(() => 'no-target-warning'),
    page.getByText(/Could not load target corpus metadata\./i).first().waitFor({ state: 'visible', timeout: 60000 }).then(() => 'metadata-warning')
  ]).catch(() => null);

  if (!readySignal) {
    throw new Error('Token Frequencies page did not become ready within 60s');
  }

  const readyElapsedMs = Date.now() - readyStartedAt;
  events?.emit('histogram', 'streamlit.token_frequency_step.go_token_frequencies_ready.duration_ms', readyElapsedMs);
  events?.emit('counter', 'streamlit.token_frequency_step.go_token_frequencies_ready.completed', 1);

  await debugLog(page, `token-frequencies-page-ready-signal=${readySignal} nav=${navMethod}`);
}

async function goToTagFrequencies(page) {
  await expandNavigation(page);
  const navMethod = await clickSidebarPageLink(page, 'Tag Frequencies');

  const readySignal = await Promise.any([
    page.getByRole('heading', { name: 'Tag Frequencies', exact: true }).first().waitFor({ state: 'attached', timeout: 60000 }).then(() => 'heading-attached'),
    page.getByRole('button', { name: containsName('Tags Table') }).first().waitFor({ state: 'visible', timeout: 60000 }).then(() => 'tags-button'),
    page.getByText(/Invalid session state\./i).first().waitFor({ state: 'visible', timeout: 60000 }).then(() => 'invalid-session-error'),
    page.getByText(/Could not load target corpus metadata\./i).first().waitFor({ state: 'visible', timeout: 60000 }).then(() => 'metadata-error')
  ]).catch(() => null);

  if (!readySignal) {
    throw new Error('Tag Frequencies page did not become ready within 60s');
  }

  await debugLog(page, `tag-frequencies-page-ready-signal=${readySignal} nav=${navMethod}`);
}

async function loadInternalTarget(page, targetCorpus, dictionaryLabel, events) {
  await page.getByRole('heading', { name: 'Manage Corpus Data', exact: true }).first().waitFor({ state: 'visible', timeout: 60000 });

  const setupStartedAt = Date.now();
  const targetSourceGroup = page.getByRole('radiogroup', { name: containsName('What kind of corpus would you like to prepare?') }).first();
  await targetSourceGroup.waitFor({ state: 'visible', timeout: 60000 });
  await targetSourceGroup.getByText('Internal', { exact: true }).click();
  await chooseComboboxOption(page, 'Select a saved corpus to load:', targetCorpus);
  const setupElapsedMs = Date.now() - setupStartedAt;

  const triggerStartedAt = Date.now();
  const processTargetButton = page.getByRole('button', { name: containsName('Process Target') }).first();
  const processTargetButtonHiddenPromise = processTargetButton
    .waitFor({ state: 'hidden', timeout: 60000 })
    .then(() => Date.now() - triggerStartedAt)
    .catch(() => null);
  await clickButton(page, 'Process Target', 60000);
  const clickDispatchedElapsedMs = Date.now() - triggerStartedAt;
  events.emit('histogram', 'streamlit.token_frequency_step.load_target_click_dispatch.duration_ms', clickDispatchedElapsedMs);
  events.emit('counter', 'streamlit.token_frequency_step.load_target_click_dispatch.completed', 1);
  const processTargetButtonHiddenElapsedMs = await processTargetButtonHiddenPromise;
  if (processTargetButtonHiddenElapsedMs !== null) {
    events.emit('histogram', 'streamlit.token_frequency_step.load_target_button_hidden.duration_ms', processTargetButtonHiddenElapsedMs);
    events.emit('counter', 'streamlit.token_frequency_step.load_target_button_hidden.completed', 1);
  }
  events.emit('histogram', 'streamlit.token_frequency_step.load_target_setup.duration_ms', setupElapsedMs);
  events.emit('counter', 'streamlit.token_frequency_step.load_target_setup.completed', 1);

  if (isProcessTargetProbeMode()) {
    const probeMode = processTargetProbeMode();
    if (probeMode === 'split_ready') {
      await page.getByText('LOAD_TEST_PROCESS_TARGET_RERUN_STARTED:target:split_ready', { exact: true }).first().waitFor({ state: 'attached', timeout: 60000 });
      const rerunStartedElapsedMs = Date.now() - triggerStartedAt;
      events.emit('histogram', 'streamlit.token_frequency_step.load_target_rerun_started.duration_ms', rerunStartedElapsedMs);
      events.emit('counter', 'streamlit.token_frequency_step.load_target_rerun_started.completed', 1);
      const postClickToRerunStartedMs = rerunStartedElapsedMs - clickDispatchedElapsedMs;
      events.emit('histogram', 'streamlit.token_frequency_step.load_target_post_click_to_rerun_started.duration_ms', postClickToRerunStartedMs);
      events.emit('counter', 'streamlit.token_frequency_step.load_target_post_click_to_rerun_started.completed', 1);

      await page.getByText('LOAD_TEST_PROCESS_TARGET_CALLBACK_FINISHED:target:split_ready', { exact: true }).first().waitFor({ state: 'attached', timeout: 60000 });
      const rerunEntryElapsedMs = Date.now() - triggerStartedAt;
      events.emit('histogram', 'streamlit.token_frequency_step.load_target_rerun_entry.duration_ms', rerunEntryElapsedMs);
      events.emit('counter', 'streamlit.token_frequency_step.load_target_rerun_entry.completed', 1);

      const callbackText = await page.locator('body').innerText().catch(() => '');
      const emittedStageTimings = new Set();
      const summaryMatch = callbackText.match(/LOAD_TEST_PROCESS_TARGET_CALLBACK_STAGE_SUMMARY:target:([^\n]+)/);
      if (summaryMatch) {
        for (const pair of summaryMatch[1].split(';')) {
          const [stageName, elapsedRaw] = pair.split('=');
          const elapsedMs = Number.parseFloat(elapsedRaw);
          if (stageName && Number.isFinite(elapsedMs) && !emittedStageTimings.has(stageName)) {
            emitRecordedStageTiming(events, stageName, elapsedMs);
            emittedStageTimings.add(stageName);
          }
        }
      }

      const callbackStagePattern = /LOAD_TEST_PROCESS_TARGET_CALLBACK_STAGE_MS:target:([^:\s]+):([0-9]+(?:\.[0-9]+)?)/g;
      for (const match of callbackText.matchAll(callbackStagePattern)) {
        const stageName = match[1];
        const elapsedMs = Number.parseFloat(match[2]);
        if (Number.isFinite(elapsedMs) && !emittedStageTimings.has(stageName)) {
          emitRecordedStageTiming(events, stageName, elapsedMs);
          emittedStageTimings.add(stageName);
        }
      }

      await page.getByText('LOAD_TEST_PROCESS_TARGET_READY_BRANCH:target:split_ready', { exact: true }).first().waitFor({ state: 'attached', timeout: 60000 });
      const readyBranchElapsedMs = Date.now() - triggerStartedAt;
      events.emit('histogram', 'streamlit.token_frequency_step.load_target_ready_branch.duration_ms', readyBranchElapsedMs);
      events.emit('counter', 'streamlit.token_frequency_step.load_target_ready_branch.completed', 1);

      const metadataLoadedPromise = page
        .getByText('LOAD_TEST_PROCESS_TARGET_SUBSTEP:target:metadata_loaded', { exact: true })
        .first()
        .waitFor({ state: 'attached', timeout: 60000 })
        .then(() => Date.now() - triggerStartedAt)
        .catch(() => null);

      const tabsCreatedPromise = page
        .getByText('LOAD_TEST_PROCESS_TARGET_SUBSTEP:target:tabs_created', { exact: true })
        .first()
        .waitFor({ state: 'attached', timeout: 60000 })
        .then(() => Date.now() - triggerStartedAt)
        .catch(() => null);

      const targetTabRenderedPromise = page
        .getByText('LOAD_TEST_PROCESS_TARGET_SUBSTEP:target:target_tab_rendered', { exact: true })
        .first()
        .waitFor({ state: 'attached', timeout: 60000 })
        .then(() => Date.now() - triggerStartedAt)
        .catch(() => null);

      const displayDonePromise = page
        .getByText('LOAD_TEST_PROCESS_TARGET_SUBSTEP:target:display_done', { exact: true })
        .first()
        .waitFor({ state: 'attached', timeout: 60000 })
        .then(() => Date.now() - triggerStartedAt)
        .catch(() => null);

      await page.getByText('Reference corpus:', { exact: true }).first().waitFor({ state: 'visible', timeout: 60000 });
      const readyElapsedMs = Date.now() - triggerStartedAt;

      const metadataLoadedElapsedMs = await metadataLoadedPromise;
      if (metadataLoadedElapsedMs !== null) {
        events.emit('histogram', 'streamlit.token_frequency_step.load_target_metadata_loaded.duration_ms', metadataLoadedElapsedMs);
        events.emit('counter', 'streamlit.token_frequency_step.load_target_metadata_loaded.completed', 1);
      }

      const tabsCreatedElapsedMs = await tabsCreatedPromise;
      if (tabsCreatedElapsedMs !== null) {
        events.emit('histogram', 'streamlit.token_frequency_step.load_target_tabs_created.duration_ms', tabsCreatedElapsedMs);
        events.emit('counter', 'streamlit.token_frequency_step.load_target_tabs_created.completed', 1);
      }

      const targetTabRenderedElapsedMs = await targetTabRenderedPromise;
      if (targetTabRenderedElapsedMs !== null) {
        events.emit('histogram', 'streamlit.token_frequency_step.load_target_target_tab_rendered.duration_ms', targetTabRenderedElapsedMs);
        events.emit('counter', 'streamlit.token_frequency_step.load_target_target_tab_rendered.completed', 1);
      }

      const displayDoneElapsedMs = await displayDonePromise;
      if (displayDoneElapsedMs !== null) {
        events.emit('histogram', 'streamlit.token_frequency_step.load_target_display_done.duration_ms', displayDoneElapsedMs);
        events.emit('counter', 'streamlit.token_frequency_step.load_target_display_done.completed', 1);
      }

      events.emit('histogram', 'streamlit.token_frequency_step.load_target_ready.duration_ms', readyElapsedMs);
      events.emit('counter', 'streamlit.token_frequency_step.load_target_ready.completed', 1);
      return;
    }

    await page.getByText(`LOAD_TEST_PROCESS_TARGET_READY:target:${probeMode}`, { exact: true }).first().waitFor({ state: 'attached', timeout: 60000 });
    const readyElapsedMs = Date.now() - triggerStartedAt;
    events.emit('histogram', 'streamlit.token_frequency_step.load_target_ready.duration_ms', readyElapsedMs);
    events.emit('counter', 'streamlit.token_frequency_step.load_target_ready.completed', 1);
    return;
  }

  await page.getByText('Reference corpus:', { exact: true }).first().waitFor({ state: 'visible', timeout: 60000 });
  const readyElapsedMs = Date.now() - triggerStartedAt;
  events.emit('histogram', 'streamlit.token_frequency_step.load_target_ready.duration_ms', readyElapsedMs);
  events.emit('counter', 'streamlit.token_frequency_step.load_target_ready.completed', 1);
}

async function loadInternalReference(page, referenceCorpus) {
  const loadReferenceGroup = page.getByRole('radiogroup', { name: containsName('Would you like to load a reference corpus?') }).first();
  await loadReferenceGroup.waitFor({ state: 'visible', timeout: 60000 });
  await loadReferenceGroup.getByText('Yes', { exact: true }).click();

  const referenceSourceGroup = page.getByRole('radiogroup', { name: containsName('What kind of reference corpus would you like to prepare?') }).first();
  await referenceSourceGroup.waitFor({ state: 'visible', timeout: 60000 });
  await referenceSourceGroup.getByText('Internal', { exact: true }).click();
  await chooseComboboxOption(page, 'Select a saved corpus to load:', referenceCorpus);
  await clickButton(page, 'Process Reference', 60000);
  await page.getByRole('tab', { name: containsName('Reference corpus') }).first().waitFor({ state: 'visible', timeout: 90000 });
}

async function generateKeyness(page) {
  await Promise.any([
    page.getByRole('heading', { name: containsName('Compare Corpora') }).first().waitFor({ state: 'visible', timeout: 30000 }),
    page.getByRole('button', { name: containsName('Keyness Table') }).first().waitFor({ state: 'visible', timeout: 30000 })
  ]);
  await debugLog(page, 'compare-corpora-mounted');

  const pvalBox = page.getByRole('combobox', { name: 'p-value threshold' }).first();
  const pvalVisible = await pvalBox.isVisible().catch(() => false);
  const pvalEnabled = pvalVisible ? await pvalBox.isEnabled().catch(() => false) : false;

  if (pvalVisible && pvalEnabled) {
    await pvalBox.click();
    const pvalOption = page.getByRole('option', { name: '0.05', exact: true }).first();
    if (await pvalOption.isVisible().catch(() => false)) {
      await pvalOption.click();
    } else {
      await page.keyboard.press('Escape');
    }
  }

  const keynessButton = page.getByRole('button', { name: containsName('Keyness Table') }).first();
  await keynessButton.waitFor({ state: 'visible', timeout: 30000 });

  const buttonEnabled = await keynessButton.isEnabled().catch(() => false);
  await debugLog(page, `keyness-button-ready enabled=${buttonEnabled}`);

  await keynessButton.click();
  await debugLog(page, 'keyness-button-clicked');

  const postClickSignal = await Promise.any([
    page.getByText(/Generating keywords/i).first().waitFor({ state: 'visible', timeout: 5000 }).then(() => 'generating-status'),
    page.getByRole('radio', { name: 'Tokens', exact: true }).first().waitFor({ state: 'visible', timeout: 5000 }).then(() => 'tokens-radio-fast'),
    page.getByRole('button', { name: 'Show/hide columns', exact: true }).first().waitFor({ state: 'visible', timeout: 5000 }).then(() => 'table-toolbar-fast')
  ]).catch(() => null);

  await debugLog(page, `post-click-signal=${postClickSignal || 'none'}`);

  const tokensRadio = page.getByRole('radio', { name: 'Tokens', exact: true }).first();
  const resetButton = page.getByRole('button', { name: containsName('Generate New Keyness Table') }).first();
  const keynessTab = page.getByRole('tab', { name: 'Keyness Table', exact: true }).first();
  const targetInfoHeading = page.getByRole('heading', { name: 'Target corpus information:', exact: true }).first();
  const tableToolbarButton = page.getByRole('button', { name: 'Show/hide columns', exact: true }).first();

  const generatedReady = await Promise.any([
    tokensRadio.waitFor({ state: 'visible', timeout: 120000 }).then(() => 'tokens-radio'),
    resetButton.waitFor({ state: 'visible', timeout: 120000 }).then(() => 'reset-button'),
    keynessTab.waitFor({ state: 'visible', timeout: 120000 }).then(() => 'keyness-tab'),
    targetInfoHeading.waitFor({ state: 'visible', timeout: 120000 }).then(() => 'target-info'),
    tableToolbarButton.waitFor({ state: 'visible', timeout: 120000 }).then(() => 'table-toolbar')
  ]).catch(() => null);

  await debugLog(page, `generated-ready=${generatedReady || 'none'}`);

  if (!generatedReady) {
    throw new Error('Keyness results did not become visible within 120s');
  }
}

async function waitForCompareCorporaReady(page) {
  await page.getByRole('heading', { name: 'Compare Corpora', exact: true }).first().waitFor({ state: 'visible', timeout: 60000 });

  const readySignal = await Promise.any([
    page.getByRole('button', { name: containsName('Keyness Table') }).first().waitFor({ state: 'visible', timeout: 60000 }).then(() => 'keyness-button'),
    page.getByRole('combobox', { name: 'p-value threshold' }).first().waitFor({ state: 'visible', timeout: 60000 }).then(() => 'pvalue-combobox'),
    page.getByRole('tab', { name: containsName('Target corpus') }).first().waitFor({ state: 'visible', timeout: 60000 }).then(() => 'target-corpus-tab')
  ]).catch(() => null);

  if (!readySignal) {
    throw new Error('Compare Corpora controls did not become visible within 60s');
  }

  await debugLog(page, `compare-ready-signal=${readySignal}`);
}

async function generateTokenFrequencyTable(page) {
  const generationButton = page.getByRole('button', { name: containsName('Frequency Table') }).first();
  await generationButton.waitFor({ state: 'visible', timeout: 60000 });
  await generationButton.click();

  const generatedReady = await Promise.any([
    page.getByRole('heading', { name: 'Target corpus information:', exact: true }).first().waitFor({ state: 'visible', timeout: 120000 }).then(() => 'target-info'),
    page.getByText(/Number of part-of-speech tokens in corpus:/i).first().waitFor({ state: 'visible', timeout: 120000 }).then(() => 'target-info-text'),
    page.getByText(/Download Options/i).first().waitFor({ state: 'visible', timeout: 120000 }).then(() => 'download-options'),
    page.getByText(/Download as Excel/i).first().waitFor({ state: 'visible', timeout: 120000 }).then(() => 'download-toggle'),
    page.getByText(/No frequency data available to display/i).first().waitFor({ state: 'visible', timeout: 120000 }).then(() => 'empty-state')
  ]).catch(() => null);

  if (!generatedReady) {
    throw new Error('Token frequency results did not become visible within 120s');
  }

  await debugLog(page, `token-frequency-ready=${generatedReady}`);
}

async function generateTagFrequencyTable(page) {
  const generationButton = page.getByRole('button', { name: containsName('Tags Table') }).first();
  await generationButton.waitFor({ state: 'visible', timeout: 60000 });
  await generationButton.click();

  const generatedReady = await Promise.any([
    page.getByRole('tab', { name: containsName('Table') }).first().waitFor({ state: 'visible', timeout: 120000 }).then(() => 'table-tab'),
    page.getByRole('tab', { name: containsName('Plot') }).first().waitFor({ state: 'visible', timeout: 120000 }).then(() => 'plot-tab'),
    page.getByText(/No frequency data available to display/i).first().waitFor({ state: 'visible', timeout: 120000 }).then(() => 'empty-state'),
    page.getByText(/No tags to plot\./i).first().waitFor({ state: 'visible', timeout: 120000 }).then(() => 'no-tags-warning')
  ]).catch(() => null);

  if (!generatedReady) {
    throw new Error('Tag frequency results did not become visible within 120s');
  }

  await debugLog(page, `tag-frequency-ready=${generatedReady}`);
}

export async function internalKeynessScenario(page, vuContext, events) {
  const baseUrl = vuContext?.vars?.baseUrl || process.env.LOAD_TEST_BASE_URL || 'http://127.0.0.1:8501';
  const targetCorpus = vuContext?.vars?.targetCorpus || 'A_MICUSP_mini';
  const referenceCorpus = vuContext?.vars?.referenceCorpus || 'C_BAWE_mini';
  const dictionaryLabel = vuContext?.vars?.dictionaryLabel || 'Large Dictionary';

  attachHttpDiagnostics(page);
  await page.goto(baseUrl, { waitUntil: 'domcontentloaded', timeout: 60000 });
  await page.waitForLoadState('networkidle', { timeout: 60000 }).catch(() => null);
  await recordPageMetrics(page, events);
  await goToManageCorpus(page, events);
  await loadInternalTarget(page, targetCorpus, dictionaryLabel, events);
  await loadInternalReference(page, referenceCorpus);
  await goToCompareCorpora(page);
  await generateKeyness(page);

  events.emit('counter', 'streamlit.keyness.generated', 1);
}

export async function internalCorpusLoadScenario(page, vuContext, events) {
  const baseUrl = vuContext?.vars?.baseUrl || process.env.LOAD_TEST_BASE_URL || 'http://127.0.0.1:8501';
  const targetCorpus = vuContext?.vars?.targetCorpus || 'A_MICUSP_mini';
  const referenceCorpus = vuContext?.vars?.referenceCorpus || 'C_BAWE_mini';
  const dictionaryLabel = vuContext?.vars?.dictionaryLabel || 'Large Dictionary';

  attachHttpDiagnostics(page);
  await page.goto(baseUrl, { waitUntil: 'domcontentloaded', timeout: 60000 });
  await page.waitForLoadState('networkidle', { timeout: 60000 }).catch(() => null);
  await recordPageMetrics(page, events);
  await goToManageCorpus(page, events);
  await loadInternalTarget(page, targetCorpus, dictionaryLabel, events);
  await loadInternalReference(page, referenceCorpus);
  await goToCompareCorpora(page);
  await waitForCompareCorporaReady(page);

  events.emit('counter', 'streamlit.compare.ready', 1);
}

export async function internalTargetReadyScenario(page, vuContext, events) {
  const baseUrl = vuContext?.vars?.baseUrl || process.env.LOAD_TEST_BASE_URL || 'http://127.0.0.1:8501';
  const targetCorpus = vuContext?.vars?.targetCorpus || 'A_MICUSP_mini';
  const dictionaryLabel = vuContext?.vars?.dictionaryLabel || 'Large Dictionary';

  attachHttpDiagnostics(page);
  await page.goto(baseUrl, { waitUntil: 'domcontentloaded', timeout: 60000 });
  await page.waitForLoadState('networkidle', { timeout: 60000 }).catch(() => null);
  await recordPageMetrics(page, events);
  await goToManageCorpus(page, events);
  await loadInternalTarget(page, targetCorpus, dictionaryLabel, events);

  events.emit('counter', 'streamlit.target.ready', 1);
}

export async function tokenFrequencyPageReadyScenario(page, vuContext, events) {
  const baseUrl = vuContext?.vars?.baseUrl || process.env.LOAD_TEST_BASE_URL || 'http://127.0.0.1:8501';
  const targetCorpus = vuContext?.vars?.targetCorpus || 'A_MICUSP_mini';
  const dictionaryLabel = vuContext?.vars?.dictionaryLabel || 'Large Dictionary';

  attachHttpDiagnostics(page);
  await page.goto(baseUrl, { waitUntil: 'domcontentloaded', timeout: 60000 });
  await page.waitForLoadState('networkidle', { timeout: 60000 }).catch(() => null);
  await recordPageMetrics(page, events);
  await goToManageCorpus(page, events);
  await loadInternalTarget(page, targetCorpus, dictionaryLabel, events);
  await goToTokenFrequencies(page, events);

  events.emit('counter', 'streamlit.token_frequency.page_ready', 1);
}

export async function tokenFrequencyScenario(page, vuContext, events) {
  const baseUrl = vuContext?.vars?.baseUrl || process.env.LOAD_TEST_BASE_URL || 'http://127.0.0.1:8501';
  const targetCorpus = vuContext?.vars?.targetCorpus || 'A_MICUSP_mini';
  const dictionaryLabel = vuContext?.vars?.dictionaryLabel || 'Large Dictionary';
  const metricBaseName = 'streamlit.token_frequency_step';

  attachHttpDiagnostics(page);
  await recordStep(events, metricBaseName, 'landing', async () => {
    await page.goto(baseUrl, { waitUntil: 'domcontentloaded', timeout: 60000 });
    await page.waitForLoadState('networkidle', { timeout: 60000 }).catch(() => null);
    await recordPageMetrics(page, events);
  });
  await recordStep(events, metricBaseName, 'go_manage_corpus', async () => {
    await goToManageCorpus(page, events);
  });
  await recordStep(events, metricBaseName, 'load_target', async () => {
    await loadInternalTarget(page, targetCorpus, dictionaryLabel, events);
  });
  await recordStep(events, metricBaseName, 'go_token_frequencies', async () => {
    await goToTokenFrequencies(page, events);
  });
  await recordStep(events, metricBaseName, 'generate_table', async () => {
    await generateTokenFrequencyTable(page);
  });

  events.emit('counter', 'streamlit.token_frequency.generated', 1);
}

export async function classroomTokenFrequencyScenario(page, vuContext, events) {
  const baseUrl = vuContext?.vars?.baseUrl || process.env.LOAD_TEST_BASE_URL || 'http://127.0.0.1:8501';
  const targetCorpus = vuContext?.vars?.targetCorpus || 'A_MICUSP_mini';
  const dictionaryLabel = vuContext?.vars?.dictionaryLabel || 'Large Dictionary';
  const metricBaseName = 'streamlit.classroom_token_frequency_step';

  attachHttpDiagnostics(page);
  await recordStep(events, metricBaseName, 'landing', async () => {
    await page.goto(baseUrl, { waitUntil: 'domcontentloaded', timeout: 60000 });
    await page.waitForLoadState('networkidle', { timeout: 60000 }).catch(() => null);
    await recordPageMetrics(page, events);
  });
  await classroomThink(page, events, 'after_landing', 5000, 3000);

  await recordStep(events, metricBaseName, 'go_manage_corpus', async () => {
    await goToManageCorpus(page, events);
  });
  await classroomThink(page, events, 'before_load_target', 8000, 5000);

  await recordStep(events, metricBaseName, 'load_target', async () => {
    await loadInternalTarget(page, targetCorpus, dictionaryLabel, events);
  });
  await classroomThink(page, events, 'after_target_ready', 10000, 5000);

  await recordStep(events, metricBaseName, 'go_token_frequencies', async () => {
    await goToTokenFrequencies(page, events);
  });
  await classroomThink(page, events, 'before_generate_table', 5000, 3000);

  await recordStep(events, metricBaseName, 'generate_table', async () => {
    await generateTokenFrequencyTable(page);
  });

  events.emit('counter', 'streamlit.classroom_token_frequency.generated', 1);
}

export async function tagFrequencyPageReadyScenario(page, vuContext, events) {
  const baseUrl = vuContext?.vars?.baseUrl || process.env.LOAD_TEST_BASE_URL || 'http://127.0.0.1:8501';
  const targetCorpus = vuContext?.vars?.targetCorpus || 'A_MICUSP_mini';
  const dictionaryLabel = vuContext?.vars?.dictionaryLabel || 'Large Dictionary';

  attachHttpDiagnostics(page);
  await page.goto(baseUrl, { waitUntil: 'domcontentloaded', timeout: 60000 });
  await page.waitForLoadState('networkidle', { timeout: 60000 }).catch(() => null);
  await recordPageMetrics(page, events);
  await goToManageCorpus(page, events);
  await loadInternalTarget(page, targetCorpus, dictionaryLabel, events);
  await goToTagFrequencies(page);

  events.emit('counter', 'streamlit.tag_frequency.page_ready', 1);
}

export async function tagFrequencyScenario(page, vuContext, events) {
  const baseUrl = vuContext?.vars?.baseUrl || process.env.LOAD_TEST_BASE_URL || 'http://127.0.0.1:8501';
  const targetCorpus = vuContext?.vars?.targetCorpus || 'A_MICUSP_mini';
  const dictionaryLabel = vuContext?.vars?.dictionaryLabel || 'Large Dictionary';

  await page.goto(baseUrl, { waitUntil: 'domcontentloaded', timeout: 60000 });
  await page.waitForLoadState('networkidle', { timeout: 60000 }).catch(() => null);
  await recordPageMetrics(page, events);
  await goToManageCorpus(page, events);
  await loadInternalTarget(page, targetCorpus, dictionaryLabel, events);
  await goToTagFrequencies(page);
  await generateTagFrequencyTable(page);

  events.emit('counter', 'streamlit.tag_frequency.generated', 1);
}