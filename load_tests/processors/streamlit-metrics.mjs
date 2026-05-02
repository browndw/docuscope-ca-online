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

function containsName(text) {
  const escaped = text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  return new RegExp(escaped, 'i');
}

function debugEnabled() {
  return process.env.LOAD_TEST_DEBUG === '1';
}

async function debugLog(page, message) {
  if (!debugEnabled()) {
    return;
  }

  const title = await page.title().catch(() => 'unknown');
  console.log(`[load-tests] ${message} | url=${page.url()} | title=${title}`);
}

export async function recordPageMetrics(page) {
  const manageCorpusLink = page.getByRole('link', { name: /Manage Corpus Data/i }).first();
  const heading = page.getByRole('heading', { name: 'DocuScope', exact: true }).first();

  const landingReady = await Promise.any([
    heading.waitFor({ state: 'visible', timeout: 60000 }).then(() => 'heading'),
    manageCorpusLink.waitFor({ state: 'visible', timeout: 60000 }).then(() => 'link')
  ]).catch(() => null);

  if (!landingReady) {
    throw new Error('Landing page did not become ready within 60s');
  }
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

async function chooseComboboxOption(page, label, optionText, timeout = 60000, index = 0) {
  const combo = page.getByRole('combobox', { name: containsName(label) }).nth(index);
  await combo.waitFor({ state: 'visible', timeout });
  await combo.click();
  const option = page.getByRole('option', { name: containsName(optionText) }).first();
  await option.waitFor({ state: 'visible', timeout });
  await option.click();
}

async function expandNavigation(page) {
  const navText = page.getByText('Navigation', { exact: true }).first();
  if (await navText.isVisible().catch(() => false)) {
    await navText.click();
  }
}

async function goToManageCorpus(page) {
  const pageLink = page.getByRole('link', { name: containsName('Manage Corpus Data') }).first();
  if (await pageLink.isVisible().catch(() => false)) {
    await pageLink.click();
    await page.getByRole('heading', { name: 'Manage Corpus Data', exact: true }).first().waitFor({ state: 'visible', timeout: 60000 });
    return;
  }

  await expandNavigation(page);
  const navLink = page.getByRole('link', { name: containsName('Manage Corpus Data') }).first();
  await navLink.waitFor({ state: 'visible', timeout: 60000 });
  await navLink.click();
  await page.getByRole('heading', { name: 'Manage Corpus Data', exact: true }).first().waitFor({ state: 'visible', timeout: 60000 });
}

async function goToCompareCorpora(page) {
  await expandNavigation(page);
  const navLink = page.getByRole('link', { name: containsName('Compare Corpora') }).first();
  await navLink.waitFor({ state: 'visible', timeout: 60000 });
  await navLink.click();
  await page.getByRole('heading', { name: 'Compare Corpora', exact: true }).first().waitFor({ state: 'visible', timeout: 60000 });
}

async function goToTokenFrequencies(page) {
  await expandNavigation(page);
  const navLink = page.getByRole('link', { name: containsName('Token Frequencies') }).first();
  await navLink.waitFor({ state: 'visible', timeout: 60000 });
  await navLink.click();
  await page.getByRole('heading', { name: 'Token Frequencies', exact: true }).first().waitFor({ state: 'visible', timeout: 60000 });
}

async function loadInternalTarget(page, targetCorpus, dictionaryLabel) {
  await page.getByRole('heading', { name: 'Manage Corpus Data', exact: true }).first().waitFor({ state: 'visible', timeout: 60000 });
  const targetSourceGroup = page.getByRole('radiogroup', { name: containsName('What kind of corpus would you like to prepare?') }).first();
  await targetSourceGroup.waitFor({ state: 'visible', timeout: 60000 });
  await targetSourceGroup.getByText('Internal', { exact: true }).click();
  await chooseComboboxOption(page, 'Select a saved corpus to load:', targetCorpus);
  await clickButton(page, 'Process Target', 60000);
  await page.getByText('Reference corpus:', { exact: true }).first().waitFor({ state: 'visible', timeout: 60000 });
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
  await page.getByRole('heading', { name: 'Compare Corpora', exact: true }).first().waitFor({ state: 'visible', timeout: 30000 });
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
  await page.getByRole('heading', { name: 'Token Frequencies', exact: true }).first().waitFor({ state: 'visible', timeout: 60000 });

  const generationButton = page.getByRole('button', { name: containsName('Frequency Table') }).first();
  await generationButton.waitFor({ state: 'visible', timeout: 60000 });
  await generationButton.click();

  const generatedReady = await Promise.any([
    page.getByRole('radio', { name: 'Parts-of-Speech', exact: true }).first().waitFor({ state: 'visible', timeout: 120000 }).then(() => 'pos-radio'),
    page.getByRole('radio', { name: 'DocuScope', exact: true }).first().waitFor({ state: 'visible', timeout: 120000 }).then(() => 'docuscope-radio'),
    page.getByText(/No frequency data available to display/i).first().waitFor({ state: 'visible', timeout: 120000 }).then(() => 'empty-state')
  ]).catch(() => null);

  if (!generatedReady) {
    throw new Error('Token frequency results did not become visible within 120s');
  }

  await debugLog(page, `token-frequency-ready=${generatedReady}`);
}

export async function internalKeynessScenario(page, vuContext, events) {
  const baseUrl = vuContext?.vars?.baseUrl || process.env.LOAD_TEST_BASE_URL || 'http://127.0.0.1:8501';
  const targetCorpus = vuContext?.vars?.targetCorpus || 'A_MICUSP_mini';
  const referenceCorpus = vuContext?.vars?.referenceCorpus || 'C_BAWE_mini';
  const dictionaryLabel = vuContext?.vars?.dictionaryLabel || 'Large Dictionary';

  await page.goto(baseUrl, { waitUntil: 'domcontentloaded', timeout: 60000 });
  await page.waitForLoadState('networkidle', { timeout: 60000 }).catch(() => null);
  await recordPageMetrics(page);
  await goToManageCorpus(page);
  await loadInternalTarget(page, targetCorpus, dictionaryLabel);
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

  await page.goto(baseUrl, { waitUntil: 'domcontentloaded', timeout: 60000 });
  await page.waitForLoadState('networkidle', { timeout: 60000 }).catch(() => null);
  await recordPageMetrics(page);
  await goToManageCorpus(page);
  await loadInternalTarget(page, targetCorpus, dictionaryLabel);
  await loadInternalReference(page, referenceCorpus);
  await goToCompareCorpora(page);
  await waitForCompareCorporaReady(page);

  events.emit('counter', 'streamlit.compare.ready', 1);
}

export async function tokenFrequencyScenario(page, vuContext, events) {
  const baseUrl = vuContext?.vars?.baseUrl || process.env.LOAD_TEST_BASE_URL || 'http://127.0.0.1:8501';
  const targetCorpus = vuContext?.vars?.targetCorpus || 'A_MICUSP_mini';
  const dictionaryLabel = vuContext?.vars?.dictionaryLabel || 'Large Dictionary';

  await page.goto(baseUrl, { waitUntil: 'domcontentloaded', timeout: 60000 });
  await page.waitForLoadState('networkidle', { timeout: 60000 }).catch(() => null);
  await recordPageMetrics(page);
  await goToManageCorpus(page);
  await loadInternalTarget(page, targetCorpus, dictionaryLabel);
  await goToTokenFrequencies(page);
  await generateTokenFrequencyTable(page);

  events.emit('counter', 'streamlit.token_frequency.generated', 1);
}