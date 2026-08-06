import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const loadTestsDir = path.resolve(__dirname, '..');
const reportsDir = path.join(loadTestsDir, 'reports');
const artifactsDir = path.join(reportsDir, 'page-audit-artifacts');

const baseUrl = process.env.LOAD_TEST_BASE_URL || 'http://127.0.0.1:8501';
const fixtureLimit = parsePositiveInteger(process.env.DOCUSCOPE_AUDIT_FIXTURE_LIMIT, 5);
const timeoutMs = parsePositiveInteger(process.env.DOCUSCOPE_AUDIT_TIMEOUT_MS, 180000);
const headed = process.env.DOCUSCOPE_AUDIT_HEADED === '1';
const profile = (process.env.DOCUSCOPE_AUDIT_PROFILE || 'uploads').trim().toLowerCase();

const corePages = [
  { label: 'Main Page', ready: /DocuScope/i, action: exerciseLanding },
  { label: 'Manage Corpus Data', ready: /Manage Corpus Data/i, action: exerciseManageCorpus },
  { label: 'Token Frequencies', ready: /Token Frequencies/i, action: generateTokenFrequencies },
  { label: 'Tag Frequencies', ready: /Tag Frequencies/i, action: generateTagFrequencies },
  { label: 'Ngrams & Clusters', ready: /N-gram and Cluster Frequency/i, action: generateNgrams },
  { label: 'Compare Corpora', ready: /Compare Corpora/i, action: generateKeyness },
  { label: 'Compare Corpus Parts', ready: /Compare Corpus Parts/i, action: exerciseCorpusParts },
  { label: 'Collocations', ready: /Collocates/i, action: generateCollocations },
  { label: 'Key Words in Context', ready: /KWIC Tables/i, action: generateKwic },
  { label: 'Advanced Plotting', ready: /Advanced Plotting/i, action: exerciseAdvancedPlotting },
  { label: 'Matrix Explorer', ready: /Matrix Explorer/i, action: loadMatrixTables },
  { label: 'Single Document', ready: /Single Documents/i, action: processDocument },
  { label: 'AI-Asissted Plotting', ready: /AI-Assisted Plotting/i, action: exerciseAiPlotting },
  { label: 'Download Corpus Data', ready: /Download Corpus Files/i, action: exerciseCorpusDownload },
  { label: 'Download Tagged Files', ready: /Download Tagged Files/i, action: exerciseTaggedDownload },
];

const report = {
  startedAt: new Date().toISOString(),
  baseUrl,
  profile,
  fixtureLimit,
  setupChecks: [],
  pages: [],
  runtimeErrors: [],
  consoleErrors: [],
  requestFailures: [],
  httpErrors: [],
};

let currentStep = 'startup';

function parsePositiveInteger(value, fallback) {
  const parsed = Number.parseInt(value || '', 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

function messageOf(error) {
  return error instanceof Error ? error.stack || error.message : String(error);
}

function slugify(value) {
  return value.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');
}

async function listFixtures(name) {
  const directory = path.join(loadTestsDir, 'test_data', name);
  const entries = await fs.readdir(directory);
  return entries
    .filter((entry) => entry.toLowerCase().endsWith('.txt'))
    .sort()
    .slice(0, fixtureLimit)
    .map((entry) => path.join(directory, entry));
}

function attachDiagnostics(page) {
  page.on('pageerror', (error) => {
    report.runtimeErrors.push({ step: currentStep, url: page.url(), message: messageOf(error) });
  });

  page.on('console', (message) => {
    if (message.type() !== 'error') {
      return;
    }
    report.consoleErrors.push({ step: currentStep, url: page.url(), message: message.text() });
  });

  page.on('requestfailed', (request) => {
    report.requestFailures.push({
      step: currentStep,
      url: request.url(),
      method: request.method(),
      resourceType: request.resourceType(),
      error: request.failure()?.errorText || 'unknown request failure',
    });
  });

  page.on('response', (response) => {
    if (response.status() < 400) {
      return;
    }
    report.httpErrors.push({
      step: currentStep,
      pageUrl: page.url(),
      url: response.url(),
      status: response.status(),
      resourceType: response.request().resourceType(),
    });
  });
}

async function waitForStreamlit(page) {
  await page.locator('[data-testid="stAppViewContainer"]').waitFor({
    state: 'visible',
    timeout: timeoutMs,
  });
  await page.waitForFunction(
    () => document.querySelector('[data-testid="stAppViewContainer"]')?.innerText.trim().length > 0,
    null,
    { timeout: timeoutMs },
  );
  await page.waitForTimeout(750);
}

async function waitForPathname(page, pathname) {
  await page.waitForFunction(
    (expected) => {
      const normalize = (value) => value.length > 1 ? value.replace(/\/$/, '') : value;
      return normalize(window.location.pathname) === normalize(expected);
    },
    pathname,
    { timeout: timeoutMs },
  );
}

async function expandNavigation(page) {
  const sidebarToggle = page.getByRole('button', {
    name: /keyboard_double_arrow_right/i,
  }).first();
  if (await sidebarToggle.isVisible().catch(() => false)) {
    await sidebarToggle.click({ force: true });
    await page.waitForTimeout(300);
  }

  const navigationExpander = page.locator('[data-testid="stExpander"]')
    .filter({ hasText: 'Navigation' })
    .first();
  const details = navigationExpander.locator('details');
  if (
    await navigationExpander.isVisible().catch(() => false)
    && !await details.evaluate((element) => element.open)
  ) {
    await navigationExpander.locator('summary').click({ force: true });
    await page.waitForTimeout(300);
  }
}

async function navigateTo(page, pageDefinition) {
  await expandNavigation(page);
  const link = page.getByRole('link', { name: pageDefinition.label }).first();
  await link.waitFor({ state: 'visible', timeout: timeoutMs });
  const href = await link.getAttribute('href');
  const expectedPathname = href ? new URL(href, page.url()).pathname : null;
  await link.click();
  if (expectedPathname) {
    await waitForPathname(page, expectedPathname);
  }
  await waitForStreamlit(page);
  await page.getByRole('heading', { name: pageDefinition.ready }).first().waitFor({
    state: 'visible',
    timeout: timeoutMs,
  });
  return expectedPathname || new URL(page.url()).pathname;
}

async function clickRadioChoice(page, groupName, choice) {
  const group = page.getByRole('radiogroup', { name: groupName }).first();
  await group.waitFor({ state: 'visible', timeout: timeoutMs });
  await group.getByText(choice, { exact: true }).click();
  await page.waitForTimeout(500);
}

async function clickButton(page, name, timeout = timeoutMs) {
  const button = page.getByRole('button', { name }).first();
  await button.waitFor({ state: 'visible', timeout });
  const buttonHandle = await button.elementHandle();
  await page.waitForFunction(
    (element) => element && !element.disabled && element.getAttribute('aria-disabled') !== 'true',
    buttonHandle,
    { timeout },
  );
  await button.click();
}

async function setupUploadedCorpora(page) {
  const targetFiles = await listFixtures('tar_corpus');
  const referenceFiles = await listFixtures('ref_corpus');
  if (targetFiles.length === 0 || referenceFiles.length === 0) {
    throw new Error('Target and reference text fixtures are required');
  }

  await clickRadioChoice(
    page,
    /What kind of corpus would you like to prepare\?/i,
    'New',
  );
  const targetUpload = page.getByLabel('Upload your target corpus').first()
    .locator('input[type="file"]');
  await targetUpload.setInputFiles(targetFiles);
  await clickButton(page, /Upload Target/i);
  await clickButton(page, /Process Target/i);
  await page.getByText('Reference corpus:', { exact: true }).first().waitFor({
    state: 'visible',
    timeout: timeoutMs,
  });

  await clickRadioChoice(
    page,
    /Would you like to load a reference corpus\?/i,
    'Yes',
  );
  await clickRadioChoice(
    page,
    /What kind of reference corpus would you like to prepare\?/i,
    'New',
  );
  const referenceUpload = page.getByLabel('Upload your reference corpus').first()
    .locator('input[type="file"]');
  await referenceUpload.setInputFiles(referenceFiles);
  await clickButton(page, /Upload Reference/i);
  await clickButton(page, /Process Reference/i);
  await page.getByRole('tab', { name: /Reference corpus/i }).first().waitFor({
    state: 'visible',
    timeout: timeoutMs,
  });

  await processMetadata(page, {
    expected: /Found 1 categories.*between 2 and 20 categories/is,
    outcome: 'graceful-not-applicable',
  });
}

async function chooseOption(page, label, option) {
  const combobox = page.getByRole('combobox', { name: label }).first();
  await combobox.waitFor({ state: 'visible', timeout: timeoutMs });
  await combobox.click();
  await page.getByRole('option', { name: option }).first().click();
}

async function setupInternalCorpora(page) {
  await clickRadioChoice(
    page,
    /What kind of corpus would you like to prepare\?/i,
    'Internal',
  );
  await chooseOption(
    page,
    /Select a saved corpus to load:/i,
    process.env.LOAD_TEST_TARGET_CORPUS || 'A_MICUSP_mini',
  );
  await clickButton(page, /Process Target/i);
  await page.getByText('Reference corpus:', { exact: true }).first().waitFor({
    state: 'visible',
    timeout: timeoutMs,
  });

  await clickRadioChoice(
    page,
    /Would you like to load a reference corpus\?/i,
    'Yes',
  );
  await clickRadioChoice(
    page,
    /What kind of reference corpus would you like to prepare\?/i,
    'Internal',
  );
  await chooseOption(
    page,
    /Select a saved corpus to load:/i,
    process.env.LOAD_TEST_REFERENCE_CORPUS || 'C_BAWE_mini',
  );
  await clickButton(page, /Process Reference/i);
  await page.getByRole('tab', { name: /Reference corpus/i }).first().waitFor({
    state: 'visible',
    timeout: timeoutMs,
  });

  await processMetadata(page, {
    expected: /Successfully processed \d+ document categories!/i,
    outcome: 'metadata-ready',
  });
}

async function processMetadata(page, expectation) {
  await clickRadioChoice(
    page,
    /Do you have categories in your file names to process\?/i,
    'Yes',
  );
  await clickButton(page, /Process Document Metadata/i);
  const message = page.getByText(expectation.expected).first();
  await message.waitFor({ state: 'visible', timeout: timeoutMs });
  report.setupChecks.push({
    check: 'process-target-metadata',
    status: 'passed',
    expectedOutcome: expectation.outcome,
    message: (await message.innerText()).trim(),
  });
}

async function setupCorpora(page) {
  await navigateTo(page, corePages[1]);
  if (profile === 'analysis') {
    await setupInternalCorpora(page);
    return;
  }
  if (profile !== 'uploads') {
    throw new Error(`Unknown DOCUSCOPE_AUDIT_PROFILE value: ${profile}`);
  }
  await setupUploadedCorpora(page);
}

async function clickVisibleButton(page, name, interactions) {
  const button = page.getByRole('button', { name }).first();
  if (!await button.isVisible().catch(() => false)) {
    return false;
  }
  if (!await button.isEnabled().catch(() => false)) {
    return false;
  }
  await button.click();
  interactions.push(`button:${name}`);
  await page.waitForTimeout(750);
  return true;
}

async function clickRequiredButton(page, name, interactions) {
  if (!await clickVisibleButton(page, name, interactions)) {
    throw new Error(`Required enabled button not found: ${name}`);
  }
}

async function waitForAny(page, locators, description) {
  const result = await Promise.any(
    locators.map((locator) => locator.first().waitFor({
      state: 'visible',
      timeout: timeoutMs,
    }).then(() => true)),
  ).catch(() => null);
  if (!result) {
    throw new Error(`Expected result did not appear: ${description}`);
  }
}

async function exerciseLanding(page, interactions) {
  const navigationExpander = page.locator('[data-testid="stExpander"]')
    .filter({ hasText: 'Navigation' })
    .first();
  await navigationExpander.locator('summary').click({ force: true });
  interactions.push('navigation:toggle');
  await page.waitForTimeout(250);
}

async function exerciseManageCorpus(page, interactions) {
  const targetTab = page.getByRole('tab', { name: /Target corpus/i }).first();
  if (await targetTab.isVisible().catch(() => false)) {
    await targetTab.click();
    interactions.push('tab:Target corpus');
    return;
  }
  const firstExpander = page.locator('[data-testid="stExpander"]')
    .filter({ hasNotText: 'Navigation' })
    .first();
  await firstExpander.locator('summary').click();
  interactions.push('expander:corpus information');
}

async function generateTokenFrequencies(page, interactions) {
  await clickRequiredButton(page, /Frequency Table/i, interactions);
  await page.locator('[data-testid="stDataFrame"]').first().waitFor({
    state: 'visible', timeout: timeoutMs,
  });
}

async function generateTagFrequencies(page, interactions) {
  await clickRequiredButton(page, /Tags Table/i, interactions);
  await page.locator('[data-testid="stDataFrame"]').first().waitFor({
    state: 'visible', timeout: timeoutMs,
  });
}

async function generateNgrams(page, interactions) {
  await clickRadioChoice(page, /What kind of table would you like to generate\?/i, 'N-grams');
  interactions.push('radio:N-grams');
  await clickRequiredButton(page, /N-grams Table/i, interactions);
  await waitForAny(page, [
    page.locator('[data-testid="stDataFrame"]'),
    page.getByText(/No n-grams match/i),
  ], 'n-gram result or explicit empty state');
}

async function generateKeyness(page, interactions) {
  await clickRequiredButton(page, /Keyness Table/i, interactions);
  await waitForAny(page, [
    page.locator('[data-testid="stDataFrame"]'),
    page.getByText(/No frequency data available/i),
  ], 'keyness result or explicit empty state');
}

async function generateCollocations(page, interactions) {
  const input = page.getByRole('textbox', { name: /Node word:/i }).first();
  if (await input.isVisible().catch(() => false)) {
    await input.fill('the');
    interactions.push('input:Node word=the');
  }
  await clickRequiredButton(page, /Collocations Table/i, interactions);
  await waitForAny(page, [
    page.locator('[data-testid="stDataFrame"]'),
    page.getByText(/No collocations/i),
  ], 'collocation result or explicit empty state');
}

async function generateKwic(page, interactions) {
  const input = page.getByRole('textbox', { name: /^Node word$/i }).first();
  if (await input.isVisible().catch(() => false)) {
    await input.fill('the');
    interactions.push('input:Node word=the');
  }
  await clickRequiredButton(page, /KWIC Table/i, interactions);
  await waitForAny(page, [
    page.locator('[data-testid="stDataFrame"]'),
    page.getByText(/No matching tokens/i),
  ], 'KWIC result or explicit empty state');
}

async function loadMatrixTables(page, interactions) {
  const loadButton = page.getByRole('button', { name: /Load Tables/i }).first();
  if (await loadButton.isVisible().catch(() => false)) {
    await clickRequiredButton(page, /Load Tables/i, interactions);
  }
  const source = page.getByRole('combobox', { name: /DFM source/i }).first();
  await source.waitFor({ state: 'visible', timeout: timeoutMs });
  await source.click();
  await page.keyboard.press('ArrowDown');
  await page.keyboard.press('Enter');
  interactions.push('select:DFM source');
}

async function processDocument(page, interactions) {
  const processButton = page.getByRole('button', { name: /Process Document/i }).first();
  if (await processButton.isVisible().catch(() => false)) {
    await clickRequiredButton(page, /Process Document/i, interactions);
  } else {
    const tagset = page.getByRole('radio').first();
    await tagset.click();
    interactions.push('radio:document tagset');
  }
}

async function exerciseCorpusParts(page, interactions) {
  if (profile === 'uploads') {
    await page.getByText(/No metadata has been processed yet/i).first().waitFor({
      state: 'visible',
      timeout: timeoutMs,
    });
    await clickRequiredButton(page, /Keyness Table of Corpus Parts/i, interactions);
    await page.getByText(/select at least one.*target category/i).first().waitFor({
      state: 'visible',
      timeout: timeoutMs,
    });
    interactions.push('expected:metadata-not-applicable');
    return;
  }

  const targetGroup = page.locator('[data-testid="stButtonGroup"]')
    .filter({ hasText: 'Select target categories:' })
    .first();
  const referenceGroup = page.locator('[data-testid="stButtonGroup"]')
    .filter({ hasText: 'Select reference categories:' })
    .first();
  await targetGroup.waitFor({ state: 'visible', timeout: timeoutMs });
  await referenceGroup.waitFor({ state: 'visible', timeout: timeoutMs });
  const targetCategory = process.env.DOCUSCOPE_AUDIT_TARGET_CATEGORY || 'BIO';
  const referenceCategory = process.env.DOCUSCOPE_AUDIT_REFERENCE_CATEGORY || 'ENG';
  const targetButton = targetGroup.getByText(targetCategory, { exact: true });
  const referenceButton = referenceGroup.getByText(referenceCategory, { exact: true });
  await targetButton.click();
  interactions.push(`category:target=${targetCategory}`);
  await referenceButton.click();
  interactions.push(`category:reference=${referenceCategory}`);
  await clickRequiredButton(page, /Keyness Table of Corpus Parts/i, interactions);
  await page.locator('[data-testid="stDataFrame"]').first().waitFor({
    state: 'visible',
    timeout: timeoutMs,
  });
}

async function exerciseAdvancedPlotting(page, interactions) {
  await clickRadioChoice(page, /What kind of plot would you like to make\?/i, 'PCA');
  interactions.push('radio:PCA');
  await clickRequiredButton(page, /Generate PCA/i, interactions);
  await page.locator('[data-testid="stPlotlyChart"]').first().waitFor({
    state: 'visible', timeout: timeoutMs,
  });
}

async function exerciseAiPlotting(page, interactions) {
  const loadButton = page.getByRole('button', { name: /Load Tables/i }).first();
  if (await loadButton.isVisible().catch(() => false)) {
    await clickRequiredButton(page, /Load Tables/i, interactions);
    await waitForStreamlit(page);
  }
  const combobox = page.getByRole('combobox').first();
  await combobox.waitFor({ state: 'visible', timeout: timeoutMs });
  await combobox.click();
  await page.keyboard.press('ArrowDown');
  await page.keyboard.press('Enter');
  interactions.push('select:AI data source');
}

async function chooseRadio(page, groupName, choice, interactions) {
  await clickRadioChoice(page, groupName, choice);
  interactions.push(`radio:${choice}`);
}

async function exerciseCorpusDownload(page, interactions) {
  const loadButton = page.getByRole('button', { name: /Load Data/i }).first();
  if (await loadButton.isVisible().catch(() => false)) {
    await clickRequiredButton(page, /Load Data/i, interactions);
    await waitForStreamlit(page);
  }
  await chooseRadio(page, /Choose a corpus/i, 'Reference', interactions);
  await chooseRadio(
    page,
    /Choose the data to download/i,
    'All of the processed data',
    interactions,
  );
}

async function exerciseTaggedDownload(page, interactions) {
  const loadButton = page.getByRole('button', { name: /Load Data/i }).first();
  if (await loadButton.isVisible().catch(() => false)) {
    await clickRequiredButton(page, /Load Data/i, interactions);
    await waitForStreamlit(page);
  }
  const tagsetGroup = page.getByRole('radiogroup').first();
  await tagsetGroup.waitFor({ state: 'visible', timeout: timeoutMs });
  const labels = tagsetGroup.locator('label');
  const index = await labels.count() > 1 ? 1 : 0;
  await labels.nth(index).click();
  interactions.push('radio:tagged-file tagset');
}

async function exerciseGenericControls(page, interactions) {
  const tabs = page.getByRole('tab');
  const tabCount = Math.min(await tabs.count(), 8);
  for (let index = 0; index < tabCount; index += 1) {
    const tab = tabs.nth(index);
    if (!await tab.isVisible().catch(() => false)) {
      continue;
    }
    const name = (await tab.innerText()).trim();
    await tab.click();
    interactions.push(`tab:${name}`);
    await page.waitForTimeout(250);
  }

  const expanders = page.locator('[data-testid="stExpander"] details:not([open]) > summary');
  const expanderCount = Math.min(await expanders.count(), 8);
  for (let index = 0; index < expanderCount; index += 1) {
    const expander = expanders.nth(index);
    if (!await expander.isVisible().catch(() => false)) {
      continue;
    }
    const name = (await expander.innerText()).trim().replace(/\s+/g, ' ');
    if (/Navigation/i.test(name)) {
      continue;
    }
    await expander.click();
    interactions.push(`expander:${name}`);
    await page.waitForTimeout(200);
  }
}

async function collectDisplayedDiagnostics(page) {
  const exceptions = await page.locator('[data-testid="stException"]').allInnerTexts();
  const alerts = await page.locator('[data-testid="stAlert"]').allInnerTexts();
  const errorAlerts = alerts.filter((text) => (
    /traceback|keyerror|exception|error loading|an error occurred|invalid session state|could not load/i
      .test(text)
  ));
  return {
    exceptions: exceptions.map((text) => text.trim()),
    alerts: alerts.map((text) => text.trim()),
    errorAlerts: errorAlerts.map((text) => text.trim()),
  };
}

async function auditPage(page, pageDefinition) {
  currentStep = `page:${pageDefinition.label}`;
  const startedAt = Date.now();
  const pageResult = {
    label: pageDefinition.label,
    pathname: null,
    status: 'passed',
    durationMs: 0,
    interactions: [],
    exceptions: [],
    alerts: [],
    errorAlerts: [],
  };

  try {
    pageResult.pathname = await navigateTo(page, pageDefinition);
    await pageDefinition.action(page, pageResult.interactions);
    await waitForStreamlit(page);
    await exerciseGenericControls(page, pageResult.interactions);
    if (pageResult.interactions.length === 0) {
      throw new Error('Page completed without exercising an interaction');
    }
    const diagnostics = await collectDisplayedDiagnostics(page);
    Object.assign(pageResult, diagnostics);
    if (diagnostics.exceptions.length > 0 || diagnostics.errorAlerts.length > 0) {
      throw new Error(
        `Displayed application errors: ${[
          ...diagnostics.exceptions,
          ...diagnostics.errorAlerts,
        ].join(' | ')}`,
      );
    }
  } catch (error) {
    pageResult.status = 'failed';
    pageResult.error = messageOf(error);
    const screenshot = path.join(artifactsDir, `${slugify(pageDefinition.label)}.png`);
    await page.screenshot({ path: screenshot, fullPage: true }).catch(() => null);
    pageResult.screenshot = path.relative(loadTestsDir, screenshot);
  } finally {
    pageResult.durationMs = Date.now() - startedAt;
    report.pages.push(pageResult);
    console.log(
      `[page-audit] ${pageResult.status.toUpperCase()} ${pageDefinition.label} `
      + `${pageResult.durationMs}ms interactions=${pageResult.interactions.length}`,
    );
  }
}

function hasFatalDiagnostics() {
  const failedRequest = report.requestFailures.some((failure) => (
    ['document', 'xhr', 'fetch', 'websocket'].includes(failure.resourceType)
  ));
  const serverError = report.httpErrors.some((error) => error.status >= 500);
  return report.runtimeErrors.length > 0 || failedRequest || serverError;
}

async function writeReport() {
  report.finishedAt = new Date().toISOString();
  report.summary = {
    passed: report.pages.filter((page) => page.status === 'passed').length,
    failed: report.pages.filter((page) => page.status === 'failed').length,
    runtimeErrors: report.runtimeErrors.length,
    consoleErrors: report.consoleErrors.length,
    requestFailures: report.requestFailures.length,
    httpErrors: report.httpErrors.length,
  };
  const stamp = report.startedAt.replace(/[:.]/g, '-');
  const reportPath = path.join(reportsDir, `page-audit-${stamp}.json`);
  await fs.writeFile(reportPath, `${JSON.stringify(report, null, 2)}\n`, 'utf8');
  console.log(`[page-audit] report: ${reportPath}`);
  return reportPath;
}

async function main() {
  await fs.mkdir(artifactsDir, { recursive: true });
  const browser = await chromium.launch({ headless: !headed });
  const context = await browser.newContext({
    viewport: { width: 1440, height: 1000 },
    acceptDownloads: false,
  });
  const page = await context.newPage();
  page.setDefaultTimeout(timeoutMs);
  attachDiagnostics(page);

  try {
    currentStep = 'landing';
    await page.goto(baseUrl, { waitUntil: 'domcontentloaded', timeout: timeoutMs });
    await waitForStreamlit(page);

    currentStep = `setup:${profile}`;
    await setupCorpora(page);
    console.log(`[page-audit] corpus setup complete profile=${profile}`);

    for (const pageDefinition of corePages) {
      await auditPage(page, pageDefinition);
    }
  } catch (error) {
    report.setupError = messageOf(error);
    const screenshot = path.join(artifactsDir, 'setup-failure.png');
    await page.screenshot({ path: screenshot, fullPage: true }).catch(() => null);
    report.setupScreenshot = path.relative(loadTestsDir, screenshot);
    console.error(`[page-audit] setup failed: ${report.setupError}`);
  } finally {
    await context.close().catch(() => null);
    await browser.close().catch(() => null);
  }

  await writeReport();
  if (
    report.setupError
    || report.pages.some((pageResult) => pageResult.status === 'failed')
    || hasFatalDiagnostics()
  ) {
    process.exitCode = 1;
  }
}

main().catch(async (error) => {
  report.fatalError = messageOf(error);
  console.error(error);
  await writeReport().catch(() => null);
  process.exitCode = 1;
});
