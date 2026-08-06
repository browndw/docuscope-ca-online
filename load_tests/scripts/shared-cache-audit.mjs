import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const loadTestsDir = path.resolve(__dirname, '..');
const reportsDir = path.join(loadTestsDir, 'reports');

const baseUrl = process.env.LOAD_TEST_BASE_URL || 'http://127.0.0.1:8501';
const timeoutMs = positiveInteger(process.env.DOCUSCOPE_CACHE_AUDIT_TIMEOUT_MS, 180000);
const maxWarmRatio = positiveNumber(process.env.DOCUSCOPE_CACHE_AUDIT_MAX_WARM_RATIO, 0.7);
const minSavedMs = positiveInteger(process.env.DOCUSCOPE_CACHE_AUDIT_MIN_SAVED_MS, 500);
const targetCorpus = process.env.LOAD_TEST_TARGET_CORPUS || 'A_MICUSP_mini';
const referenceCorpus = process.env.LOAD_TEST_REFERENCE_CORPUS || 'C_BAWE_mini';
const targetCategory = process.env.DOCUSCOPE_CACHE_AUDIT_TARGET_CATEGORY || 'BIO';
const referenceCategory = process.env.DOCUSCOPE_CACHE_AUDIT_REFERENCE_CATEGORY || 'HIS';
const collocationNode = process.env.DOCUSCOPE_CACHE_AUDIT_COLLOCATION_NODE || 'however';
const ngramSpan = positiveInteger(process.env.DOCUSCOPE_CACHE_AUDIT_NGRAM_SPAN, 4);

const report = {
  startedAt: new Date().toISOString(),
  baseUrl,
  contract: { maxWarmRatio, minSavedMs },
  corpora: { targetCorpus, referenceCorpus },
  scenarios: [],
  browserErrors: [],
};

function positiveInteger(value, fallback) {
  const parsed = Number.parseInt(value || '', 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

function positiveNumber(value, fallback) {
  const parsed = Number.parseFloat(value || '');
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

function contains(text) {
  return text instanceof RegExp ? text : new RegExp(text, 'i');
}

async function waitForStreamlit(page) {
  await page.locator('[data-testid="stAppViewContainer"]').waitFor({
    state: 'visible', timeout: timeoutMs,
  });
  await page.waitForTimeout(500);
}

async function expandNavigation(page) {
  const sidebarToggle = page.getByRole('button', {
    name: /keyboard_double_arrow_right/i,
  }).first();
  if (await sidebarToggle.isVisible().catch(() => false)) {
    await sidebarToggle.click({ force: true });
  }

  const expander = page.locator('[data-testid="stExpander"]')
    .filter({ hasText: 'Navigation' })
    .first();
  if (!await expander.isVisible().catch(() => false)) {
    return;
  }
  const details = expander.locator('details');
  if (!await details.evaluate((element) => element.open)) {
    await expander.locator('summary').click({ force: true });
  }
}

async function navigate(page, label, heading) {
  await expandNavigation(page);
  const link = page.getByRole('link', { name: label }).first();
  await link.waitFor({ state: 'visible', timeout: timeoutMs });
  await link.click();
  await waitForStreamlit(page);
  await page.getByRole('heading', { name: contains(heading) }).first().waitFor({
    state: 'visible', timeout: timeoutMs,
  });
}

async function clickRadio(page, groupName, choice) {
  const group = page.getByRole('radiogroup', { name: contains(groupName) }).first();
  await group.waitFor({ state: 'visible', timeout: timeoutMs });
  await group.getByText(choice, { exact: true }).click();
  await page.waitForTimeout(300);
}

async function chooseOption(page, label, option) {
  const combobox = page.getByRole('combobox', { name: contains(label) }).first();
  await combobox.waitFor({ state: 'visible', timeout: timeoutMs });
  await combobox.click();
  await page.getByRole('option', { name: contains(option) }).first().click();
}

async function clickButton(page, name) {
  const button = page.getByRole('button', { name: contains(name) }).first();
  await button.waitFor({ state: 'visible', timeout: timeoutMs });
  await button.click();
}

async function loadInternalCorpora(page, includeReference = true) {
  await navigate(page, 'Manage Corpus Data', 'Manage Corpus Data');
  await clickRadio(page, 'What kind of corpus would you like to prepare?', 'Internal');
  await chooseOption(page, 'Select a saved corpus to load:', targetCorpus);
  await clickButton(page, 'Process Target');
  await page.getByText('Reference corpus:', { exact: true }).first().waitFor({
    state: 'visible', timeout: timeoutMs,
  });

  if (!includeReference) {
    return;
  }
  await clickRadio(page, 'Would you like to load a reference corpus?', 'Yes');
  await clickRadio(
    page,
    'What kind of reference corpus would you like to prepare?',
    'Internal',
  );
  await chooseOption(page, 'Select a saved corpus to load:', referenceCorpus);
  await clickButton(page, 'Process Reference');
  await page.getByRole('tab', { name: /Reference corpus/i }).first().waitFor({
    state: 'visible', timeout: timeoutMs,
  });
}

async function processMetadata(page) {
  await clickRadio(
    page,
    'Do you have categories in your file names to process?',
    'Yes',
  );
  await clickButton(page, 'Process Document Metadata');
  await page.getByText(/Successfully processed \d+ document categories!/i).first()
    .waitFor({ state: 'visible', timeout: timeoutMs });
}

async function createUser(browser, name, includeReference = true, metadata = false) {
  const context = await browser.newContext({ viewport: { width: 1440, height: 1000 } });
  const page = await context.newPage();
  page.setDefaultTimeout(timeoutMs);
  page.on('pageerror', (error) => {
    report.browserErrors.push({ user: name, type: 'pageerror', message: error.message });
  });
  page.on('response', (response) => {
    if (response.status() >= 500) {
      report.browserErrors.push({
        user: name,
        type: 'http',
        status: response.status(),
        url: response.url(),
      });
    }
  });
  await page.goto(baseUrl, { waitUntil: 'domcontentloaded', timeout: timeoutMs });
  await waitForStreamlit(page);
  await loadInternalCorpora(page, includeReference);
  if (metadata) {
    await processMetadata(page);
  }
  return { context, page, name };
}

async function measureGeneration(page, trigger, queuePattern) {
  const queueStatus = page.getByText(queuePattern).first();
  const result = page.locator('[data-testid="stDataFrame"]').first();
  const startedAt = performance.now();
  const firstOutcome = Promise.race([
    queueStatus.waitFor({ state: 'visible', timeout: timeoutMs }).then(() => 'queue'),
    result.waitFor({ state: 'visible', timeout: timeoutMs }).then(() => 'result'),
  ]);
  await trigger();
  const queueSeen = await firstOutcome === 'queue';
  await result.waitFor({ state: 'visible', timeout: timeoutMs });
  const durationMs = performance.now() - startedAt;
  const renderedDataframes = await page.locator('[data-testid="stDataFrame"]').count();
  return { durationMs: Math.round(durationMs), queueSeen, renderedDataframes };
}

function evaluateTiming(name, cold, warm) {
  const savedMs = cold.durationMs - warm.durationMs;
  const ratio = warm.durationMs / cold.durationMs;
  const coldQualified = cold.queueSeen;
  const warmQualified = !warm.queueSeen;
  const timingPassed = ratio <= maxWarmRatio && savedMs >= minSavedMs;
  const status = !coldQualified
    ? 'inconclusive'
    : warmQualified && timingPassed ? 'passed' : 'failed';
  const result = {
    name,
    status,
    cold,
    warm,
    savedMs,
    warmToColdRatio: Number(ratio.toFixed(3)),
    assertions: {
      user1EnteredSharedQueue: coldQualified,
      user2AvoidedSharedQueue: warmQualified,
      maxWarmRatio,
      minSavedMs,
      timingPassed,
    },
  };
  if (!coldQualified) {
    result.reason = (
      'User 1 did not enter background generation. The selected manifest was already '
      + 'warm, so this run cannot prove cold-to-warm reuse. Change the configured '
      + 'parameters or run against a clean artifact store.'
    );
  }
  return result;
}

async function runScenario(name, users, prepare, trigger, queuePattern) {
  for (const user of users) {
    await prepare(user.page);
  }
  const cold = await measureGeneration(
    users[0].page,
    () => trigger(users[0].page),
    queuePattern,
  );
  const warm = await measureGeneration(
    users[1].page,
    () => trigger(users[1].page),
    queuePattern,
  );
  const result = evaluateTiming(name, cold, warm);
  report.scenarios.push(result);
  console.log(
    `[shared-cache-audit] ${result.status.toUpperCase()} ${name} `
    + `cold=${cold.durationMs}ms warm=${warm.durationMs}ms `
    + `ratio=${result.warmToColdRatio} saved=${result.savedMs}ms`,
  );
}

async function prepareKeyness(page) {
  await navigate(page, 'Compare Corpora', 'Compare Corpora');
  const threshold = page.getByRole('combobox', { name: /p-value threshold/i }).first();
  if (await threshold.isVisible().catch(() => false)) {
    await threshold.click();
    const option = page.getByRole('option', { name: '0.05', exact: true }).first();
    if (await option.isVisible().catch(() => false)) {
      await option.click();
    } else {
      await page.keyboard.press('Escape');
    }
  }
}

async function prepareCorpusParts(page) {
  await navigate(page, 'Compare Corpus Parts', 'Compare Corpus Parts');
  const targetGroup = page.locator('[data-testid="stButtonGroup"]')
    .filter({ hasText: 'Select target categories:' }).first();
  const referenceGroup = page.locator('[data-testid="stButtonGroup"]')
    .filter({ hasText: 'Select reference categories:' }).first();
  await targetGroup.getByText(targetCategory, { exact: true }).click();
  await referenceGroup.getByText(referenceCategory, { exact: true }).click();
}

async function prepareNgrams(page) {
  await navigate(page, 'Ngrams & Clusters', 'N-gram and Cluster Frequency');
  await clickRadio(page, 'What kind of table would you like to generate?', 'N-grams');
  const spanGroup = page.getByRole('radiogroup', { name: /Span of your n-grams:/i }).first();
  await spanGroup.getByText(String(ngramSpan), { exact: true }).click();
  const tagsetGroup = page.getByRole('radiogroup', { name: /Select a tagset:/i }).first();
  await tagsetGroup.getByText('DocuScope', { exact: true }).click();
}

async function prepareCollocations(page) {
  await navigate(page, 'Collocations', 'Collocates');
  await page.getByRole('textbox', { name: /Node word:/i }).first().fill(collocationNode);
}

async function writeReport() {
  report.finishedAt = new Date().toISOString();
  report.summary = {
    passed: report.scenarios.filter((scenario) => scenario.status === 'passed').length,
    failed: report.scenarios.filter((scenario) => scenario.status === 'failed').length,
    inconclusive: report.scenarios.filter((scenario) => scenario.status === 'inconclusive').length,
    browserErrors: report.browserErrors.length,
  };
  const stamp = report.startedAt.replace(/[:.]/g, '-');
  const reportPath = path.join(reportsDir, `shared-cache-audit-${stamp}.json`);
  await fs.mkdir(reportsDir, { recursive: true });
  await fs.writeFile(reportPath, `${JSON.stringify(report, null, 2)}\n`, 'utf8');
  console.log(`[shared-cache-audit] report: ${reportPath}`);
}

async function main() {
  const browser = await chromium.launch({ headless: process.env.DOCUSCOPE_AUDIT_HEADED !== '1' });
  const users = [];
  try {
    users.push(await createUser(browser, 'user-1', true, true));
    users.push(await createUser(browser, 'user-2', true, true));

    await runScenario(
      'keyness', users, prepareKeyness,
      (page) => clickButton(page, 'Keyness Table'),
      /Generating keyness tables in the background/i,
    );
    await runScenario(
      'corpus-parts-keyness', users, prepareCorpusParts,
      (page) => clickButton(page, 'Keyness Table of Corpus Parts'),
      /Generating corpus-parts keyness tables in the background/i,
    );
    await runScenario(
      'ngrams', users, prepareNgrams,
      (page) => clickButton(page, 'N-grams Table'),
      /Generating n-gram table in the background/i,
    );
    await runScenario(
      'collocations', users, prepareCollocations,
      (page) => clickButton(page, 'Collocations Table'),
      /Preparing shared collocations table/i,
    );
  } catch (error) {
    report.fatalError = error instanceof Error ? error.stack || error.message : String(error);
    console.error(report.fatalError);
  } finally {
    for (const user of users) {
      await user.context.close().catch(() => null);
    }
    await browser.close().catch(() => null);
  }

  await writeReport();
  if (
    report.fatalError
    || report.browserErrors.length > 0
    || report.scenarios.some((scenario) => scenario.status !== 'passed')
  ) {
    process.exitCode = 1;
  }
}

main().catch(async (error) => {
  report.fatalError = error instanceof Error ? error.stack || error.message : String(error);
  console.error(report.fatalError);
  await writeReport().catch(() => null);
  process.exitCode = 1;
});
