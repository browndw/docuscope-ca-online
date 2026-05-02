import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';

import { internalKeynessScenario } from '../processors/streamlit-metrics.mjs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const reportsDir = path.resolve(__dirname, '../reports');

async function main() {
  await fs.mkdir(reportsDir, { recursive: true });

  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext();
  const page = await context.newPage();

  const events = {
    emit(name, metric, value) {
      console.log(`[plain-playwright] emit ${name} ${metric} ${value}`);
    }
  };

  const vuContext = {
    vars: {
      baseUrl: process.env.LOAD_TEST_BASE_URL || 'http://127.0.0.1:8501',
      targetCorpus: process.env.LOAD_TEST_TARGET_CORPUS || 'A_MICUSP_mini',
      referenceCorpus: process.env.LOAD_TEST_REFERENCE_CORPUS || 'C_BAWE_mini',
      dictionaryLabel: process.env.LOAD_TEST_DICTIONARY_LABEL || 'Large Dictionary'
    }
  };

  try {
    await internalKeynessScenario(page, vuContext, events);
    console.log('[plain-playwright] internal keyness flow succeeded');
  } catch (error) {
    const screenshotPath = path.join(reportsDir, 'plain-keyness-failure.png');
    await page.screenshot({ path: screenshotPath, fullPage: true }).catch(() => {});
    console.error(`[plain-playwright] failure screenshot: ${screenshotPath}`);
    throw error;
  } finally {
    await context.close().catch(() => {});
    await browser.close().catch(() => {});
  }
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});