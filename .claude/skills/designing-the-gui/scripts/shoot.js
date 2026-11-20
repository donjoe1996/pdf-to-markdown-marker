// Full-page screenshot of each tab. Node, because this environment ships
// Playwright for Node; any Chromium Playwright install works.
const { chromium } = require('playwright');
const [url, prefix] = process.argv.slice(2);
(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1400, height: 1000 } });
  await page.goto(url);
  await page.waitForSelector("[data-testid='stTab']", { timeout: 60000 });
  await page.waitForTimeout(4000);
  const tabs = page.locator("[data-testid='stTab']");
  for (let i = 0; i < await tabs.count(); i++) {
    await tabs.nth(i).click();
    await page.waitForTimeout(2500);
    await page.screenshot({ path: `${prefix}-${i}.png`, fullPage: true });
  }
  await browser.close();
})();
