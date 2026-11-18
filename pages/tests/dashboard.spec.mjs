import { expect, test } from "@playwright/test";

// Serve from a subdirectory like a GitHub Pages project site. The built app
// must start without a CDN, a Python server, or the XM/NOAA APIs at runtime.
test("renders the historical dashboard and simulates a nodal scenario", async ({ page, context, baseURL }) => {
  const diagnostics = [];
  page.on("pageerror", error => diagnostics.push(error.message));
  page.on("console", message => {
    if (message.type() === "error") diagnostics.push(message.text());
  });
  page.on("response", response => {
    if (response.status() >= 400) diagnostics.push(`${response.status()} ${response.url()}`);
  });
  await context.route(/^https?:\/\//, route =>
    new URL(route.request().url()).origin === baseURL ? route.continue() : route.abort());

  try {
    await page.goto("/dist/");
    await expect(page.getByRole("heading", { name: /Agua, precio y El Niño/ })).toBeVisible({ timeout: 60_000 });
    await expect(page.getByText("GitHub Pages · modo de referencia.")).toBeVisible();
    await expect(page.getByText(/Último ONI disponible/)).toBeVisible();
    await expect(page.locator(".js-plotly-plot").first()).toBeVisible();

    await page.getByRole("tab", { name: /Economía ENSO/ }).click();
    await expect(page.getByRole("heading", { name: "El Niño en dinero" })).toBeVisible();
    await expect(page.getByRole("tabpanel").locator(".js-plotly-plot")).toBeVisible();

    await page.getByRole("tab", { name: /Precios nodales/ }).click();
    await page.getByRole("button", { name: "Simular escenario" }).click();
    await expect(page.getByRole("heading", { name: /Mapa de calor del precio marginal/ }))
      .toBeVisible({ timeout: 60_000 });
    await expect(page.getByRole("tabpanel").locator(".js-plotly-plot")).toHaveCount(2);

    await page.getByRole("tab", { name: /Datos/ }).click();
    await expect(page.getByRole("button", { name: "Descargar oni_noaa.csv" })).toBeVisible();
    await expect(page.getByTestId("stException")).toHaveCount(0);
  } catch (error) {
    const summary = diagnostics.slice(-15).join("\n").slice(-4000);
    const body = await page.locator("body").innerText({ timeout: 1000 }).catch(() => "");
    throw new Error(`${error.message}\n\nBrowser errors:\n${summary}\n\nPage text:\n${body.slice(0, 2000)}`, { cause: error });
  }
});
