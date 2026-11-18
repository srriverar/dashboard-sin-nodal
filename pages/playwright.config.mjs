import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./tests",
  timeout: 150_000,
  expect: { timeout: 10_000 },
  reporter: process.env.GITHUB_ACTIONS ? [["list"], ["github"]] : "list",
  use: {
    baseURL: "http://127.0.0.1:8765",
    trace: "retain-on-failure",
  },
  webServer: {
    command: "python3 -m http.server --bind 127.0.0.1 8765",
    url: "http://127.0.0.1:8765/dist/",
  },
});
