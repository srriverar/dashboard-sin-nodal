import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./tests",
  timeout: 180_000,
  use: {
    baseURL: "http://127.0.0.1:8765",
    trace: "retain-on-failure",
  },
  webServer: {
    command: "python3 -m http.server --bind 127.0.0.1 8765",
    url: "http://127.0.0.1:8765/dist/",
  },
});
