import { execFileSync } from "node:child_process";
import { copyFileSync, mkdirSync, mkdtempSync, readFileSync, readdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const pages = dirname(fileURLToPath(import.meta.url));
const repo = dirname(pages);
const dist = join(pages, "dist");
const project = mkdtempSync(join(tmpdir(), "sin-pages-"));

// Publicar solo código y semillas versionadas, nunca data/raw, certificados ni secretos.
const appFiles = [
  "app.py",
  "data/reference/oni_noaa.csv",
  "data/reference/nino34_mensual.csv",
  "data/reference/nino_mensual.csv",
  "data/reference/v4_mensual.csv",
];

try {
  for (const file of appFiles) {
    const dest = join(project, file);
    mkdirSync(dirname(dest), { recursive: true });
    copyFileSync(join(repo, file), dest);
  }
  copyFileSync(join(pages, "dashboard.py"), join(project, "dashboard.py"));

  const cli = join(pages, "node_modules", "@stlite", "cli", "dist", "cli.js");
  execFileSync(process.execPath, [
    cli, "web", project, "--entrypoint", "dashboard.py",
    "--requirements", join(pages, "requirements.txt"), "--out", dist,
  ], { stdio: "inherit" });

  // El CLI genera rutas relativas para servir también bajo /dashboard-sin-nodal/.
  const index = join(dist, "index.html");
  const html = readFileSync(index, "utf8")
    .replace("<html>", '<html lang="es">')
    .replace("<title>Stlite app</title>", "<title>Agua, precio y El Niño · SIN Colombia</title>")
    .replace('<div id="root"></div>', '<div id="root"><p role="status">Cargando el tablero del SIN…</p></div>')
    .replace("<body>", "<body><noscript>Activa JavaScript para ejecutar el tablero en tu navegador.</noscript>");
  writeFileSync(index, html);
  writeFileSync(join(dist, ".nojekyll"), "");
  for (const file of readdirSync(dist)) {
    if (file.endsWith(".map")) rmSync(join(dist, file));
  }
} finally {
  rmSync(project, { recursive: true, force: true });
}
