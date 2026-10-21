// Entry point so that `node --test tests/frontend/` works on Node 22+, which
// resolves a bare directory argument as a module instead of a search root.
// Every *.test.mjs in this directory is imported, registering its tests.
import { readdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const files = readdirSync(here)
  .filter((name) => name.endsWith(".test.mjs"))
  .sort();

for (const name of files) {
  await import(pathToFileURL(join(here, name)).href);
}
