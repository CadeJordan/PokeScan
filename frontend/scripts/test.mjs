// `npm test`: runs `node --test` on the frontend's test files and fails when
// no test ran. `node --test` exits 0 when its glob matches nothing, so a moved
// test folder or a broken glob would otherwise pass CI with zero tests.
//
//   node scripts/test.mjs                 # src/**/*.test.ts
//   node scripts/test.mjs "<glob>" ...    # other files (e.g. to check the guard)
import { spawnSync } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const globs = process.argv.slice(2);
const files = globs.length > 0 ? globs : ["src/**/*.test.ts"];

// Keep node's usual console reporter (spec on a terminal, TAP otherwise) and
// also write TAP to a temp file, whose summary has the number of tests run.
const dir = mkdtempSync(join(tmpdir(), "pokescan-test-"));
const tap = join(dir, "report.tap");
let status;
let summary;
try {
  const result = spawnSync(
    process.execPath,
    [
      "--test",
      "--experimental-strip-types",
      `--test-reporter=${process.stdout.isTTY ? "spec" : "tap"}`,
      "--test-reporter-destination=stdout",
      "--test-reporter=tap",
      `--test-reporter-destination=${tap}`,
      ...files,
    ],
    { stdio: "inherit" },
  );
  if (result.error) throw result.error;
  status = result.status ?? 1;
  try {
    summary = readFileSync(tap, "utf8");
  } catch {
    summary = "";
  }
} finally {
  rmSync(dir, { recursive: true, force: true });
}

const match = /^# tests (\d+)$/m.exec(summary);
const count = match ? Number(match[1]) : 0;
if (status !== 0) process.exit(status);
if (count === 0) {
  console.error(`\nnpm test: no tests ran (files: ${files.join(" ")}); failing.`);
  process.exit(1);
}
