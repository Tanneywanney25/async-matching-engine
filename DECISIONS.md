# Decisions log

One line per decision taken without a human in the loop, with the reason. Newest at the bottom.

- 2026-10-21 03:45 Started a fresh `gpu-analytics` branch from `main` (aba2b95); no earlier run, `data/`, recorder PID, results or Kaggle dataset existed, so there was nothing to build on.
- 2026-10-21 03:45 The project venv is Python 3.14.7 while the system `kaggle` CLI (2.2.4) lives in Python 3.12; the remote runner shells out to the `kaggle` executable instead of importing it, so the venv stays unchanged.
- 2026-10-21 03:45 `node --test tests/frontend/` fails on Node 24 because a bare directory is resolved as a module; added `tests/frontend/index.js` (plus a `package.json` with `type: module`) that imports every `*.test.mjs`, so the directory form in the acceptance list runs the same suite.
- 2026-10-21 03:45 Session budget is 6 hours (the goal line overrides the 16 hours in the spec), so the recording window is capped accordingly and the actual snapshot count will be reported if the 1,200,000 target is not reached.
