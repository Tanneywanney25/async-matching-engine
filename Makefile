.PHONY: install test bench dashboard api lint clean

install:  ## Install the package with dev extras
	pip install -e ".[dev]"

test:  ## Run the test suite
	pytest -q

bench:  ## Run the latency benchmark
	python scripts/benchmark.py

dashboard:  ## Launch the Rich terminal dashboard
	python -m exchange_engine.dashboard

api:  ## Run the FastAPI server
	uvicorn exchange_engine.engine:app --port 8000 --reload

clean:  ## Remove caches and build artifacts
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

# ---- GPU analytics layer (every target is a thin wrapper around a python command) ----
.PHONY: record record-stop record-status archive synth bench-cpu probe remote-synth remote-real render check

record:  ## Start the raw Coinbase recorder in the foreground
	python scripts/record.py

record-stop:  ## Stop the recorder gracefully through its PID file
	python scripts/record.py --stop

record-status:  ## Print the recorder sidecar
	python scripts/record.py --status

archive:  ## Build the deterministic snapshot archive from data/raw
	python scripts/build_archive.py --raw data/raw --out data/archive

synth:  ## Write a synthetic archive with the same schema
	python scripts/synth_archive.py --out data/synth

bench-cpu:  ## CPU-side checks of every kernel's NumPy reference
	pytest -q tests/test_ref_*.py tests/test_archive.py

probe:  ## Environment probe on a Kaggle T4
	python -m remote.kaggle probe

remote-synth:  ## Full benchmark job on the synthetic archive (never reaches the docs)
	python -m remote.kaggle run --job bench_all --dataset tanushbhatia/exchange-engine-archive-synth --source synthetic

remote-real:  ## Rebuild, upload, run, pull and re-render everything from real data
	python -m bench.reproduce --source real

render:  ## Render every document and the dashboard data from results/*/metrics.json
	python scripts/render_docs.py

check:  ## Provenance and bullet checks
	python scripts/check_numbers.py && python scripts/check_bullets.py
