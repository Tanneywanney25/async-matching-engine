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
