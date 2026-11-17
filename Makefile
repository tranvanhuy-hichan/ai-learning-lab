.PHONY: web test lint format

web:            ## run the web app at http://127.0.0.1:8000
	cd projects/lab-web && python -m labweb

test:           ## run every project's tests
	@for p in projects/*/; do echo "== $$p"; (cd $$p && python -m pytest -q) || exit 1; done

lint:
	ruff check .
	ruff format --check .

format:
	ruff format .
	ruff check --fix .
