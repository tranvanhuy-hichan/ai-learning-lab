.PHONY: install run test lint format demo-figure

install:
	pip install -r requirements.txt -r requirements-dev.txt

run:
	python app.py

test:
	pytest

lint:
	ruff check .
	ruff format --check .

format:
	ruff format .
	ruff check --fix .

demo-figure:
	python scripts/make_demo_figure.py
