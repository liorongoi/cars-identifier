PY ?= .venv/bin/python
.PHONY: install check test lint typecheck ui serve

install:
	python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
	cd frontend && npm ci

lint:
	.venv/bin/ruff check src tests
	cd frontend && npm run lint

typecheck:
	.venv/bin/mypy
	cd frontend && npm run typecheck

test:
	.venv/bin/pytest --cov=carid --cov-report=term-missing
	cd frontend && npm test

check: lint typecheck test

ui:
	cd frontend && npm run build

serve: ui
	.venv/bin/carid init && .venv/bin/carid serve
