PYTHON=python3

.PHONY: format lint typecheck check

format:
	ruff format src

lint:
	ruff check src

typecheck:
	mypy src

check: format lint typecheck
