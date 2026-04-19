PYTHON=python3

.PHONY: format lint typecheck test check

format:
	ruff format src tests

lint:
	ruff check src tests

typecheck:
	PYTHONPATH=src $(PYTHON) -m mypy src tests

test:
	PYTHONPATH=src $(PYTHON) -m pytest tests/

check: lint typecheck test
