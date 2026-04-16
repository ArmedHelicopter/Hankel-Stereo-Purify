PYTHON=python3

.PHONY: format lint typecheck test check

format:
	ruff format src tests

lint:
	ruff check .

typecheck:
	PYTHONPATH=src $(PYTHON) -m mypy src/ tests/

test:
	PYTHONPATH=src $(PYTHON) -m pytest tests/

# 与 .github/workflows/ci.yml 一致：不自动 format，避免未提交的格式化改动
check: lint typecheck test
