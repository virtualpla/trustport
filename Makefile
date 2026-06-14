PY ?= python
PKG := trustport

.PHONY: help install dev lint type test smoke docker clean

help:
	@echo "targets: install dev lint type test smoke docker clean"

install:
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install --no-deps -e .

dev:
	$(PY) -m pip install ruff black isort mypy pytest hypothesis

lint:
	$(PY) -m ruff check .
	$(PY) -m black --check .
	$(PY) -m isort --check-only .

type:
	$(PY) -m mypy --strict $(PKG)

test:
	$(PY) -m pytest -q

smoke:
	$(PY) -m $(PKG).wheelhouse train --config configs/experiment/_smoke.yaml --out runs/smoke --steps 5
	$(PY) -m $(PKG).wheelhouse evaluate --config configs/experiment/_smoke.yaml

docker:
	docker build -t $(PKG):0.1.0 .

clean:
	rm -rf runs .pytest_cache .mypy_cache .ruff_cache *.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
