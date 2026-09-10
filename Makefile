# uv may not be on PATH yet (installer puts it in ~/.local/bin).
UV := $(shell command -v uv 2>/dev/null || echo $(HOME)/.local/bin/uv)

.PHONY: install install-train test lint format compile

install:          ## Create .venv if needed and install pinned deps
	$(UV) venv .venv --python 3.12 --allow-existing
	$(UV) pip install -r requirements.txt

install-train:    ## Add training deps (torch, transformers). Never on the runtime path.
	$(UV) pip install -r requirements-train.txt

test:             ## Run the test suite
	$(UV) run pytest

lint:             ## Check style and formatting (no writes)
	$(UV) run ruff check .
	$(UV) run ruff format --check .

format:           ## Apply formatting and safe autofixes
	$(UV) run ruff format .
	$(UV) run ruff check --fix .

compile:          ## Re-pin both requirement sets; train is constrained by runtime
	$(UV) pip compile requirements.in -o requirements.txt
	$(UV) pip compile requirements-train.in -c requirements.txt -o requirements-train.txt
