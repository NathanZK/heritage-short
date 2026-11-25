PY ?= python3
PIPELINE_FLAGS ?=
PYTEST_ARGS ?=

film:
	$(PY) -m pipeline.run $(PIPELINE_FLAGS)

smoke:
	$(PY) -m pipeline.smoke $(PIPELINE_FLAGS)

kaggle-setup:
	$(PY) -m pipeline.kaggle setup

install:
	$(PY) -m pip install -r requirements.txt

install-dev:
	$(PY) -m pip install -r requirements-dev.txt

test:
	$(PY) -m pytest -q $(PYTEST_ARGS)

.PHONY: film smoke kaggle-setup install install-dev test
