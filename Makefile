ifneq (,$(wildcard ./.env))
    include .env
    export
endif

POETRY = poetry
DOCKER = docker
MODELS_PATH ?= models
PYTHON_FILES = $(shell git ls-files '*.py')

BINARIES_NAMES := $(addsuffix .pkl,train valid test enc)
preprocess_binaries := $(addprefix $(MODELS_PATH)/,$(BINARIES_NAMES))

.PHONY: data train register all setup build lint format tests

all: build

data: $(preprocess_binaries)

train: src/models/hpo.py $(wordlist 1, 3, $(preprocess_binaries))
	$(POETRY) run python $< --train-data $(word 1, $(preprocess_binaries)) --val-data $(word 2, $(preprocess_binaries))

register: src/models/register.py $(word 3, $(preprocess_binaries))
	$(POETRY) run python $< -i $(word 2, $^)


$(preprocess_binaries) &: src/data/preprocess.py
	mkdir -p $(MODELS_PATH)
	$(POETRY) run python $< --input-path data/mushrooms.csv --output-directory $(MODELS_PATH)

setup:
	$(POETRY) install --no-dev

build:
	$(DOCKER) build -t mushroom-classification .

lint:
	ruff check .
	mypy .
	pylint $(PYTHON_FILES)

format:
	isort .
	black .

tests:
	pytest tests/

coverage:
	coverage run -m pytest
	coverage report
