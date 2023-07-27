ifneq (,$(wildcard ./.env))
    include .env
    export
endif

POETRY = poetry
MODELS_PATH ?= models

BINARIES_NAMES := $(addsuffix .pkl,train valid test enc)
preprocess_binaries := $(addprefix $(MODELS_PATH)/,$(BINARIES_NAMES))

.PHONY: data train register all

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
	docker build -t mushroom-classification .

lint:
	pylint --recursive=y src/

format:
	isort src/
	black src/
