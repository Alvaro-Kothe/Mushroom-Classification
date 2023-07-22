POETRY=poetry
BINARIES_PATH=binaries

BINARIES_NAMES:=$(addsuffix .pkl,train valid test enc)
preprocess_binaries:=$(addprefix $(BINARIES_PATH)/,$(BINARIES_NAMES))

.PHONY: data train register

data: $(preprocess_binaries)

train: src/models/hpo.py $(wordlist 1, 3, $(preprocess_binaries))
	$(POETRY) run python $< --train-data $(word 1, $(preprocess_binaries)) --val-data $(word 2, $(preprocess_binaries))

register: src/models/register.py $(word 3, $(preprocess_binaries))
	$(POETRY) run python $< -i $(word 2, $^)


$(preprocess_binaries) &: src/data/preprocess.py
	mkdir -p $(BINARIES_PATH)
	$(POETRY) run python $< --input-path data/mushrooms.csv --output-directory $(BINARIES_PATH)

setup:
	$(POETRY) install --no-dev
