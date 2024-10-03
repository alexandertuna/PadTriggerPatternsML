#################################################################################
# Taken from cookiecutter-data-science and adapted
#################################################################################

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = pads_ml
PYTHON_VERSION = 3.11
PYTHON_INTERPRETER = python3.11

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python Dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt


## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using flake8 and black (use `make format` to do formatting)
.PHONY: lint
lint:
	flake8 pads_ml
	isort --check --diff --profile black pads_ml
	black --check --config pyproject.toml pads_ml


## Format source code with black
.PHONY: format
format:
	black --config pyproject.toml pads_ml


#################################################################################
# Grabbing latest files                                                         #
#################################################################################

LATEST_MODEL := $(shell ls -1 model.*.pt | tail -n 1)
LATEST_FEATURES := $(shell ls -1 train/*features.npy | tail -n 1)
LATEST_LABELS := $(shell ls -1 train/*labels.npy | tail -n 1)


#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

.PHONY: generate_random
generate_random:
	rm -f gen.*
	rm -f train/gen.*
	rm -f valid/gen.*
	for i in {0..9}; do $(PYTHON_INTERPRETER) scripts/generate.py -n 100_000 --onehot --background random; done
	mv gen.* train/
	for i in {0..1}; do $(PYTHON_INTERPRETER) scripts/generate.py -n 100_000 --onehot --background random; done
	mv gen.* valid/


.PHONY: generate_smear
generate_smear:
	rm -f gen.*
	rm -f train/gen.*
	rm -f valid/gen.*
	for i in {0..9}; do $(PYTHON_INTERPRETER) scripts/generate.py -n 100_000 --onehot --background smear; done
	mv gen.* train/
	for i in {0..1}; do $(PYTHON_INTERPRETER) scripts/generate.py -n 100_000 --onehot --background smear; done
	mv gen.* valid/


.PHONY: draw_signal
draw_signal:
	$(PYTHON_INTERPRETER) scripts/generate.py -n 1_000_000
	$(PYTHON_INTERPRETER) scripts/draw_signal.py -i $(shell ls -1 gen.*signal.parquet | tail -n 1)
	open draw_signal.pdf


.PHONY: train
train:
	$(PYTHON_INTERPRETER) scripts/train_network.py

.PHONY: inference
inference:
	$(PYTHON_INTERPRETER) scripts/do_inference.py -m $(LATEST_MODEL) -f $(LATEST_FEATURES) -l $(LATEST_LABELS)


## Make Dataset
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) pads_ml/dataset.py


