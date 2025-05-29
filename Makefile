CUR_DIR := ${CURDIR}
OS := $(shell uname)
CORES ?= all

.PHONY: clean_snakemake
clean_snakemake:
	@echo "cleaning up snakemake..."
	@rm -rf ${CUR_DIR}/.snakemake
	@rm -rf ${CUR_DIR}/data
	@rm -rf ${CUR_DIR}/dag.pdf

.PHONY: clean
clean: clean_snakemake
	@echo "cleaning up..."
	@(unset VIRTUAL_ENV || true; poetry env remove --all)
	@rm -rf ${CUR_DIR}/.venv
	@rm -rf ${CUR_DIR}/core.*


.PHONY: set_pyenv
set_pyenv:
	@echo "delete existing venv"
	@sudo rm -rf /opt/venv /usr/bin/python
	@echo "setting up pyenv..."
	@pyenv install 3.11.12 -s

.PHONY: setup
setup: set_pyenv
ifeq ($(OS),Darwin)
	@echo "Mac"
else
	@echo "Linux"
	# python3 -m keyring --disable
endif
	export POETRY_INSTALLER_MAX_WORKERS=5
	export POETRY_REQUESTS_MAX_RETRIES=10
	@(unset VIRTUAL_ENV || true; poetry config virtualenvs.create true; poetry config virtualenvs.in-project true; poetry install)

.PHONY: jupyter
jupyter:
	@echo "starting jupyter..."
	@(unset VIRTUAL_ENV || true; poetry run jupyter notebook --no-browser)

.PHONY: format
format:
	@echo "formatting..."
	@(unset VIRTUAL_ENV || true; poetry install)
	# add isort
	@poetry run isort .
	@poetry run black .

.PHONY: cs
cs: clean_snakemake

# to run this on kaggle, run: make snakemake DATA_PATH=/kaggle/input SCRIPT_PATH=scripts
.PHONY: snakemake
snakemake: clean_snakemake
	@make setup
	@echo "running snakemake with $(CORES) cores...for $(KFOLD) fold and $(TRIALS) trials"
	@(unset VIRTUAL_ENV || true; poetry run snakemake all --cores $(CORES) --config base_data_path=$(DATA_PATH) base_script_path=$(SCRIPT_PATH) kfold=$(KFOLD) trials=$(TRIALS) --nolock --ignore-incomplete)

.PHONY: snakemake_kaggle
snakemake_kaggle:
	snakemake --cores all --config base_script_path=gwi/scripts base_data_path=/kaggle/input/waveform-inversion kfold=10 trials=1 --nolock --ignore-incomplete
