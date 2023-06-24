.PHONY: install
install:
	@if ! command -v pyenv &> /dev/null; then \
		echo "Error: pyenv is not installed. Visit https://github.com/pyenv/pyenv#installation for installation instructions."; \
		exit 1; \
	fi
	@if ! command -v poetry &> /dev/null; then \
		echo "Error: poetry is not installed. Visit https://python-poetry.org/docs/#installation for installation instructions."; \
		exit 1; \
	fi
	@if [ -z "$$(pyenv versions | grep '3\.10\..*')" ]; then \
		pyenv install 3.10; \
	else \
		echo "Python 3.10 is already installed."; \
	fi
	poetry env use 3.10
	poetry install
	poetry run pre-commit install
	@SITE_PACKAGES_DIR=$$(python -c "import sys, os; print([path for path in sys.path if '.venv' in path and 'site-packages' in path][0])"); \
	PROJECT_ROOT=$$(echo $${SITE_PACKAGES_DIR} | rev | cut -d'/' -f5- | rev); \
	echo $${PROJECT_ROOT} > $${SITE_PACKAGES_DIR}/tsuumo.pth

.PHONY: run
run:
	python src/main.py

.PHONY: pre-commit
pre-commit:
	poetry run pre-commit run --all-files

.PHONY: test
test:
	poetry run pytest -s -vv --cov=. --cov-branch --cov-report=html
