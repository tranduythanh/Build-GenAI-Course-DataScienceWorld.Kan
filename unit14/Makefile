.PHONY: help install test run

help:
	@echo "Unit14 Commands:" && echo "  install  Install dependencies via Poetry" && echo "  test     Run unit tests" && echo "  run      Run Streamlit app"

install:
	poetry config virtualenvs.in-project true
	poetry install --no-root

test:
	PYTHONPATH=$(PWD)/.. poetry run python -m unittest discover -v tests

run:
	poetry run streamlit run app.py
