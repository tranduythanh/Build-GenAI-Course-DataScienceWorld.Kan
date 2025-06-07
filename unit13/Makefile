.PHONY: help install test run

help:
	@echo "Unit13 Commands:" && \
		echo "  install  Install dependencies via Poetry" && \
		echo "  test     Run unit tests" && \
		echo "  run      Run CLI app"

install:
	poetry config virtualenvs.in-project true
	poetry install --no-root

test:
	poetry run python -m unittest discover -v tests

run:
	poetry run python app.py
