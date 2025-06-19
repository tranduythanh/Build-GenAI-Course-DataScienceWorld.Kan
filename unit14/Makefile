.PHONY: help install test run

help:
	@echo "Unit13 Commands:" && \\
		echo "  install  Install dependencies via uv" && \\
		echo "  test     Run unit tests" && \\
		echo "  run      Run CLI app"

install:
	uv pip install -r requirements.txt

test:
	.venv/bin/python -m unittest discover -v tests

run:
	.venv/bin/python app.py
