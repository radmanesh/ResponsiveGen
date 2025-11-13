.PHONY: install install-pip install-mamba run-app run-cli test format lint clean help

help:
	@echo "Available targets:"
	@echo "  install        - Install dependencies using Poetry"
	@echo "  install-pip    - Install dependencies using pip"
	@echo "  install-mamba  - Install dependencies using mamba/conda"
	@echo "  run-app        - Launch Streamlit web interface"
	@echo "  run-cli        - Show CLI help"
	@echo "  test           - Run test suite"
	@echo "  format         - Format code with black"
	@echo "  lint           - Lint code with ruff"
	@echo "  clean          - Remove generated files and caches"
	@echo "  playwright     - Install Playwright browsers"

install:
	poetry install
	poetry run playwright install chromium

install-pip:
	pip install -r requirements.txt
	playwright install chromium

install-mamba:
	mamba env create -f environment.yml
	@echo "Now run: mamba activate responsive && playwright install chromium"

run-app:
	poetry run streamlit run streamlit_app.py

run-cli:
	poetry run python cli.py --help

test:
	poetry run pytest tests/ -v

format:
	poetry run black responsive_gen/ cli.py streamlit_app.py
	poetry run ruff check --fix responsive_gen/ cli.py streamlit_app.py

lint:
	poetry run ruff check responsive_gen/ cli.py streamlit_app.py
	poetry run mypy responsive_gen/ cli.py streamlit_app.py

playwright:
	poetry run playwright install chromium

clean:
	rm -rf outputs/
	rm -rf .cache/
	rm -rf __pycache__/
	rm -rf **/__pycache__/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

