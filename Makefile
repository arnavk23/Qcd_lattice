# Makefile for QCD Lattice Field Theory Project

# Python interpreter
PYTHON = python3

# Project directories
SRC_DIR = src
TEST_DIR = tests
NOTEBOOK_DIR = notebooks

# Virtual environment
VENV = .venv
VENV_PYTHON = $(VENV)/bin/python
VENV_PIP = $(VENV)/bin/pip

.PHONY: help setup install test demo clean notebook lint format

help:
	@echo "QCD Lattice Field Theory Project"
	@echo "================================"
	@echo ""
	@echo "Available targets:"
	@echo "  setup     - Create virtual environment and install dependencies"
	@echo "  install   - Install dependencies (assumes venv exists)"
	@echo "  test      - Run unit tests"
	@echo "  demo      - Run demonstration script"
	@echo "  notebook  - Start Jupyter notebook server"
	@echo "  lint      - Run code linting"
	@echo "  format    - Format code with black"
	@echo "  clean     - Clean up generated files"
	@echo "  help      - Show this help message"

setup: $(VENV)/bin/activate

$(VENV)/bin/activate: requirements.txt
	@echo "Creating virtual environment..."
	$(PYTHON) -m venv $(VENV)
	@echo "Installing dependencies..."
	$(VENV_PIP) install --upgrade pip
	$(VENV_PIP) install -r requirements.txt
	@echo "Setup complete! Activate with: source $(VENV)/bin/activate"

install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt

test:
	@echo "Running unit tests..."
	$(PYTHON) -m pytest $(TEST_DIR)/test_implementations.py -v

demo:
	@echo "Running demonstration script..."
	$(PYTHON) demo.py

notebook:
	@echo "Starting Jupyter notebook server..."
	@echo "Opening notebook: $(NOTEBOOK_DIR)/qcd_lattice_mcmc.ipynb"
	jupyter notebook $(NOTEBOOK_DIR)/qcd_lattice_mcmc.ipynb

lint:
	@echo "Running code linting..."
	@if command -v flake8 >/dev/null 2>&1; then \
		flake8 $(SRC_DIR) $(TEST_DIR) demo.py; \
	else \
		echo "flake8 not installed. Install with: pip install flake8"; \
	fi

format:
	@echo "Formatting code with black..."
	@if command -v black >/dev/null 2>&1; then \
		black $(SRC_DIR) $(TEST_DIR) demo.py; \
	else \
		echo "black not installed. Install with: pip install black"; \
	fi

clean:
	@echo "Cleaning up generated files..."
	rm -rf __pycache__
	rm -rf $(SRC_DIR)/__pycache__
	rm -rf $(TEST_DIR)/__pycache__
	rm -rf .pytest_cache
	rm -f *.png
	rm -f *.pdf
	rm -f *.log
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete

# Development targets
dev-setup: setup
	@echo "Installing development dependencies..."
	$(VENV_PIP) install pytest black flake8 jupyter

# Quick test of specific components
test-metropolis:
	@echo "Testing Metropolis implementation..."
	$(PYTHON) -c "from src.metropolis import MetropolisGaussian; print('✓ Metropolis import successful')"

test-field-theory:
	@echo "Testing Field Theory implementation..."
	$(PYTHON) -c "from src.field_theory_1d import FieldTheory1D; print('✓ Field Theory import successful')"

test-hmc:
	@echo "Testing HMC implementation..."
	$(PYTHON) -c "from src.hmc import HMCFieldTheory1D; print('✓ HMC import successful')"

test-imports: test-metropolis test-field-theory test-hmc
	@echo "All imports successful!"

# Project information
info:
	@echo "Project: QCD Lattice Field Theory Implementation"
	@echo "Description: Monte Carlo methods for lattice field theory"
	@echo "Author: Based on Pablo's plan and references"
	@echo ""
	@echo "Project structure:"
	@echo "  src/           - Source code implementations"
	@echo "  tests/         - Unit tests"
	@echo "  notebooks/     - Jupyter notebooks"
	@echo "  requirements.txt - Python dependencies"
	@echo "  demo.py        - Demonstration script"
	@echo ""
	@echo "Key references:"
	@echo "  - Quantum Chromodynamics on the Lattice (Chapters 1, 4)"
	@echo "  - Creutz article on Monte Carlo methods"
	@echo "  - MCMC for Dummies"

# Check if all dependencies are installed
check-deps:
	@echo "Checking dependencies..."
	@$(PYTHON) -c "import numpy; print('✓ NumPy:', numpy.__version__)"
	@$(PYTHON) -c "import matplotlib; print('✓ Matplotlib:', matplotlib.__version__)"
	@$(PYTHON) -c "import scipy; print('✓ SciPy:', scipy.__version__)"
	@$(PYTHON) -c "import tqdm; print('✓ tqdm:', tqdm.__version__)"
	@$(PYTHON) -c "import jupyter; print('✓ Jupyter:', jupyter.__version__)"
	@echo "All dependencies OK!"

# Run full project validation
validate: test-imports check-deps test demo
	@echo ""
	@echo "✓ Full project validation completed successfully!"
	@echo "The implementation is ready for use."
