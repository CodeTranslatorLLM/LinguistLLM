# Makefile

# Variables
PYTHON := python3
PIP := pip

# Default target
all: install run_train

# Install dependencies
install:
	$(PIP) install unsloth "xformers==0.0.28.post2"
	$(PIP) uninstall unsloth -y && $(PIP) install --upgrade --no-cache-dir "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Run the main training Python code
run_train:
	$(PYTHON) train.py

run_inference:
	$(PYTHON) inference.py

# Clean target
clean:
	rm -rf __pycache__

