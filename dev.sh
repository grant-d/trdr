#!/bin/bash

# Activate virtual environment if it exists and not already activated
if [ -z "$VIRTUAL_ENV" ] && [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
    echo "Virtual environment activated: $VIRTUAL_ENV"
fi

# Lint all Python files
function lint() {
  echo "Running flake8..."
  flake8 .
}

# Format all Python files
function format() {
  echo "Running black..."
  black .
}

# Run all tests
function test() {
  echo "Running pytest..."
  pytest --maxfail=1 --disable-warnings -q
}

# Run the custom test runner script
function testcustom() {
  echo "Running custom test suite..."
  ./tests/run_tests.py
}

# Install requirements (uses venv pip)
function install() {
  echo "Installing requirements..."
  pip install -r requirements.txt -r requirements-dev.txt
}

# Run mypy for type checking
function typecheck() {
  echo "Running mypy..."
  mypy .
}

# Run a Python script with the venv
function run() {
  if [ $# -lt 2 ]; then
    echo "Usage: $0 run <script.py> [args...]"
    exit 1
  fi
  shift  # Remove 'run' from arguments
  echo "Running with venv Python: $@"
  python "$@"
}

if [ $# -eq 0 ]; then
  echo "Usage: $0 {lint|format|test|install|typecheck|tc|run}"
  exit 1
fi

case "$1" in
  lint)
    lint
    ;;
  format)
    format
    ;;
  test)
    test
    ;;
  testcustom)
    testcustom
    ;;
  install)
    install
    ;;
  typecheck)
    typecheck
    ;;
  tc)
    typecheck
    ;;
  run)
    run "$@"
    ;;
  *)
    echo "Usage: $0 {lint|format|test|install|typecheck|tc|testcustom|run}"
    ;;
esac
