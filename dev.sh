#!/bin/bash

# Lint all Python files
function lint() {
  echo "Running flake8..."
  .venv/bin/flake8 .
}

# Format all Python files
function format() {
  echo "Running black..."
  .venv/bin/black .
}

# Run all tests
function test() {
  echo "Running pytest..."
  .venv/bin/pytest --maxfail=1 --disable-warnings -q
}

# Run the custom test runner script
function testcustom() {
  echo "Running custom test suite..."
  ./run_tests.py
}

# Install requirements (venv must be active)
function install() {
  echo "Installing requirements..."
  .venv/bin/pip install -r requirements.txt -r requirements-dev.txt
}

# Run mypy for type checking
function typecheck() {
  echo "Running mypy..."
  .venv/bin/mypy .
}

if [ $# -eq 0 ]; then
  echo "Usage: $0 {lint|format|test|install|uv_install|typecheck|tc|reqs}"
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
  *)
    echo "Usage: $0 {lint|format|test|install|typecheck|tc|testcustom}"
    ;;
esac
