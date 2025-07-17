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

# Install requirements (venv must be active)
function install() {
  echo "Installing requirements..."
  .venv/bin/pip install -r requirements.txt -r requirements-dev.txt
}

# Install requirements with uv (if preferred)
function uv_install() {
  echo "Installing requirements with uv..."
  uv pip install -r requirements.txt -r requirements-dev.txt
}

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
  install)
    install
    ;;
  uv_install)
    uv_install
    ;;
  *)
    echo "Usage: $0 {lint|format|test|install|uv_install}"
    ;;
esac
