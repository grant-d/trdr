# Python Environment Guide for This Project

## Current Setup
- **Virtual Environment**: `.venv` (Python 3.12.10)
- **Location**: `/Users/grantdickinson/repos/trdr/.venv`

## How to Activate the Virtual Environment

### Option 1: Auto-activation (Recommended)
Add this to your shell config (~/.zshrc or ~/.bash_profile):
```bash
# Auto-activate venv when entering project directory
cd() {
    builtin cd "$@"
    if [[ -f .venv/bin/activate ]]; then
        source .venv/bin/activate
    fi
}
```

### Option 2: Manual activation
Every time you open a new terminal in this project:
```bash
source .venv/bin/activate
```

## How to Check You're Using the Correct Environment

Run these commands:
```bash
which python   # Should show: /Users/grantdickinson/repos/trdr/.venv/bin/python
which pip      # Should show: /Users/grantdickinson/repos/trdr/.venv/bin/pip
```

## Installing Packages

### Always use one of these methods:
1. `python -m pip install <package>`  (Recommended - always uses the right pip)
2. `pip install <package>` (Only after verifying `which pip` shows venv path)

### Never use:
- System pip directly
- pip3 without checking which one

## Common Issues and Fixes

### Issue: "Module not found" but package is installed
**Cause**: Package installed in wrong environment
**Fix**: 
```bash
# Check active python
which python

# If not showing .venv path, activate venv:
source .venv/bin/activate

# Reinstall in correct environment:
python -m pip install -r requirements.txt
```

### Issue: pip installs to wrong location
**Fix**: Always use `python -m pip` instead of just `pip`

## Quick Commands

```bash
# Activate virtual environment
source .venv/bin/activate

# Install all requirements
python -m pip install -r requirements.txt

# Install a specific package
python -m pip install pymoo

# List installed packages
python -m pip list

# Check if a package is installed
python -m pip show pymoo
```

## VS Code / Cursor Setup

Make sure your editor is using the correct Python interpreter:
1. Open Command Palette (Cmd+Shift+P)
2. Type "Python: Select Interpreter"
3. Choose the interpreter at `./.venv/bin/python`