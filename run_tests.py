#!/usr/bin/env python3
"""
Test runner script for data cleaning functionality.
"""

import subprocess
import sys
import os
from pyparsing import Literal


def run_command(cmd: str, description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n{'='*50}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print(f"{'='*50}")

    result = subprocess.run(cmd, shell=True, capture_output=False)

    if result.returncode == 0:
        print(f"✓ {description} passed")
        return True
    else:
        print(f"✗ {description} failed")
        return False


def main() -> int:
    """Main test runner."""
    print("🧪 Data Cleaning Test Suite")
    print("=" * 50)

    # Change to script directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    tests = [
        ("python -m pytest tests/test_data_cleaning.py -v", "All data cleaning tests"),
        (
            "python -m pytest tests/test_data_cleaning.py::TestDataCleaning -v",
            "Core data cleaning tests",
        ),
        (
            "python -m pytest tests/test_data_cleaning.py::TestMissingValueHandling -v",
            "Missing value handling tests",
        ),
        (
            "python -m pytest tests/test_data_cleaning.py::TestOutlierDetection -v",
            "Outlier detection tests",
        ),
        (
            "python -m pytest tests/test_data_cleaning.py::TestOHLCVIntegrityValidation -v",
            "OHLCV integrity tests",
        ),
        (
            "python -m pytest tests/test_data_cleaning.py::TestDerivedFieldCalculation -v",
            "Derived field tests",
        ),
        (
            "python -m pytest tests/test_data_cleaning.py::TestEdgeCasesAndErrorHandling -v",
            "Edge case tests",
        ),
    ]

    results = []

    for cmd, description in tests:
        success = run_command(cmd, description)
        results.append((description, success))

    # Summary
    print(f"\n{'='*50}")
    print("🔍 Test Results Summary")
    print(f"{'='*50}")

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for description, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status} {description}")

    print(f"\n📊 Overall: {passed}/{total} test suites passed")

    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
