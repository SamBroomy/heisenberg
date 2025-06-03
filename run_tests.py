#!/usr/bin/env python3
# filepath: /Users/broomfis/src/rust/heisenberg/run_tests.py
"""
Comprehensive test runner for the Heisenberg location searcher.

This script runs both Rust unit tests and Python integration tests,
providing a complete overview of the system's health.
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(command, description, cwd=None):
    """Run a command and return success/failure."""
    print(f"\n{'=' * 60}")
    print(f"Running: {description}")
    print(f"Command: {command}")
    print(f"{'=' * 60}")

    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
        )

        if result.stdout:
            print("STDOUT:")
            print(result.stdout)

        if result.stderr:
            print("STDERR:")
            print(result.stderr)

        if result.returncode == 0:
            print(f"✅ {description} PASSED")
            return True
        else:
            print(f"❌ {description} FAILED (exit code: {result.returncode})")
            return False

    except subprocess.TimeoutExpired:
        print(f"⏰ {description} TIMED OUT")
        return False
    except Exception as e:
        print(f"💥 {description} ERROR: {e}")
        return False


def main():
    """Run all tests."""
    print("🧪 Heisenberg Location Searcher - Comprehensive Test Suite")
    print("=" * 60)

    # Get the project root directory
    project_root = Path(__file__).parent
    os.chdir(project_root)

    results = []

    # 1. Run Rust unit tests
    results.append(
        run_command(
            "cargo test --lib", "Rust Unit Tests (Core Functionality)", cwd=project_root
        )
    )

    # 2. Run Rust integration tests
    results.append(
        run_command("cargo test --tests", "Rust Integration Tests", cwd=project_root)
    )

    # 3. Build Python wheel to ensure compilation works
    results.append(
        run_command("maturin develop", "Python Extension Build", cwd=project_root)
    )

    # 4. Run Python basic tests
    results.append(
        run_command(
            "python -m pytest python/tests/test_basic.py -v",
            "Python Basic Integration Tests",
            cwd=project_root,
        )
    )

    # 5. Run Python integration tests
    results.append(
        run_command(
            "python -m pytest python/tests/test_integration.py -v",
            "Python Comprehensive Integration Tests",
            cwd=project_root,
        )
    )

    # 6. Run Python edge case tests
    results.append(
        run_command(
            "python -m pytest python/tests/test_edge_cases.py -v",
            "Python Edge Case Tests",
            cwd=project_root,
        )
    )

    # 7. Run all Python tests together
    results.append(
        run_command(
            "python -m pytest python/tests/ -v --tb=short",
            "All Python Tests Combined",
            cwd=project_root,
        )
    )

    # 8. Check code formatting (if available)
    results.append(
        run_command("cargo fmt --check", "Rust Code Formatting Check", cwd=project_root)
    )

    # 9. Run clippy for additional Rust linting (if available)
    results.append(
        run_command(
            "cargo clippy -- -D warnings", "Rust Clippy Linting", cwd=project_root
        )
    )

    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)

    passed = sum(results)
    total = len(results)

    test_names = [
        "Rust Unit Tests",
        "Rust Integration Tests",
        "Python Extension Build",
        "Python Basic Tests",
        "Python Integration Tests",
        "Python Edge Case Tests",
        "All Python Tests",
        "Rust Formatting Check",
        "Rust Clippy Linting",
    ]

    for i, (name, result) in enumerate(zip(test_names, results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{i + 1:2d}. {name:<30} {status}")

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! The system is healthy.")
        sys.exit(0)
    else:
        print("⚠️  Some tests failed. Please review the output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
