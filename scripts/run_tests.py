#!/usr/bin/env python3
"""Test runner script with different test categories."""

import sys
import subprocess
import argparse
from pathlib import Path


def run_command(cmd, description):
    """Run a command and report results."""
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

        if result.returncode == 0:
            print("✅ PASSED")
            if result.stdout:
                print(result.stdout)
        else:
            print("❌ FAILED")
            if result.stderr:
                print("STDERR:", result.stderr)
            if result.stdout:
                print("STDOUT:", result.stdout)

        return result.returncode == 0

    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False


def main():
    """Run tests based on command line arguments."""
    parser = argparse.ArgumentParser(description="Run RAG system tests")
    parser.add_argument(
        "--category",
        choices=["unit", "api", "integration", "all"],
        default="all",
        help="Test category to run"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output"
    )

    args = parser.parse_args()

    print("RAG System Test Runner")
    print("=" * 40)

    # Base pytest command
    pytest_cmd = "pytest"
    if args.verbose:
        pytest_cmd += " -v"

    success = True

    if args.category in ["unit", "all"]:
        print(f"\n🧪 Running Unit Tests...")

        # Data loader tests
        cmd = f"{pytest_cmd} ../tests/test_data_loader.py"
        success &= run_command(cmd, "Data Loader Tests")

        # Embedding generator tests
        cmd = f"{pytest_cmd} ../tests/test_embedding_generator.py"
        success &= run_command(cmd, "Embedding Generator Tests")

        # Vector store tests
        cmd = f"{pytest_cmd} ../tests/test_vector_store.py"
        success &= run_command(cmd, "Vector Store Tests")

    if args.category in ["api", "all"]:
        print(f"\n🌐 Running API Tests...")

        # API tests
        cmd = f"{pytest_cmd} ../tests/test_api.py"
        success &= run_command(cmd, "FastAPI Tests")

    if args.category in ["integration", "all"]:
        print(f"\n🔄 Running Integration Tests...")

        # Full integration test
        cmd = f"python scripts/demo_step_by_step.py"
        print("\nNote: Integration test requires manual input. Skipping in automated run.")
        print("To run integration test manually: python scripts/demo_step_by_step.py")

    print(f"\n{'='*60}")
    if success:
        print("🎉 ALL TESTS PASSED!")
        exit_code = 0
    else:
        print("❌ SOME TESTS FAILED!")
        exit_code = 1

    print(f"{'='*60}")

    print("\n📋 Test Categories Available:")
    print("  --category unit        : Run unit tests only")
    print("  --category api         : Run API tests only")
    print("  --category integration : Run integration tests")
    print("  --category all         : Run all tests (default)")
    print("\n💡 Tips:")
    print("  - Use -v for verbose output")
    print("  - API tests may take longer due to embedding generation")
    print("  - Make sure Azure authentication is working")

    return exit_code


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)