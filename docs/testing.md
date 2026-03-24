# Testing

See [testing_guide.md](testing_guide.md) for the full testing guide.

Quick reference:

```bash
# Install test dependencies
uv sync --group dev

# Run all tests
python scripts/run_tests.py

# Run specific categories
python scripts/run_tests.py --category unit -v
python scripts/run_tests.py --category api -v

# Run individual test files
pytest tests/test_data_loader.py -v
pytest tests/test_api.py -v
```

All tests require Azure authentication:
```bash
azd auth login --scope api://ailab/Model.Access
```