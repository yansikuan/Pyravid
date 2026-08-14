# Contributing

1. Create a focused branch and keep generated artifacts out of Git.
2. Install development dependencies with `python -m pip install -e ".[dev]"`.
3. Run `ruff check .`, `python -m unittest discover -s tests -v`, and
   `python scripts/scan_sensitive.py` before opening a pull request.
4. Add tests for behavioral changes. Tests must run offline unless explicitly
   marked as integration tests.
5. Never commit credentials, private endpoints, personal filesystem paths,
   restricted datasets, model weights, or raw experiment logs.

Experimental variants should be expressed as configuration where possible.
When a new implementation is necessary, use a descriptive strategy name rather
than a numeric suffix such as `v2` or `v3`.
