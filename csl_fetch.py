#!/usr/bin/env python3
"""Compatibility entrypoint. Prefer uv run --locked -m csl_validation fetch."""
import sys
from csl_validation import fetch

if __name__ == "__main__":
    raise SystemExit(fetch.main())
sys.modules[__name__] = fetch
