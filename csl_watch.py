#!/usr/bin/env python3
"""Compatibility entrypoint. Prefer uv run --locked -m csl_validation watch."""
import sys
from csl_validation import watch

if __name__ == "__main__":
    raise SystemExit(watch.main())
sys.modules[__name__] = watch
