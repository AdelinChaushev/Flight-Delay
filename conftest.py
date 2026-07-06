"""Pytest configuration: make the project root (and thus ``src/``) importable
regardless of where pytest is launched from."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
