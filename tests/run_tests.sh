#!/bin/bash
# Runs the whole suite. Needs python3 + numpy only: no HAT, no amixer, no model —
# tests/fake_capture.py stands in for arecord and the classifier is faked.
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python3 -m unittest discover -s tests -v "$@"
