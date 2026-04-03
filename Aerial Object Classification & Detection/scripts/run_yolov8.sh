#!/usr/bin/env bash
set -e

PROJECT_ROOT="/content/drive/MyDrive/Aerial_Object_Classification_Detection"
cd "$PROJECT_ROOT"

python -m pytest tests/test_detection_labels.py -q