#!/bin/bash
set -eu
cd /tmp/feedback-research-m/frozen-13f1c71/repo
export PYTHONDONTWRITEBYTECODE=1
/src/users/lens/feedback-is-all-you-need/.venv/bin/python -m unittest discover -s experiments/study_m -p 'test_*.py' -v > /tmp/feedback-research-m/frozen-tests.log 2>&1
/src/users/lens/feedback-is-all-you-need/.venv/bin/python -m experiments.study_m.run --source-revision 13f1c71c991ed991507f8ccc66609d4e5b88ec04 --development --output /tmp/feedback-research-m/development-v1 > /tmp/feedback-research-m/development-console.log 2>&1
/src/users/lens/feedback-is-all-you-need/.venv/bin/python -m experiments.study_m.analyze --input /tmp/feedback-research-m/development-v1 --output /tmp/feedback-research-m/development-audit-v1 > /tmp/feedback-research-m/development-audit-console.log 2>&1
