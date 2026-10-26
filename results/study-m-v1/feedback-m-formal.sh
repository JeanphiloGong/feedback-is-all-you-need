#!/bin/bash
set -eu
cd /tmp/feedback-research-m/frozen-13f1c71/repo
export PYTHONDONTWRITEBYTECODE=1
/src/users/lens/feedback-is-all-you-need/.venv/bin/python -m experiments.study_m.run --source-revision 13f1c71c991ed991507f8ccc66609d4e5b88ec04 --development-audit /tmp/feedback-research-m/development-audit-v1/audit.json --output /tmp/feedback-research-m/formal-v1 > /tmp/feedback-research-m/formal-console.log 2>&1
/src/users/lens/feedback-is-all-you-need/.venv/bin/python -m experiments.study_m.analyze --input /tmp/feedback-research-m/formal-v1 --output /tmp/feedback-research-m/formal-audit-v1 > /tmp/feedback-research-m/formal-audit-console.log 2>&1
