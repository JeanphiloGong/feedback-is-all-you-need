"""Freeze Study G development and formal environments before model calls."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.study_b.tasks import make_environment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    environments = [make_environment(seed) for seed in [19, *range(9000, 9048)]]
    for environment in environments:
        del environment["corrupted"]
    payload = (json.dumps(dict(numpy_version=np.__version__, environments=environments),
                          sort_keys=True) + "\n").encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write(payload)
    print(hashlib.sha256(payload).hexdigest())


if __name__ == "__main__":
    main()
