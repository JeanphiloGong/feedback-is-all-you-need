"""Freeze fresh Study E environments before formal inference."""

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
    data = {
        "numpy_version": np.__version__,
        "seeds": list(range(7000, 7024)),
        "environments": [make_environment(seed) for seed in range(7000, 7024)],
    }
    payload = (json.dumps(data, sort_keys=True) + "\n").encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write(payload)
    print(hashlib.sha256(payload).hexdigest())


if __name__ == "__main__":
    main()
