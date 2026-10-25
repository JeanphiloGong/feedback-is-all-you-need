"""Frozen synthetic command environments and prompts for Study B."""

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

COMMANDS = ("dax", "wug", "zup", "fep")
SYSTEM = "You execute an unfamiliar command language. Follow the task description and return only one digit from 0 to 7."
INSTRUCTION = (
    "States are integers 0 through 7. Each command (dax, wug, zup, fep) adds "
    "its own fixed but unknown integer offset modulo 8. The four offsets are "
    "distinct and may include 0. Apply all commands to the starting state. "
    "Infer offsets from the examples if available. Return only the final state digit."
)


def execute(state, commands, offsets):
    for command in commands:
        state = (state + offsets[COMMANDS.index(command)]) % 8
    return int(state)


def make_environment(seed):
    offsets_rng, split_rng, compose_rng, noise_rng = [
        np.random.default_rng([seed, stream]) for stream in range(4)]
    offsets = offsets_rng.choice(8, 4, replace=False).tolist()
    train, direct = [], []
    for command in COMMANDS:
        states = split_rng.permutation(8).tolist()
        for i, state in enumerate(states):
            query = dict(state=state, commands=[command],
                         target=execute(state, [command], offsets))
            (train if i < 4 else direct).append(query)
    train = [train[i] for i in split_rng.permutation(len(train))]
    candidates = list(itertools.product(range(8), COMMANDS, COMMANDS))
    compose = []
    for i in compose_rng.choice(len(candidates), 32, replace=False):
        state, a, b = candidates[i]
        compose.append(dict(state=state, commands=[a,b],
                            target=execute(state,[a,b],offsets)))
    corrupted = []
    for query in train:
        corrupt = bool(noise_rng.random() < 0.5)
        wrong = int((query["target"] + noise_rng.integers(1,8)) % 8)
        corrupted.append(dict(query, target=wrong if corrupt else query["target"], corrupt=corrupt))
    return dict(seed=seed,offsets=offsets,train=train,direct=direct,
                compose=compose,corrupted=corrupted)


def query_text(query):
    return f"Start: {query['state']}; commands: {' '.join(query['commands'])}; result:"


def messages(query, history, offsets=None):
    text = INSTRUCTION
    if offsets is not None:
        text += "\nKnown offsets: " + "; ".join(f"{c} adds {v}" for c,v in zip(COMMANDS,offsets)) + "."
    if history:
        text += "\nExamples:\n" + "\n".join(f"{query_text(q)} {q['target']}" for q in history)
    text += "\nNow solve:\n" + query_text(query)
    return [{"role":"system","content":SYSTEM},{"role":"user","content":text}]


def symbolic_offsets(history):
    found = {}
    for query in history:
        command, = query["commands"]
        offset = (query["target"] - query["state"]) % 8
        if command in found:
            assert found[command] == offset
        found[command] = offset
    return [found[c] for c in COMMANDS]


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    data = dict(numpy_version=np.__version__,environments=[make_environment(s) for s in range(3000,3024)])
    with args.output.open("x") as stream:
        json.dump(data,stream,indent=2)
        stream.write("\n")
    print(hashlib.sha256(args.output.read_bytes()).hexdigest())
