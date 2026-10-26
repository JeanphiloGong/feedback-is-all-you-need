"""Controlled finite permutation tasks; hidden structure is generator-only."""

import argparse
import hashlib
from itertools import product
import json
from pathlib import Path
import random

from experiments.study_h.search import execute


DEVELOPMENT_SEED = 23
FORMAL_SEEDS = tuple(range(10000, 10032))


def rng(seed, role):
    payload = json.dumps([seed, role], separators=(",", ":")).encode()
    return random.Random(int.from_bytes(hashlib.sha256(payload).digest()[:8], "little"))


def signature(primitives, program):
    return tuple(execute(primitives, program, state) for state in range(16))


def make_environment(seed):
    primitive_rng = rng(seed, "primitive_tables")
    primitives = []
    primitive_rejections = 0
    while len(primitives) < 4:
        candidate = primitive_rng.sample(range(16), 16)
        if candidate == list(range(16)) or candidate in primitives:
            primitive_rejections += 1
            continue
        primitives.append(candidate)
    motifs = rng(seed, "latent_motifs").sample(list(product(range(4), repeat=3)), 4)
    wrapper_rng = rng(seed, "experience_wrappers")
    experience, seen, omitted = [], set(), 0
    for motif in motifs:
        for prefix, suffix in wrapper_rng.sample(list(product(range(4), repeat=2)), 6):
            program = (prefix,) + motif + (suffix,)
            function = signature(primitives, program)
            if function in seen:
                omitted += 1
                continue
            experience.append(program)
            seen.add(function)
    rng(seed, "experience_order").shuffle(experience)
    programs = {"experience": experience, "composition": [], "shifted": []}
    rejections = {"primitive_tables": primitive_rejections, "experience": omitted}
    for family in ("composition", "shifted"):
        source = rng(seed, family + "_programs")
        rejections[family] = 0
        for length in (6, 9):
            accepted = 0
            while accepted < 8:
                if family == "composition":
                    program = tuple(op for _ in range(length // 3) for op in source.choice(motifs))
                else:
                    program = tuple(source.choice(range(4)) for _ in range(length))
                function = signature(primitives, program)
                if function in seen:
                    rejections[family] += 1
                    continue
                programs[family].append(program)
                seen.add(function)
                accepted += 1
    tasks = {}
    for family, family_programs in programs.items():
        tasks[family] = []
        for index, program in enumerate(family_programs):
            states = rng(seed, f"inputs:{family}:{index}").sample(range(16), 16)
            examples = [(state, execute(primitives, program, state)) for state in states]
            tasks[family].append(dict(id=f"{family}-{index:02d}", generating_program=program,
                                      train=examples[:8], test=examples[8:]))
    return dict(seed=seed, primitives=primitives, latent_motifs=motifs,
                rejections=rejections, tasks=tasks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    environments = [make_environment(seed) for seed in (DEVELOPMENT_SEED,) + FORMAL_SEEDS]
    with args.output.open("x") as stream:
        stream.write(json.dumps(environments, indent=2) + "\n")
    print(hashlib.sha256(args.output.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
