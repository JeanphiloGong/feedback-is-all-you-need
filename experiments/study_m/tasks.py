"""Finite action worlds with hidden transition maps and unused goal cues."""

import argparse
import hashlib
import json
from pathlib import Path

from experiments.study_h.tasks import rng


STATES = 32
ACTIONS = 4
FAMILIES = ('structured', 'random')
DEVELOPMENT_SEED = 43
FORMAL_SEEDS = tuple(range(15000, 15032))


def structured_map(permutation, mask):
    return [sum(((((state >> source) & 1) ^ ((mask >> target) & 1)) << target)
                for target, source in enumerate(permutation)) for state in range(STATES)]


def strongly_connected(primitives):
    for start in range(STATES):
        seen, pending = {start}, [start]
        while pending:
            state = pending.pop()
            for action in primitives:
                following = action[state]
                if following not in seen:
                    seen.add(following)
                    pending.append(following)
        if len(seen) != STATES:
            return False
    return True


def make_world(seed, family):
    source = rng(seed, f'study_m:{family}:maps')
    rejections = dict(worlds=0, actions=0)
    while True:
        primitives, permutations, masks = [], [], []
        while len(primitives) < ACTIONS:
            if family == 'structured':
                permutation, mask = source.sample(range(5), 5), source.randrange(STATES)
                candidate = structured_map(permutation, mask)
            else:
                candidate = source.sample(range(STATES), STATES)
            if candidate == list(range(STATES)) or candidate in primitives:
                rejections['actions'] += 1
                continue
            primitives.append(candidate)
            if family == 'structured':
                permutations.append(permutation)
                masks.append(mask)
        if strongly_connected(primitives):
            break
        rejections['worlds'] += 1
    goal_rng = rng(seed, f'study_m:{family}:goals')
    held_out = sorted(goal_rng.sample(range(STATES), 8))
    training = [state for state in range(STATES) if state not in held_out]
    episode_rng = rng(seed, f'study_m:{family}:episodes')
    episodes = []
    for _ in range(513):
        goal = episode_rng.choice(training)
        start = episode_rng.choice([state for state in range(STATES) if state != goal])
        episodes.append(dict(start=start, goal=goal))
    warmup_rng = rng(seed, f'study_m:{family}:warmup')
    world = dict(seed=seed, family=family, primitives=primitives, rejections=rejections,
        held_out_goals=held_out, training_goals=training, episodes=episodes,
        warmup=dict(start=warmup_rng.randrange(STATES),
                    actions=[warmup_rng.randrange(ACTIONS) for _ in range(16)]),
        probes=[dict(start=start, goal=goal) for goal in held_out
                for start in range(STATES) if start != goal])
    if family == 'structured':
        world.update(bit_permutations=permutations, masks=masks)
    return world


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    worlds = [make_world(seed, family) for seed in (DEVELOPMENT_SEED,) + FORMAL_SEEDS
              for family in FAMILIES]
    with args.output.open('x') as stream:
        stream.write(json.dumps(worlds, indent=2) + '\n')
    print(hashlib.sha256(args.output.read_bytes()).hexdigest())


if __name__ == '__main__':
    main()
