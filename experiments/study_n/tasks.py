"""Isomorphic encodings with paired, balanced, disjoint transition queries."""

import argparse
import json
from pathlib import Path

from experiments.study_h.tasks import rng
from experiments.study_m.tasks import structured_map, strongly_connected


DEVELOPMENT_SEED = 47
FORMAL_SEEDS = tuple(range(16000, 16032))
ENCODINGS = ('canonical', 'relabeled')
CHECKPOINTS = (0, 16, 32, 64)


def make_world(seed):
    source = rng(seed, 'study_n:maps')
    rejected = dict(actions=0, worlds=0)
    while True:
        maps, permutations, masks = [], [], []
        while len(maps) < 4:
            permutation, mask = source.sample(range(5), 5), source.randrange(32)
            action = structured_map(permutation, mask)
            if action == list(range(32)) or action in maps:
                rejected['actions'] += 1
                continue
            maps.append(action)
            permutations.append(permutation)
            masks.append(mask)
        if strongly_connected(maps):
            break
        rejected['worlds'] += 1
    labels = rng(seed, 'study_n:labels').sample(range(32), 32)
    sources = [rng(seed, f'study_n:sources:{a}').sample(range(32), 32) for a in range(4)]
    order = rng(seed, 'study_n:query_order')
    queries = [[sources[a][i], a] for i in range(16) for a in order.sample(range(4), 4)]
    selected = set(map(tuple, queries))
    test = [[s, a] for a in range(4) for s in range(32) if (s, a) not in selected]
    return dict(seed=seed, primitives=maps, bit_permutations=permutations,
                masks=masks, rejections=rejected, labels=labels, queries=queries, test=test)


def conjugate(graph, labels):
    """Change only the state labels, including absent edges of partial graphs."""
    result = [[None] * 32 for _ in range(4)]
    for action in range(4):
        for state in range(32):
            following = graph[action][state]
            result[action][labels[state]] = labels[following] if following is not None else None
    return result


def signed_candidates(observations):
    """A family-informed diagnostic, given only canonical observed triples."""
    candidates = [[[(bit, sign) for bit in range(5) for sign in (0, 1)]
                   for _ in range(5)] for _ in range(4)]
    comparisons = 0
    for state, action, following in observations:
        for output in range(5):
            options = candidates[action][output]
            comparisons += len(options)
            candidates[action][output] = [(bit, sign) for bit, sign in options
                                          if ((state >> bit & 1) ^ sign) == (following >> output & 1)]
    return candidates, comparisons


def signed_prediction(candidates, state, action):
    result = 0
    for output, options in enumerate(candidates[action]):
        values = {(state >> bit & 1) ^ sign for bit, sign in options}
        if len(values) != 1:
            return None
        result |= values.pop() << output
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    worlds = [make_world(s) for s in (DEVELOPMENT_SEED, *FORMAL_SEEDS)]
    with args.output.open('x') as stream:
        stream.write(json.dumps(worlds, indent=2) + '\n')


if __name__ == '__main__':
    main()
