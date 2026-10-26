"""Assign disjoint stream/probe tasks and tabulate the existing local prior."""

from itertools import product


STREAM_INDICES = (0, 8, 2, 10, 4, 12, 6, 14)
PROBE_INDICES = (1, 9, 3, 11, 5, 13, 7, 15)
ORDER_FAMILIES = {
    'composition_first': ('composition', 'shifted'),
    'shifted_first': ('shifted', 'composition'),
}


def task_roles(environment):
    tasks = environment['tasks']
    streams = {order: [(family, tasks[family][index])
                       for family in families for index in STREAM_INDICES]
               for order, families in ORDER_FAMILIES.items()}
    probes = [(family, tasks[family][index])
              for family in ('composition', 'shifted') for index in PROBE_INDICES]
    return streams, probes


def local_probability_table(model):
    model.eval()
    return {context: model.probabilities(context)
            for length in range(3) for context in product(range(4), repeat=length)}
