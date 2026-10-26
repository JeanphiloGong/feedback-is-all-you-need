"""Plan only through a supplied predicted graph or observed-edge graph."""

from collections import deque


def choose_action(graph, state, goal, fallback):
    """The interaction loop calls this only before reaching the goal."""
    expanded = edges = 0
    if graph is not None:
        pending, seen = deque([(state, ())]), {state}
        while pending:
            current, path = pending.popleft()
            if len(path) == 8:
                continue
            expanded += 1
            for action in range(4):
                edges += 1
                following = graph[action][current]
                if following is None or following in seen:
                    continue
                candidate = path + (action,)
                if following == goal:
                    return dict(action=candidate[0], planned_path=list(candidate),
                                expanded_states=expanded, edge_checks=edges, fallback=False)
                seen.add(following)
                pending.append((following, candidate))
    return dict(action=fallback, planned_path=[], expanded_states=expanded,
                edge_checks=edges, fallback=True)
