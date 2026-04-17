from __future__ import annotations


def simple_branch_loop(x: int) -> int:
    total = 0
    i = 0
    while i < x:
        if i < 2:
            total = total + i
        else:
            total = total + 2
        i = i + 1
    return total


def stack_decoder_host(costs: list[float], max_steps: int) -> float:
    frontier = [{"path": [0], "metric": 0.0, "depth": 0}]
    best_metric = 9999.0
    best_path = []
    steps = 0
    while steps < max_steps:
        if len(frontier) == 0:
            return best_metric
        best_index = 0
        scan = 1
        while scan < len(frontier):
            if frontier[scan]["metric"] < frontier[best_index]["metric"]:
                best_index = scan
            scan = scan + 1
        candidate = frontier.pop(best_index)
        score = candidate["metric"] + costs[candidate["depth"]]
        if score < best_metric:
            best_metric = score
            best_path = candidate["path"]
        next_depth = candidate["depth"] + 1
        if next_depth < len(costs):
            left = {"path": candidate["path"] + [0], "metric": score, "depth": next_depth}
            right = {"path": candidate["path"] + [1], "metric": score + 0.25, "depth": next_depth}
            frontier.append(left)
            frontier.append(right)
        steps = steps + 1
    return best_metric


def bp_summary_update(frontier: list[dict], costs: list[float], damping: float) -> float:
    idx = 0
    summary = 0.0
    while idx < len(frontier):
        item = frontier[idx]
        summary = summary + item["metric"] + costs[item["depth"]]
        idx = idx + 1
    return summary * damping
