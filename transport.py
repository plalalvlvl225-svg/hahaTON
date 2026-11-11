"""Construction of a feasible transportation plan."""


def solve(supply, demand, costs):
    """Fill each supplier's row until the outstanding demand is exhausted."""
    plan = [{} for _ in supply]
    remaining = demand.copy()
    j = 0
    for i, available in enumerate(supply):
        while available:
            while remaining[j] == 0:
                j += 1
            quantity = min(available, remaining[j])
            plan[i][j] = quantity
            available -= quantity
            remaining[j] -= quantity
    return plan
