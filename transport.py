"""Exact minimum-cost transportation for balanced integer margins."""

import numpy as np

from simplex import solve_simplex


def normalized_costs(costs):
    """Remove row and column offsets. Use fixed-width math only with headroom."""
    # Convert before subtraction: both endpoints may fit int64 while their
    # difference does not. Only the size of the reduced costs matters later.
    if costs.dtype != object and int(costs.max()) - int(costs.min()) > np.iinfo(costs.dtype).max:
        costs = costs.astype(object)
    weights = costs - costs.min(axis=1)[:, None]
    weights -= weights.min(axis=0)

    # A tree potential is a signed sum along a path of < n+m edges. The
    # reduced cost also subtracts two such potentials. Reserve extra room
    # for the bounded number of dual updates in the tight-route algorithm.
    bound = (2 * sum(weights.shape) + 128) * int(weights.max())
    if bound <= 2**31 - 1:
        return weights.astype(np.int32, copy=False)
    if bound <= 2**63 - 1:
        return weights.astype(np.int64, copy=False)
    return weights.astype(object, copy=False)


def solve(supply: list[int], demand: list[int], costs: np.ndarray) -> list[dict[int, int]]:
    """Return one optimal sparse row per supplier, using the original column IDs."""
    rows = [i for i, amount in enumerate(supply) if amount]
    cols = [j for j, amount in enumerate(demand) if amount]
    plan = [{} for _ in supply]
    if not rows:
        return plan
    if len(rows) == 1:
        plan[rows[0]] = {j: demand[j] for j in cols}
        return plan
    if len(cols) == 1:
        for i in rows:
            plan[i][cols[0]] = supply[i]
        return plan

    if len(rows) == len(supply) and len(cols) == len(demand):
        return solve_simplex(supply, demand, normalized_costs(costs))

    active_supply = [supply[i] for i in rows]
    active_demand = [demand[j] for j in cols]
    weights = normalized_costs(costs[np.ix_(rows, cols)])
    active_plan = solve_simplex(active_supply, active_demand, weights)
    for i, row in zip(rows, active_plan):
        plan[i] = {cols[j]: amount for j, amount in row.items()}
    return plan
