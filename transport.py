"""Exact minimum-cost transportation for balanced integer margins."""

from math import gcd

import numpy as np

from simplex import solve_simplex
from tight_flow import solve_tight


def normalized_costs(costs):
    """Remove row and column offsets. Use fixed-width math only with headroom."""
    # A tree potential is a signed sum on a path of < n+m edges. A reduced
    # cost uses two potentials. Reserve room for the 32 tight-flow updates too.
    headroom = 2 * sum(costs.shape) + 128
    span = int(costs.max()) - int(costs.min())
    if costs.dtype == object or span * headroom > 2**63 - 1:
        # Four-cell differences cancel *any* supplier and consumer surcharges.
        # If the remaining tariffs share a large unit, it will now be visible.
        # Use Python integers before these subtractions to avoid overflow.
        exact = costs.astype(object, copy=False)
        weights = exact - exact[:, :1]
        weights -= weights[0].copy()
        weights -= weights.min(axis=1)[:, None]
    else:
        # The public file reader supplies int64. Widen smaller integer arrays
        # too when their row differences cannot fit in their original dtype.
        if span > int(np.iinfo(costs.dtype).max):
            costs = costs.astype(np.int64)
        weights = costs - costs.min(axis=1)[:, None]
    weights -= weights.min(axis=0)

    highest = int(weights.max())
    if highest * headroom > 2**63 - 1:
        # Large prices can share a unit (e.g. monetary amounts scaled by 10**40).
        # A short sample can disprove a common factor. Verify it on all entries
        # before dividing; a positive divisor preserves every optimum.
        flat = weights.ravel()
        divisor = highest
        for value in flat[::max(1, flat.size // 64)]:
            divisor = gcd(divisor, int(value))
            if divisor == 1:
                break
        if divisor > 1:
            divisor = int(np.gcd.reduce(flat, initial=divisor))
            if divisor > 1:
                weights //= divisor
                highest //= divisor
    bound = headroom * highest
    if bound <= 2**31 - 1:
        return weights.astype(np.int32, copy=False)
    if bound <= 2**63 - 1:
        return weights.astype(np.int64, copy=False)
    return weights.astype(object, copy=False)


def _solve_active(supply, demand, costs):
    weights = normalized_costs(costs)
    plan = solve_tight(supply, demand, weights)
    if plan is None:
        plan = solve_simplex(supply, demand, weights)
    return plan


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
        return _solve_active(supply, demand, costs)

    active_supply = [supply[i] for i in rows]
    active_demand = [demand[j] for j in cols]
    active_plan = _solve_active(active_supply, active_demand, costs[np.ix_(rows, cols)])
    for i, row in zip(rows, active_plan):
        plan[i] = {cols[j]: amount for j, amount in row.items()}
    return plan
