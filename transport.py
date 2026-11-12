"""Exact minimum-cost transportation for balanced integer margins."""

from math import gcd

import numpy as np

from simplex import solve_simplex
from tight_flow import solve_tight


def _compress_wide_costs(costs):
    """Recover a common price unit even in the presence of additive surcharges."""
    exact = costs.astype(object, copy=False)
    n, m = exact.shape
    # D divides all four-cell differences iff row and column offsets can make
    # every price a multiple of D. A sample can cheaply disprove a common unit.
    divisor = 0
    for index in range(0, n * m, max(1, n * m // 64)):
        i, j = divmod(index, m)
        diff = exact[i, j] - exact[i, 0] - exact[0, j] + exact[0, 0]
        divisor = gcd(divisor, int(diff))
        if divisor == 1:
            return exact
    differences = exact - exact[:, :1]
    differences -= differences[0].copy()
    divisor = abs(int(np.gcd.reduce(differences.ravel(), initial=divisor)))
    if divisor == 0:  # Pure row and column surcharges; all plans cost the same.
        return np.zeros((n, m), dtype=np.int32)
    if divisor == 1:
        return exact
    differences //= divisor
    # Restore the column part in whole units: an arbitrary reference row
    # would otherwise distort the initial dual prices on large random tables.
    differences += exact[0] // divisor
    return differences


def normalized_costs(costs):
    """Remove row and column offsets. Use fixed-width math only with headroom."""
    # A tree potential is a signed sum on a path of < n+m edges. A reduced
    # cost uses two potentials. Reserve room for the 32 tight-flow updates too.
    if costs.dtype == object:
        # An object array provided by a caller can still contain fixed-width
        # NumPy scalar integers. Make the wide arithmetic exact in that case.
        costs = np.frompyfunc(int, 1, 1)(costs)
    headroom = 2 * sum(costs.shape) + 128
    span = int(costs.max()) - int(costs.min())
    if span * headroom > 2**63 - 1:
        costs = _compress_wide_costs(costs)
    elif costs.dtype != object and span > int(np.iinfo(costs.dtype).max):
        costs = costs.astype(np.int64)
    weights = costs - costs.min(axis=1)[:, None]
    weights -= weights.min(axis=0)
    bound = headroom * int(weights.max())
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
