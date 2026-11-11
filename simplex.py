"""Transportation simplex on a spanning tree of supplier and consumer vertices."""

import numpy as np


def _shortlist(cost):
    """Keep cheap routes for every vertex, including vertices on the shorter side."""
    n, m = cost.shape
    k = min(24, m)
    cols = np.argpartition(cost, k - 1, axis=1)[:, :k]
    from_rows = np.arange(n)[:, None] * m + cols
    k = min(24, n)
    rows = np.argpartition(cost, k - 1, axis=0)[:k]
    from_cols = rows * m + np.arange(m)
    return np.unique(np.concatenate((from_rows.ravel(), from_cols.ravel())))


def _initial_basis(supply, demand, cost, candidates):
    """Fill cheap routes, finish remaining balances, then connect the forest."""
    n, m = cost.shape
    routes = [{} for _ in range(n + m)]
    flow = {}

    def insert(i, j, amount):
        edge = i * m + j
        routes[i][n + j] = routes[n + j][i] = edge
        flow[edge] = amount

    a, b = supply.copy(), demand.copy()

    def ship(i, j):
        amount = min(a[i], b[j])
        insert(i, j, amount)
        a[i] -= amount
        b[j] -= amount

    ordered = candidates[np.argsort(cost.ravel()[candidates], kind="stable")].tolist()
    for edge in ordered:
        i, j = divmod(edge, m)
        if a[i] and b[j]:
            ship(i, j)

    i = j = 0
    while i < n and j < m:
        if not a[i]:
            i += 1
        elif not b[j]:
            j += 1
        else:
            ship(i, j)

    # Every shipment exhausted an endpoint, so positive routes form a forest.
    # A simultaneous exhaustion may have left several disconnected trees.
    component = [-1] * (n + m)
    for root in range(n + m):
        if component[root] != -1:
            continue
        vertices = [root]
        component[root] = root
        for vertex in vertices:
            for other in routes[vertex]:
                if component[other] == -1:
                    component[other] = root
                    vertices.append(other)

    def find(vertex):
        while component[vertex] != vertex:
            component[vertex] = component[component[vertex]]
            vertex = component[vertex]
        return vertex

    # The first row and first column together can connect any remaining pieces.
    bridges = [i * m for i in range(n)] + list(range(m))
    for edge in ordered + bridges:
        if len(flow) == n + m - 1:
            break
        i, j = divmod(edge, m)
        left, right = find(i), find(n + j)
        if left != right:
            component[right] = left
            insert(i, j, 0)
    return routes, flow


class _Basis:
    """A tree edge has zero reduced cost even if its current volume is zero."""

    def __init__(self, supply, demand, cost, candidates):
        self.n, self.m = cost.shape
        self.routes, self.flow = _initial_basis(supply, demand, cost, candidates)
        self.parent = [-1] * (self.n + self.m)
        self.depth = [0] * (self.n + self.m)
        self.price = np.zeros(self.n + self.m, dtype=cost.dtype)
        flat = cost.ravel()
        vertices = [0]
        for u in vertices:
            for v, edge in self.routes[u].items():
                if v != self.parent[u]:
                    self.parent[v] = u
                    self.depth[v] = self.depth[u] + 1
                    # price[supplier] - price[consumer] == cost on tree edges.
                    step = flat[edge] if u >= self.n else -flat[edge]
                    self.price[v] = self.price[u] + step
                    vertices.append(v)

    def pivot(self, edge, reduced_cost):
        """Insert a route and send the largest feasible amount around its cycle."""
        i, j = divmod(edge, self.m)
        j += self.n
        x, y = i, j
        increase, decrease = [], []
        # Climb both ends of the unique tree path to their common ancestor.
        # A decreasing edge also records which endpoint lies in its subtree.
        while x != y:
            if self.depth[x] >= self.depth[y]:
                p = self.parent[x]
                route = self.routes[x][p]
                if x < self.n:
                    decrease.append((self.flow[route], route, i, j))
                else:
                    increase.append(route)
                x = p
            else:
                p = self.parent[y]
                route = self.routes[y][p]
                if y >= self.n:
                    decrease.append((self.flow[route], route, j, i))
                else:
                    increase.append(route)
                y = p
        # Smallest edge ID resolves ties for the leaving variable in Bland's rule.
        amount, leaving, child, new_parent = min(decrease)
        if amount:
            for route in increase:
                self.flow[route] += amount
            for _, route, _, _ in decrease:
                self.flow[route] -= amount
        old_i, old_j = divmod(leaving, self.m)
        old_j += self.n
        del self.flow[leaving], self.routes[old_i][old_j], self.routes[old_j][old_i]
        self.routes[i][j] = self.routes[j][i] = edge
        self.flow[edge] = amount
        self._reattach(child, new_parent, reduced_cost if child == i else -reduced_cost)
        return amount

    def _reattach(self, child, new_parent, shift):
        """Only the component detached by the leaving edge changes its prices."""
        self.parent[child] = new_parent
        self.depth[child] = self.depth[new_parent] + 1
        moved = [child]
        for u in moved:
            for v in self.routes[u]:
                if v != self.parent[u]:
                    self.parent[v] = u
                    self.depth[v] = self.depth[u] + 1
                    moved.append(v)
        self.price[moved] += shift


def _optimize_candidates(basis, cost, pool):
    # Scan consecutive blocks. A full round without a negative reduced cost
    # proves optimality on this pool. Its edge IDs are sorted for Bland's rule.
    rows, columns = np.divmod(pool, basis.m)
    columns += basis.n
    weights = cost.ravel()[pool]
    block_size = 2048
    blocks = (len(pool) + block_size - 1) // block_size
    start = stalled = 0
    while True:
        if stalled >= 8:
            # From here until the next strict objective decrease, use the first
            # eligible edge globally. Together with the leaving tie-break this
            # rules out a cycle of zero-volume pivots.
            reduced = weights - basis.price[rows] + basis.price[columns]
            eligible = np.flatnonzero(reduced < 0)
            if not eligible.size:
                return
            pos = int(eligible[0])
            delta = int(reduced[pos])
        else:
            for _ in range(blocks):
                part = slice(start, start + block_size)
                reduced = weights[part] - basis.price[rows[part]] + basis.price[columns[part]]
                offset = int(reduced.argmin())
                pos = start + offset
                start += block_size
                if start >= len(pool):
                    start = 0
                delta = int(reduced[offset])
                if delta < 0:
                    break
            else:
                return
        amount = basis.pivot(int(pool[pos]), delta)
        stalled = 0 if amount else stalled + 1


def solve_simplex(supply: list[int], demand: list[int], weights: np.ndarray) -> list[dict[int, int]]:
    """Return an optimal plan for positive active margins and normalized costs.

    Integer weights must either have sufficient int64 headroom for tree-path
    potential calculations or use dtype=object with Python integers.
    """
    pool = _shortlist(weights)
    basis = _Basis(supply, demand, weights, pool)
    n = len(supply)
    while True:
        _optimize_candidates(basis, weights, pool)
        # Verify the dual inequalities on the *entire* transportation table.
        reduced = weights - basis.price[:n, None] + basis.price[n:]
        missing = np.flatnonzero(reduced < 0)
        if not missing.size:
            break
        pool = np.union1d(pool, missing)
    plan = [{} for _ in supply]
    for edge, amount in basis.flow.items():
        if amount:
            i, j = divmod(edge, basis.m)
            plan[i][j] = amount
    return plan
