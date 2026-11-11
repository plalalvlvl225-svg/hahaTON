"""Primal-dual optimization on routes with zero reduced cost."""

import numpy as np


class _Shipments:
    def __init__(self, supply, demand):
        self.left = supply.copy()
        self.right = demand.copy()
        self.columns = [{} for _ in demand]
        # Only active (positive) demands are passed by the front end.
        self.unfilled = (1 << len(demand)) - 1

    def _augment(self, root, goal, from_row, from_col):
        """Follow predecessors twice: find the bottleneck, then change the flow."""
        amount = min(self.left[root], self.right[goal])
        j = goal
        while (i := from_row[j]) != root:
            j = from_col[i]
            amount = min(amount, self.columns[j][i])

        self.left[root] -= amount
        self.right[goal] -= amount
        if self.right[goal] == 0:
            self.unfilled ^= 1 << goal
        j = goal
        while True:
            i = from_row[j]
            column = self.columns[j]
            column[i] = column.get(i, 0) + amount
            if i == root:
                return
            j = from_col[i]
            column = self.columns[j]
            column[i] -= amount
            if column[i] == 0:
                del column[i]

    def maximize(self, neighbors):
        """Fill along alternating paths; return the closed cut if demand remains."""
        n, m = len(self.left), len(self.right)
        from_row = [-1] * m  # Preceding supplier of each visited consumer.
        from_col = [-1] * n  # Reverse step to a supplier with a positive shipment.
        seen = [0] * n
        search = 0
        closed_rows = np.zeros(n, dtype=bool)
        closed_cols = 0
        for root in range(n):
            if closed_rows[root]:
                continue
            while self.left[root]:
                search += 1
                seen[root] = search
                queue = [root]
                visited = closed_cols
                for i in queue:
                    options = neighbors[i] & ~visited
                    goals = options & self.unfilled
                    if goals:
                        goal = (goals & -goals).bit_length() - 1
                        from_row[goal] = i
                        self._augment(root, goal, from_row, from_col)
                        break
                    visited |= options
                    while options:
                        bit = options & -options
                        j = bit.bit_length() - 1
                        options -= bit
                        from_row[j] = i
                        for supplier in self.columns[j]:
                            if seen[supplier] != search:
                                seen[supplier] = search
                                from_col[supplier] = j
                                queue.append(supplier)
                else:
                    # No path leaves this set for an unfilled consumer. Later
                    # successful augmentations in this phase cannot enter it.
                    closed_rows[queue] = True
                    closed_cols = visited
                    break
        return closed_rows, closed_cols

    def rows(self):
        plan = [{} for _ in self.left]
        for j, column in enumerate(self.columns):
            for i, amount in column.items():
                plan[i][j] = amount
        return plan


def solve_tight(supply, demand, weights):
    """Return a certified plan, or hand a difficult price pattern to simplex."""
    if weights.dtype == object:
        return None
    n, m = weights.shape
    u = np.zeros(n, dtype=weights.dtype)
    v = np.zeros(m, dtype=weights.dtype)
    flow = _Shipments(supply, demand)
    # Small integral gaps can open many routes in one update. With many
    # distinct prices the simplex is faster than repeated full-table scans.
    for _ in range(32):
        slack = weights - u[:, None] - v
        packed = np.packbits(slack == 0, axis=1, bitorder="little")
        neighbors = [int.from_bytes(row, "little") for row in packed]
        rows, col_bits = flow.maximize(neighbors)
        if flow.unfilled == 0:
            return flow.rows()

        packed_cols = col_bits.to_bytes((m + 7) // 8, "little")
        cols = np.unpackbits(np.frombuffer(packed_cols, dtype=np.uint8),
                             count=m, bitorder="little").view(bool)
        gap = slack[np.ix_(rows, ~cols)].min()
        u[rows] += gap
        v[cols] -= gap
    return None
