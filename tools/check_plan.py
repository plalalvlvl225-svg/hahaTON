"""Verify feasibility and optimality using only the two specified text files."""
import argparse
import sys
from pathlib import Path


def read_row(file, width, description, *, first=False):
    line = file.readline()
    if first:
        line = line.removeprefix(b"\xef\xbb\xbf")
    row = list(map(int, line.split()))
    if len(row) != width:
        raise ValueError(f"{description}: expected {width} integers, found {len(row)}")
    return row


def check(input_path, output_path):
    with Path(input_path).open("rb") as source, Path(output_path).open("rb") as result:
        n, m = read_row(source, 2, "dimensions", first=True)
        supply = read_row(source, n, "supply")
        demand = read_row(source, m, "demand")
        expected, = read_row(result, 1, "objective")
        positive = [[] for _ in range(n + m)]
        received = [0] * m
        cost = []
        total, used = 0, 0
        for i in range(n):
            prices = read_row(source, m, f"cost row {i}")
            flow = read_row(result, m, f"plan row {i}")
            if sum(flow) != supply[i] or any(q < 0 for q in flow):
                raise ValueError(f"Invalid row {i}: flow or row sum")
            for j, (price, quantity) in enumerate(zip(prices, flow)):
                if quantity:
                    positive[i].append((n + j, price))
                    positive[n + j].append((i, -price))
                    received[j] += quantity
                    total += quantity * price
                    used += 1
            cost.append(prices)
        if source.read().strip() or result.read().strip():
            raise ValueError("Additional data after matrix")
        if received != demand or total != expected:
            raise ValueError("Demand or objective differs from reported plan")

    # Each positive route forces equality p[consumer] - p[supplier] == cost.
    component = [-1] * (n + m)
    base = [0] * (n + m)
    groups = 0
    for root in range(n + m):
        if component[root] >= 0:
            continue
        component[root] = groups
        stack = [root]
        while stack:
            u = stack.pop()
            for v, difference in positive[u]:
                value = base[u] + difference
                if component[v] < 0:
                    base[v] = value
                    component[v] = groups
                    stack.append(v)
                elif base[v] != value:
                    raise ValueError("Contradictory equalities on positive shipments")
        groups += 1

    # Each component may shift by one unknown integer. The tightest upper
    # bound for a pair of component shifts enforces every route between them.
    edges = [{} for _ in range(groups)]
    for i, row in enumerate(cost):
        a = component[i]
        adjacent = edges[a]
        for j, price in enumerate(row, n):
            b = component[j]
            bound = price + base[i] - base[j]
            if a == b:
                if bound < 0:
                    raise ValueError(f"Violated dual constraint on route {i}, {j-n}")
            elif b not in adjacent or bound < adjacent[b]:
                adjacent[b] = bound

    # Bellman-Ford on component offsets detects inconsistent inequalities.
    shift = [0] * groups
    for _ in range(groups):
        changed = False
        for a, adjacent in enumerate(edges):
            for b, bound in adjacent.items():
                limit = shift[a] + bound
                if shift[b] > limit:
                    shift[b] = limit
                    changed = True
        if not changed:
            break
    else:
        raise ValueError("Negative cycle: plan is suboptimal")

    p = [value + shift[group] for value, group in zip(base, component)]
    for i, row in enumerate(cost):
        for j, price in enumerate(row, n):
            if p[j] - p[i] > price:
                raise ValueError("Constructed potential fails a route inequality")
    dual = sum(p[n + j] * q for j, q in enumerate(demand))
    dual -= sum(p[i] * q for i, q in enumerate(supply))
    if total != dual:
        raise ValueError("Primal cost and dual lower bound differ")
    return n, m, used, groups, total, dual


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Проверить допустимость и оптимальность плана")
    parser.add_argument("input", nargs="?", default="in.txt")
    parser.add_argument("output", nargs="?", default="out.txt")
    args = parser.parse_args()
    sys.set_int_max_str_digits(0)
    rows, cols, used, components, total, lower = check(args.input, args.output)
    print(f"{rows}x{cols}, {used} positive routes, {components} components")
    print(f"plan cost={total}, independently reconstructed dual lower bound={lower}")
