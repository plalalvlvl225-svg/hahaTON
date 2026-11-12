"""Regression tests for the optimizer and its contest file interface."""

import operator
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import transport
from tools.check_plan import check


MAIN = Path(transport.__file__).resolve().with_name("main.py")


def row_allocations(total, limits):
    """Enumerate all ways to place one supplier's stock in the remaining columns."""
    if len(limits) == 1:
        if total <= limits[0]:
            yield (total,)
        return
    minimum = max(0, total - sum(limits[1:]))
    maximum = min(total, limits[0])
    for first in range(minimum, maximum + 1):
        for suffix in row_allocations(total - first, limits[1:]):
            yield (first, *suffix)


def exhaustive_minimum(supply, demand, costs):
    """Enumerate feasible matrices, using only balances and integer arithmetic."""
    def totals(i, outstanding):
        if i == len(supply):
            if not any(outstanding):
                yield 0
            return
        for row in row_allocations(supply[i], outstanding):
            rest = [d - q for d, q in zip(outstanding, row)]
            price = sum(q * c for q, c in zip(row, costs[i]))
            for tail in totals(i + 1, rest):
                yield price + tail

    return min(totals(0, demand))


class OptimizerTests(unittest.TestCase):
    def assert_optimum(self, supply, demand, prices, expected, dtype=np.int64):
        costs = np.array(prices, dtype=dtype)
        plan = transport.solve(supply.copy(), demand.copy(), costs)
        self.assertEqual(len(plan), len(supply))
        delivered = [0] * len(demand)
        cost = 0
        for i, row in enumerate(plan):
            shipped = 0
            for key, value in row.items():
                j, quantity = operator.index(key), operator.index(value)
                self.assertIn(j, range(len(demand)))
                self.assertGreaterEqual(quantity, 0)
                shipped += quantity
                delivered[j] += quantity
                cost += prices[i][j] * quantity
            self.assertEqual(shipped, supply[i])
        self.assertEqual(delivered, demand)
        self.assertEqual(cost, expected)

    def test_small_problems_against_all_feasible_plans(self):
        rng = random.Random(726491)
        for case in range(150):
            n, m = rng.randint(1, 4), rng.randint(1, 4)
            supply, demand = [0] * n, [0] * m
            for _ in range(rng.randint(0, 7)):
                supply[rng.randrange(n)] += 1
                demand[rng.randrange(m)] += 1
            prices = [[rng.randint(-5, 5) for _ in range(m)] for _ in range(n)]
            with self.subTest(case=case, supply=supply, demand=demand, prices=prices):
                self.assert_optimum(supply, demand, prices,
                                    exhaustive_minimum(supply, demand, prices))

    def test_simplex_against_all_feasible_plans(self):
        rng = random.Random(1489)
        supply, demand = [2, 1, 2], [1, 2, 2]
        with patch.object(transport, "solve_tight", return_value=None):
            for _ in range(40):
                prices = [[rng.randrange(-2, 3) for _ in demand] for _ in supply]
                self.assert_optimum(supply, demand, prices,
                                    exhaustive_minimum(supply, demand, prices))

    def test_long_common_unit_after_row_and_column_shifts(self):
        unit = 10 ** 80
        prices = [[i - j + unit * w for j, w in enumerate(row)]
                  for i, row in enumerate([[0, 2, 3], [0, 5, 7], [4, 0, 0]])]
        self.assert_optimum([1, 1, 1], [1, 1, 1], prices, 2 * unit, dtype=object)
        self.assertEqual(transport.normalized_costs(np.array(prices, dtype=object)).dtype,
                         np.dtype("int32"))

    def test_greedy_shipment_must_be_reassigned(self):
        # Committing both units to the zero-price route leaves two at price 100.
        self.assert_optimum([2, 2], [2, 2], [[0, 3], [1, 100]], 8)

    def test_zero_margins_and_equal_prices(self):
        self.assert_optimum([0, 8, 0, 5], [2, 0, 11], [[-7] * 3 for _ in range(4)], -91)
        self.assert_optimum([0, 0], [0, 0, 0], [[2, -3, 6], [0, 7, -1]], 0)

    def test_rare_expensive_route_is_required(self):
        # The last supplier can send only 47 units to the other consumers.
        n = 48
        supply = [1] * (n - 1) + [500]
        prices = [[0] * n for _ in range(n)]
        prices[-1][-1] = 7
        self.assert_optimum(supply, supply, prices, 7 * (500 - (n - 1)))

    def test_int64_extremes_and_quantities_larger_than_int64(self):
        low, high = -(2 ** 63), 2 ** 63 - 1
        q = 10 ** 50 + 1
        self.assert_optimum([q, q], [q, q], [[high, low], [low, high]], 2 * q * low)

    def test_numpy_integers_inside_object_array(self):
        low = -(1 << 63)
        values = [np.int64(x) for x in (low, low, low, 0)]
        costs = np.array(values, dtype=object).reshape(2, 2)
        plan = transport.solve([1, 1], [1, 1], costs)
        self.assertEqual(plan, [{1: 1}, {0: 1}])

    def test_arbitrary_precision_costs_differ_by_one(self):
        base = 10 ** 100
        q = 10 ** 70 + 3
        self.assert_optimum([q, q], [q, q],
                            [[base + 1, base], [base, base]], 2 * q * base, dtype=object)


class FileInterfaceTests(unittest.TestCase):
    def run_main(self, input_data):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "in.txt").write_bytes(input_data)
            result = subprocess.run([sys.executable, str(MAIN)], cwd=root, capture_output=True,
                                    text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            return (root / "out.txt").read_text(encoding="ascii").splitlines()

    def test_official_example_with_bom_and_windows_line_endings(self):
        data = "3 4\r\n30 40 20\r\n20 30 30 10\r\n2 3 2 4\r\n3 2 5 1\r\n4 3 2 6\r\n"
        lines = self.run_main(data.encode("utf-8-sig"))
        self.assertEqual(lines, ["170", "20 0 10 0", "0 30 0 10", "0 0 20 0"])

    def test_switches_to_large_integers_mid_matrix(self):
        base = 10 ** 100 + 3
        data = f"3 3\n1 1 1\n1 1 1\n0 5 5\n{base+3} {base} {base+1}\n5 0 0\n"
        self.assertEqual(self.run_main(data.encode("ascii"))[0], str(base))

    def test_reads_and_writes_five_thousand_digit_integers(self):
        quantity = "9" * 5000
        # Multiplication by ten appends a zero, so the oracle requires no parser.
        data = f"1 1\n{quantity}\n{quantity}\n10\n"
        self.assertEqual(self.run_main(data.encode("ascii")), [quantity + "0", quantity])


class CertificateTests(unittest.TestCase):
    def test_detects_suboptimal_plan_even_when_totals_are_valid(self):
        with tempfile.TemporaryDirectory() as directory:
            source, result = Path(directory) / "in.txt", Path(directory) / "out.txt"
            source.write_text("2 2\n1 1\n1 1\n2 1\n1 2\n", encoding="ascii")
            result.write_text("4\n1 0\n0 1\n", encoding="ascii")
            with self.assertRaisesRegex(ValueError, "Negative cycle"):
                check(source, result)
            result.write_text("2\n0 1\n1 0\n", encoding="ascii")
            self.assertEqual(check(source, result)[-2:], (2, 2))


if __name__ == "__main__":
    unittest.main()
