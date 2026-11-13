"""Тесты решателя. Запуск из корня репозитория: python -m unittest discover tests"""

import os
import random
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import reference
import solver
from solver import solve_transport, verify_plan


def random_problem(rnd, n_rows, n_cols, max_volume, max_cost, min_cost=0):
    """Случайная закрытая задача: суммарные запасы равны суммарным потребностям."""
    supply = [rnd.randint(0, max_volume) for _ in range(n_rows)]
    demand = [rnd.randint(0, max_volume) for _ in range(n_cols)]
    gap = sum(supply) - sum(demand)
    if gap > 0:
        demand[rnd.randrange(n_cols)] += gap
    else:
        supply[rnd.randrange(n_rows)] -= gap
    costs = [[rnd.randint(min_cost, max_cost) for _ in range(n_cols)] for _ in range(n_rows)]
    return supply, demand, costs


class ExampleFromTask(unittest.TestCase):
    def test_example(self):
        supply = [30, 40, 20]
        demand = [20, 30, 30, 10]
        costs = [[2, 3, 2, 4], [3, 2, 5, 1], [4, 3, 2, 6]]
        plan = solve_transport(supply, demand, costs)
        self.assertEqual(plan.total_cost, 170)
        verify_plan(supply, demand, costs, plan)


class AgainstReference(unittest.TestCase):
    """Сверка с независимым эталонным решателем на множестве случайных задач."""

    def check(self, supply, demand, costs):
        plan = solve_transport(supply, demand, costs)
        verify_plan(supply, demand, costs, plan)
        self.assertEqual(plan.total_cost, reference.min_cost(supply, demand, costs))

    def test_small_random(self):
        rnd = random.Random(2026)
        for _ in range(300):
            n_rows, n_cols = rnd.randint(1, 7), rnd.randint(1, 7)
            self.check(*random_problem(rnd, n_rows, n_cols, max_volume=20, max_cost=15))

    def test_many_equal_costs(self):
        rnd = random.Random(7)
        for _ in range(150):
            n_rows, n_cols = rnd.randint(1, 8), rnd.randint(1, 8)
            self.check(*random_problem(rnd, n_rows, n_cols, max_volume=5, max_cost=2))

    def test_negative_and_large_costs(self):
        rnd = random.Random(11)
        for _ in range(100):
            n_rows, n_cols = rnd.randint(1, 6), rnd.randint(1, 6)
            self.check(*random_problem(rnd, n_rows, n_cols, 50, 10**9, min_cost=-10**9))

    def test_rectangular(self):
        rnd = random.Random(3)
        for n_rows, n_cols in [(1, 12), (12, 1), (2, 15), (15, 2), (3, 20), (20, 3)]:
            for _ in range(10):
                self.check(*random_problem(rnd, n_rows, n_cols, max_volume=30, max_cost=40))


class LargerProblems(unittest.TestCase):
    """Задачи побольше: оптимальность доказывается потенциалами (verify_plan)."""

    def check(self, n_rows, n_cols, max_volume, max_cost, seed):
        supply, demand, costs = random_problem(random.Random(seed), n_rows, n_cols, max_volume, max_cost)
        plan = solve_transport(supply, demand, costs)
        verify_plan(supply, demand, costs, plan)
        return plan

    def test_square(self):
        self.check(120, 120, 999, 999, seed=1)

    def test_rectangular(self):
        self.check(40, 300, 999, 999, seed=2)
        self.check(300, 40, 999, 999, seed=3)

    def test_wide_cost_range(self):
        self.check(80, 80, 999, 10**9, seed=4)

    def test_narrow_cost_range(self):
        self.check(100, 100, 999, 3, seed=5)

    def test_candidate_list_is_extended_when_needed(self):
        # С одной клеткой-кандидатом на вершину списка заведомо не хватит,
        # и алгоритм обязан расширить его сам, сохранив точность ответа.
        saved = solver.CANDIDATES_PER_NODE
        solver.CANDIDATES_PER_NODE = 1
        try:
            for seed in range(5):
                self.check(60, 60, 999, 999, seed=seed)
        finally:
            solver.CANDIDATES_PER_NODE = saved


class EdgeCases(unittest.TestCase):
    def test_single_cell(self):
        plan = solve_transport([5], [5], [[7]])
        self.assertEqual(plan.total_cost, 35)
        self.assertEqual(plan.shipments, [[(0, 5)]])

    def test_nothing_to_ship(self):
        plan = solve_transport([0, 0], [0, 0, 0], [[1, 2, 3], [4, 5, 6]])
        self.assertEqual(plan.total_cost, 0)
        self.assertEqual(plan.to_matrix(3), [[0, 0, 0], [0, 0, 0]])

    def test_zero_supplier_and_consumer(self):
        supply, demand = [0, 9, 4], [6, 0, 7]
        costs = [[1, 1, 1], [5, 1, 2], [3, 1, 9]]
        plan = solve_transport(supply, demand, costs)
        verify_plan(supply, demand, costs, plan)
        self.assertEqual(plan.total_cost, reference.min_cost(supply, demand, costs))

    def test_all_costs_equal(self):
        supply, demand = [3, 4, 5], [6, 6]
        costs = [[4, 4]] * 3
        plan = solve_transport(supply, demand, costs)
        verify_plan(supply, demand, costs, plan)
        self.assertEqual(plan.total_cost, 4 * 12)

    def test_huge_volumes(self):
        # Объёмы и итоговая стоимость не помещаются в 64 бита – считаем точно.
        big = 10**30
        supply, demand = [big, 2 * big], [2 * big, big]
        costs = [[1, 5], [4, 2]]
        plan = solve_transport(supply, demand, costs)
        verify_plan(supply, demand, costs, plan)
        self.assertEqual(plan.total_cost, reference.min_cost(supply, demand, costs))

    def test_open_problem_keeps_surplus_at_suppliers(self):
        supply, demand = [10, 10, 10], [7, 8]
        costs = [[4, 6], [2, 9], [5, 1]]
        plan = solve_transport(supply, demand, costs)
        verify_plan(supply, demand, costs, plan)
        # Эталон: та же задача с фиктивным потребителем и нулевыми ценами.
        expected = reference.min_cost(supply, demand + [15], [row + [0] for row in costs])
        self.assertEqual(plan.total_cost, expected)

    def test_not_enough_supply(self):
        with self.assertRaises(ValueError):
            solve_transport([1], [2], [[1]])

    def test_wrong_matrix_size(self):
        with self.assertRaises(ValueError):
            solve_transport([1, 1], [2], [[1]])

    def test_negative_volume(self):
        with self.assertRaises(ValueError):
            solve_transport([3, -1], [2], [[1], [1]])

    def test_verify_rejects_non_optimal_plan(self):
        supply, demand = [1, 1], [1, 1]
        costs = np.array([[1, 5], [5, 1]])
        plan = solve_transport(supply, demand, costs)
        plan.shipments = [[(1, 1)], [(0, 1)]]      # дорогой план «крест-накрест»
        plan.total_cost = 10
        with self.assertRaises(ValueError):
            verify_plan(supply, demand, costs, plan)


if __name__ == "__main__":
    unittest.main()
