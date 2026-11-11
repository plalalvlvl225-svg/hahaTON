"""Тесты чтения in.txt и записи out.txt. Запуск: python -m unittest discover tests"""

import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from transport_io import read_problem, write_plan

EXAMPLE = b"3 4\n30 40 20\n20 30 30 10\n2 3 2 4\n3 2 5 1\n4 3 2 6\n"


class ReadProblem(unittest.TestCase):
    def read(self, content):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "in.txt")
            with open(path, "wb") as file:
                file.write(content)
            supply, demand, costs = read_problem(path)
        return supply, demand, costs.tolist()

    def check_example(self, content):
        supply, demand, costs = self.read(content)
        self.assertEqual(supply, [30, 40, 20])
        self.assertEqual(demand, [20, 30, 30, 10])
        self.assertEqual(costs, [[2, 3, 2, 4], [3, 2, 5, 1], [4, 3, 2, 6]])

    def test_unix_line_endings(self):
        self.check_example(EXAMPLE)

    def test_windows_line_endings_without_final_newline(self):
        self.check_example(EXAMPLE.replace(b"\n", b"\r\n").rstrip())

    def test_utf8_bom(self):
        self.check_example(b"\xef\xbb\xbf" + EXAMPLE)

    def test_extra_spaces_and_blank_lines(self):
        self.check_example(EXAMPLE.replace(b" ", b"   ").replace(b"\n", b" \n\n"))

    def test_example_file_from_task(self):
        supply, demand, costs = read_problem(os.path.join(ROOT, "examples", "in.txt"))
        self.assertEqual((supply, demand), ([30, 40, 20], [20, 30, 30, 10]))
        self.assertEqual(costs.shape, (3, 4))

    def test_too_few_numbers(self):
        with self.assertRaises(ValueError):
            self.read(b"2 2\n1 1\n1 1\n5 5\n5\n")

    def test_not_a_number(self):
        with self.assertRaises(ValueError):
            self.read(b"1 1\n1\n1\nabc\n")


class WritePlan(unittest.TestCase):
    def write(self, total_cost, shipments, n_cols):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "out.txt")
            write_plan(path, total_cost, shipments, n_cols)
            with open(path, newline="") as file:
                return file.read().replace("\r\n", "\n")

    def test_example(self):
        shipments = [[(0, 20), (2, 10)], [(1, 30), (3, 10)], [(2, 20)]]
        self.assertEqual(self.write(170, shipments, 4), "170\n20 0 10 0\n0 30 0 10\n0 0 20 0")

    def test_empty_row_and_edges(self):
        shipments = [[], [(0, 7)], [(2, 123456789)], [(0, 1), (1, 2), (2, 3)]]
        self.assertEqual(self.write(5, shipments, 3), "5\n0 0 0\n7 0 0\n0 0 123456789\n1 2 3")

    def test_single_column(self):
        self.assertEqual(self.write(0, [[], [(0, 4)]], 1), "0\n0\n4")


class WholeProgram(unittest.TestCase):
    def test_example_from_task(self):
        """python main.py на примере из задания даёт ответ из задания."""
        with tempfile.TemporaryDirectory() as folder:
            with open(os.path.join(ROOT, "examples", "in.txt"), "rb") as file:
                example = file.read()
            with open(os.path.join(folder, "in.txt"), "wb") as file:
                file.write(example)
            subprocess.run([sys.executable, os.path.join(ROOT, "main.py")], cwd=folder, check=True,
                           stdout=subprocess.DEVNULL)
            with open(os.path.join(folder, "out.txt")) as file:
                answer = file.read().split()
        with open(os.path.join(ROOT, "examples", "out.txt")) as file:
            expected = file.read().split()
        self.assertEqual(answer, expected)


if __name__ == "__main__":
    unittest.main()
