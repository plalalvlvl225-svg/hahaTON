"""File interface for the hackathon's balanced transportation problem."""

from pathlib import Path
import sys

import numpy as np

from transport import solve


def read_row(stream, width):
    row = list(map(int, stream.readline().split()))
    if len(row) != width:
        raise ValueError(f"Ожидалось чисел в строке: {width}, получено: {len(row)}")
    return row


def read_problem(path):
    with Path(path).open("rb") as stream:
        header = stream.readline().removeprefix(b"\xef\xbb\xbf").split()
        n, m = map(int, header)
        if n < 1 or m < 1:
            raise ValueError("Размеры матрицы должны быть положительными")
        supply = read_row(stream, n)
        demand = read_row(stream, m)
        if min(supply) < 0 or min(demand) < 0 or sum(supply) != sum(demand):
            raise ValueError("Нужны неотрицательные и сбалансированные запасы и потребности")
        costs = [read_row(stream, m) for _ in range(n)]
        if stream.read().strip():
            raise ValueError("Лишние данные после матрицы тарифов")
    try:
        matrix = np.array(costs, dtype=np.int64)
    except OverflowError:
        matrix = np.array(costs, dtype=object)
    return supply, demand, matrix


def write_plan(path, costs, shipments):
    total = sum(int(costs[i, j]) * amount
                for i, row in enumerate(shipments) for j, amount in row.items())
    width = costs.shape[1]
    with Path(path).open("w", encoding="ascii", newline="\n") as stream:
        stream.write(f"{total}\n")
        for row in shipments:
            fields = ["0"] * width
            for j, amount in row.items():
                fields[j] = str(amount)
            stream.write(" ".join(fields) + "\n")


def main():
    # The problem does not bound the decimal length of an integer.
    sys.set_int_max_str_digits(0)
    supply, demand, costs = read_problem("in.txt")
    shipments = solve(supply, demand, costs)
    write_plan("out.txt", costs, shipments)


if __name__ == "__main__":
    main()
