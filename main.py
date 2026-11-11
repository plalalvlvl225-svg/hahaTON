"""Хакатон «Оптимальный маршрут»: решение транспортной задачи.

Программа читает условие из файла in.txt и записывает оптимальный план
перевозок в файл out.txt (оба файла – в текущей папке).

Запуск:
    python main.py
"""

import os
import time

# Отсчёт времени начинается до загрузки NumPy: она тоже входит во время работы.
START = time.perf_counter()

from solver import solve_transport
from transport_io import read_problem, write_plan

IN_FILE = "in.txt"
OUT_FILE = "out.txt"


def data_folder():
    """Папка с in.txt: текущая, а если файла в ней нет – папка программы."""
    if os.path.exists(IN_FILE):
        return ""
    return os.path.dirname(os.path.abspath(__file__))


def main():
    folder = data_folder()
    supply, demand, costs = read_problem(os.path.join(folder, IN_FILE))

    plan = solve_transport(supply, demand, costs)

    write_plan(os.path.join(folder, OUT_FILE), plan.total_cost, plan.shipments, len(demand))
    elapsed = time.perf_counter() - START

    print(f"Поставщиков: {len(supply)}, потребителей: {len(demand)}")
    print(f"Минимальные затраты: {plan.total_cost}")
    print(f"Результат записан в {OUT_FILE}, время работы: {elapsed:.3f} с")


if __name__ == "__main__":
    main()
