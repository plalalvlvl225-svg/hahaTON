"""Хакатон «Оптимальный маршрут»: решение транспортной задачи.

Программа читает условие из файла in.txt и записывает оптимальный план
перевозок в файл out.txt (оба файла – в текущей папке).

Запуск:
    python main.py            – решить задачу;
    python main.py --verify   – после записи out.txt дополнительно доказать,
                                что найденный план оптимален.
"""

import os
import sys
import time

# Отсчёт времени начинается до загрузки NumPy: она тоже входит во время работы.
START = time.perf_counter()

from solver import solve_transport, verify_plan
from transport_io import read_problem, write_plan

IN_FILE = "in.txt"
OUT_FILE = "out.txt"


def data_folder():
    """Папка с in.txt: текущая, а если файла в ней нет – папка программы."""
    if os.path.exists(IN_FILE):
        return ""
    return os.path.dirname(os.path.abspath(__file__))


def report(message):
    """Печатает сообщение для человека.

    Результат к этому моменту уже записан в out.txt, поэтому сбой печати
    (например, из-за кодировки консоли) не должен завершать программу ошибкой.
    """
    try:
        print(message)
    except (UnicodeError, OSError):
        pass


def main():
    folder = data_folder()
    supply, demand, costs = read_problem(os.path.join(folder, IN_FILE))

    plan = solve_transport(supply, demand, costs)

    write_plan(os.path.join(folder, OUT_FILE), plan.total_cost, plan.shipments, len(demand))
    elapsed = time.perf_counter() - START

    report(f"Поставщиков: {len(supply)}, потребителей: {len(demand)}")
    report(f"Минимальные затраты: {plan.total_cost}")
    report(f"Результат записан в {OUT_FILE}, время работы: {elapsed:.3f} с")

    if "--verify" in sys.argv[1:]:
        verify_plan(supply, demand, costs, plan)
        report("Проверка пройдена: план допустим, его оптимальность доказана потенциалами")


if __name__ == "__main__":
    main()
