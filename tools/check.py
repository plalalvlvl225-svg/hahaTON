"""Независимая проверка ответа: читает in.txt и out.txt и сообщает, верен ли план.

Запуск из корня репозитория:
    python tools/check.py                    – проверить in.txt и out.txt
    python tools/check.py задача.txt ответ.txt

Проверяется всё, что требует задание:
  * формат out.txt: N+1 строк, в первой одно число, в остальных по M чисел;
  * допустимость: перевозки неотрицательны, вывоз не превышает запасов,
    потребности закрыты полностью;
  * первая строка равна стоимости записанного плана;
  * оптимальность: дешевле плана не существует.

Оптимальность проверяется без участия решателя. Программа сама пытается
подобрать потенциалы u, v, при которых c[i][j] - u[i] - v[j] >= 0 во всех
клетках и = 0 в клетках с перевозками. Если такие потенциалы есть, план
оптимален (теорема двойственности); если их нет – план можно удешевить.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transport_io import read_problem


def read_answer(path, n_rows, n_cols):
    """Читает out.txt, строго проверяя формат. Возвращает (стоимость, план N x M)."""
    with open(path, encoding="utf-8-sig") as file:
        lines = file.read().splitlines()
    while lines and not lines[-1].strip():
        lines.pop()                              # пустые строки в конце не считаем ошибкой
    if len(lines) != n_rows + 1:
        raise ValueError(f"в файле {len(lines)} строк, а должно быть {n_rows + 1}")

    first = lines[0].split()
    if len(first) != 1:
        raise ValueError("в первой строке должно быть одно число – сумма затрат")
    total_cost = int(first[0])

    plan = []
    for number, line in enumerate(lines[1:], start=2):
        row = [int(token) for token in line.split()]
        if len(row) != n_cols:
            raise ValueError(f"в строке {number} чисел: {len(row)}, а должно быть {n_cols}")
        plan.append(row)
    return total_cost, plan


def check_feasibility(supply, demand, costs, total_cost, plan):
    """Проверяет ограничения задачи и стоимость. Возвращает список найденных ошибок."""
    errors = []
    if any(x < 0 for row in plan for x in row):
        errors.append("в плане есть отрицательные перевозки")
    for i, row in enumerate(plan):
        if sum(row) > supply[i]:
            errors.append(f"поставщик {i + 1}: вывезено {sum(row)}, а запас {supply[i]}")
    for j, received in enumerate(map(sum, zip(*plan))):
        if received != demand[j]:
            errors.append(f"потребитель {j + 1}: получено {received}, а нужно {demand[j]}")

    prices = costs.tolist()
    real_cost = sum(x * c for row, price_row in zip(plan, prices) for x, c in zip(row, price_row) if x)
    if real_cost != total_cost:
        errors.append(f"в первой строке {total_cost}, а стоимость плана равна {real_cost}")
    return errors


def is_optimal(supply, costs, plan):
    """Проверяет оптимальность допустимого плана подбором потенциалов.

    Условия на потенциалы – это система неравенств вида «разность двух
    неизвестных не больше константы»:
        v[j] - u'[i] <= c[i][j]        для всех клеток (u' = -u),
        u'[i] - v[j] <= -c[i][j]       для клеток с перевозками.
    Такая система решается алгоритмом Беллмана – Форда: неизвестные – это
    кратчайшие расстояния в графе, где каждому неравенству отвечает дуга.
    Если расстояния перестали уменьшаться – потенциалы найдены. Если они
    уменьшаются дольше, чем вершин в графе, значит, в графе есть цикл
    отрицательной длины: по нему груз можно перераспределить и удешевить
    план, то есть план не оптимален.
    """
    n_rows, n_cols = costs.shape
    plan = np.array(plan, dtype=object)

    # Открытую задачу сводим к закрытой: остатки «везём» фиктивному
    # потребителю по нулевой цене.
    leftovers = [a - int(sum(row)) for a, row in zip(supply, plan)]
    if any(leftovers):
        costs = np.hstack([costs, np.zeros((n_rows, 1), dtype=np.int64)])
        plan = np.hstack([plan, np.array(leftovers, dtype=object).reshape(n_rows, 1)])
        n_cols += 1

    used_rows, used_cols = np.nonzero(plan)
    used_costs = costs[used_rows, used_cols]

    row_dist = np.zeros(n_rows, dtype=np.int64)       # u' = -u
    col_dist = np.zeros(n_cols, dtype=np.int64)       # v
    for _ in range(n_rows + n_cols + 1):
        new_col = np.minimum(col_dist, (costs + row_dist[:, None]).min(axis=0))
        new_row = row_dist.copy()
        np.minimum.at(new_row, used_rows, new_col[used_cols] - used_costs)
        if np.array_equal(new_row, row_dist) and np.array_equal(new_col, col_dist):
            return True
        row_dist, col_dist = new_row, new_col
    return False


def main():
    in_path = sys.argv[1] if len(sys.argv) > 1 else "in.txt"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "out.txt"

    supply, demand, costs = read_problem(in_path)
    try:
        total_cost, plan = read_answer(out_path, len(supply), len(demand))
    except ValueError as error:
        print(f"ОШИБКА формата {out_path}: {error}")
        return 1

    errors = check_feasibility(supply, demand, costs, total_cost, plan)
    if errors:
        for error in errors[:20]:
            print(f"ОШИБКА: {error}")
        return 1
    print(f"Формат верен, план допустим, сумма затрат {total_cost} совпадает с планом")

    if is_optimal(supply, costs, plan):
        print("План оптимален: дешевле перевезти нельзя")
        return 0
    print("ОШИБКА: план не оптимален, затраты можно уменьшить")
    return 1


if __name__ == "__main__":
    sys.exit(main())
