"""Чтение условия транспортной задачи из in.txt и запись плана в out.txt."""

import numpy as np

_UTF8_BOM = b"\xef\xbb\xbf"


def read_problem(path):
    """Читает условие задачи. Возвращает (запасы, потребности, матрица цен).

    Формат файла (все числа целые, разделены пробелами):
        N M
        a1 a2 ... aN         – запасы поставщиков
        b1 b2 ... bM         – потребности потребителей
        c11 c12 ... c1M      – цены перевозок, N строк по M чисел
        ...
    Разбиение на строки не проверяется: числа читаются подряд, поэтому
    файл подходит с любыми окончаниями строк (Windows или Unix) и с
    меткой BOM в начале (её ставит, например, Блокнот).
    """
    with open(path, "rb") as file:
        raw = file.read()
    if raw.startswith(_UTF8_BOM):
        raw = raw[len(_UTF8_BOM):]

    numbers = _parse_integers(raw)
    if numbers.size < 2:
        raise ValueError("В первой строке должны быть два числа: N и M")
    n_rows, n_cols = int(numbers[0]), int(numbers[1])
    if n_rows < 1 or n_cols < 1:
        raise ValueError(f"Некорректные размеры задачи: N = {n_rows}, M = {n_cols}")

    expected = 2 + n_rows + n_cols + n_rows * n_cols
    if numbers.size < expected:
        raise ValueError(f"В файле {numbers.size} чисел, а для N = {n_rows}, M = {n_cols} нужно {expected}")

    supply_end = 2 + n_rows
    demand_end = supply_end + n_cols
    supply = numbers[2:supply_end].tolist()
    demand = numbers[supply_end:demand_end].tolist()
    costs = numbers[demand_end:expected].reshape(n_rows, n_cols)
    return supply, demand, costs


def _parse_integers(raw):
    """Превращает текст с целыми числами в массив int64."""
    try:
        # Самый быстрый способ: разбор текста целиком выполняет NumPy.
        return np.fromstring(raw, dtype=np.int64, sep=" ")
    except Exception:
        # Запасной путь на случай непредвиденного содержимого файла:
        # медленнее, но сообщает, какое именно значение не является числом.
        return np.array([int(token) for token in raw.split()], dtype=np.int64)


def write_plan(path, total_cost, shipments, n_cols):
    """Записывает план перевозок.

    Формат файла:
        суммарные затраты
        x11 x12 ... x1M      – объёмы перевозок, N строк по M чисел
        ...
    shipments[i] – список пар (j, x) ненулевых перевозок поставщика i,
    упорядоченный по j (см. solver.TransportPlan).

    В плане почти все числа – нули, поэтому строка собирается из готовых
    кусков «0 0 0 ...», между которыми вставляются ненулевые перевозки.
    """
    zeros = "0 " * n_cols
    lines = [str(total_cost)]
    for cells in shipments:
        parts = []
        position = 0                      # столько столбцов строки уже записано
        for col, amount in cells:
            parts.append(zeros[:2 * (col - position)])
            parts.append(f"{amount} ")
            position = col + 1
        parts.append(zeros[:2 * (n_cols - position)])
        lines.append("".join(parts)[:-1])  # без пробела в конце строки

    # Как и в примере из задания, после последней строки перевода строки нет.
    with open(path, "w", encoding="ascii") as file:
        file.write("\n".join(lines))
