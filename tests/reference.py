"""Эталонный решатель для тестов: простой, медленный, но очевидно верный.

Транспортная задача решается как задача о потоке минимальной стоимости:
груз отправляется порциями по самому дешёвому пути в остаточной сети,
путь ищется алгоритмом Беллмана – Форда. Алгоритм никак не связан с
методом из solver.py, поэтому совпадение ответов – хорошая проверка.
Годится только для маленьких задач.
"""

INFINITY = float("inf")


def min_cost(supply, demand, costs):
    """Возвращает минимальную стоимость перевозок (весь спрос должен быть закрыт)."""
    n_rows, n_cols = len(supply), len(demand)
    supply_left = list(supply)
    demand_left = list(demand)
    flow = [[0] * n_cols for _ in range(n_rows)]
    total = 0

    while sum(demand_left) > 0:
        # Кратчайшие расстояния от поставщиков с остатком груза.
        dist_row = [0 if supply_left[i] > 0 else INFINITY for i in range(n_rows)]
        dist_col = [INFINITY] * n_cols
        came_to_col = [None] * n_cols      # из какой строки пришли в столбец
        came_to_row = [None] * n_rows      # из какого столбца пришли в строку
        changed = True
        while changed:
            changed = False
            for i in range(n_rows):
                for j in range(n_cols):
                    # Вперёд по клетке (i, j): везём больше, платим c[i][j].
                    if dist_row[i] + costs[i][j] < dist_col[j]:
                        dist_col[j] = dist_row[i] + costs[i][j]
                        came_to_col[j] = i
                        changed = True
                    # Назад по клетке (i, j): отменяем часть перевозки.
                    if flow[i][j] > 0 and dist_col[j] - costs[i][j] < dist_row[i]:
                        dist_row[i] = dist_col[j] - costs[i][j]
                        came_to_row[i] = j
                        changed = True

        target = min(
            (j for j in range(n_cols) if demand_left[j] > 0), key=lambda j: dist_col[j]
        )
        if dist_col[target] == INFINITY:
            raise ValueError("Запасов не хватает, чтобы закрыть потребности")

        # Восстанавливаем путь и находим, сколько груза по нему пройдёт.
        path = []                          # клетки пути: (i, j, +1 или -1)
        j = target
        while True:
            i = came_to_col[j]
            path.append((i, j, +1))
            if came_to_row[i] is None:
                break
            j = came_to_row[i]
            path.append((i, j, -1))
        amount = min(supply_left[i], demand_left[target])
        for r, c, sign in path:
            if sign < 0:
                amount = min(amount, flow[r][c])

        for r, c, sign in path:
            flow[r][c] += sign * amount
            total += sign * amount * costs[r][c]
        supply_left[i] -= amount
        demand_left[target] -= amount

    return total
