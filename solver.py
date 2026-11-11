"""Решение транспортной задачи прямо-двойственным (венгерским) методом.

Идея метода
-----------
У каждого поставщика i есть потенциал u[i], у каждого потребителя j –
потенциал v[j]. Потенциалы всегда подобраны так, что «приведённая
стоимость» любой клетки неотрицательна:

    r[i][j] = c[i][j] - u[i] - v[j] >= 0.

Клетка с r[i][j] = 0 называется «нулевой». Груз возится только по нулевым
клеткам. Тогда, как только весь груз развезён, план оптимален: его
стоимость равна sum(a[i]*u[i]) + sum(b[j]*v[j]), а дешевле этой величины
не может быть ни один допустимый план (теорема двойственности).

Алгоритм чередует два шага, пока весь груз не развезён:

1. «Поток». По нулевым клеткам развозится максимум груза (алгоритм
   Диница: поиск в ширину расставляет расстояния, поиск в глубину
   проталкивает груз по кратчайшим цепочкам).
2. «Потенциалы». Когда по нулевым клеткам больше ничего не провезти,
   потенциалы сдвигаются на минимально возможную величину delta так,
   чтобы появились новые нулевые клетки, а старые перевозки остались
   на нулевых клетках.

Почему это быстро
-----------------
* Матрица 1000x1000 – это миллион клеток, но в оптимальный план попадают
  только самые дешёвые. Поэтому алгоритм работает с коротким списком
  клеток-кандидатов (r <= порога), а остальные клетки не просматривает.
  Список расширяется сам, если порога перестаёт хватать, поэтому ответ
  остаётся точным (см. `_PrimalDual._dual_step`).
* Все операции над миллионом клеток и над списком кандидатов выполняет
  NumPy; на чистом Python остаются только обходы маленькой сети нулевых
  клеток (около двух тысяч дуг).
"""

import sys

import numpy as np

# Сколько клеток-кандидатов в среднем берётся на одного поставщика
# (потребителя). В оптимальный план попадает 1-3 клетки на строку,
# запас нужен, чтобы список не приходилось расширять.
CANDIDATES_PER_NODE = 32

# Размер выборки, по которой подбирается начальный порог отбора кандидатов.
THRESHOLD_SAMPLE_SIZE = 20_000

# Метка «от вершины нельзя добраться до потребителя с дефицитом» в массивах расстояний.
_UNREACHED = -1


class TransportPlan:
    """Оптимальный план перевозок.

    total_cost      – суммарные затраты;
    shipments[i]    – список пар (j, x): поставщик i везёт x единиц
                      потребителю j; пары упорядочены по j, нулевые
                      перевозки не хранятся;
    row_potentials, col_potentials – потенциалы u и v, доказывающие
                      оптимальность плана (см. `verify_plan`).
    """

    def __init__(self, total_cost, shipments, row_potentials, col_potentials):
        self.total_cost = total_cost
        self.shipments = shipments
        self.row_potentials = row_potentials
        self.col_potentials = col_potentials

    def to_matrix(self, n_cols):
        """Возвращает план в виде обычной таблицы N x M (список списков)."""
        matrix = []
        for cells in self.shipments:
            row = [0] * n_cols
            for col, amount in cells:
                row[col] = amount
            matrix.append(row)
        return matrix


def solve_transport(supply, demand, costs):
    """Находит план перевозок минимальной стоимости.

    supply – запасы поставщиков (N целых чисел);
    demand – потребности потребителей (M целых чисел);
    costs  – матрица цен N x M (целые числа).

    Суммарные запасы должны быть не меньше суммарных потребностей.
    Если запасов больше (открытая задача), лишний груз остаётся
    у поставщиков: для него добавляется фиктивный потребитель
    с нулевой ценой перевозки.
    """
    supply = [int(x) for x in supply]
    demand = [int(x) for x in demand]
    costs = np.ascontiguousarray(costs, dtype=np.int64)
    n_rows, n_cols = len(supply), len(demand)

    if costs.shape != (n_rows, n_cols):
        raise ValueError(
            f"Матрица цен имеет размер {costs.shape}, а ожидался ({n_rows}, {n_cols})"
        )
    if n_rows == 0 or n_cols == 0:
        raise ValueError("Нужен хотя бы один поставщик и один потребитель")
    if min(supply) < 0 or min(demand) < 0:
        raise ValueError("Запасы и потребности не могут быть отрицательными")

    surplus = sum(supply) - sum(demand)
    if surplus < 0:
        raise ValueError("Суммарных запасов меньше, чем суммарных потребностей")
    if surplus > 0:
        # Открытая задача: фиктивный потребитель забирает излишек бесплатно.
        costs = np.hstack([costs, np.zeros((n_rows, 1), dtype=np.int64)])
        demand = demand + [surplus]

    state = _PrimalDual(supply, demand, costs)
    state.run()
    shipments = state.shipments()
    u, v = state.row_potentials(), state.col_potentials()
    if surplus > 0:
        shipments = [[cell for cell in cells if cell[0] < n_cols] for cells in shipments]
        # Потенциалы определены с точностью до общего сдвига. Сдвигаем их так,
        # чтобы у фиктивного потребителя потенциал был нулевым: тогда u <= 0,
        # а у поставщиков с остатком груза u = 0 (условия оптимальности
        # открытой задачи).
        shift = int(v[n_cols])
        u, v = u + shift, v[:n_cols] - shift

    return TransportPlan(_total_cost(costs, shipments), shipments, u, v)


def verify_plan(supply, demand, costs, plan):
    """Проверяет, что план допустим и оптимален. При ошибке бросает ValueError.

    Оптимальность проверяется по теореме двойственности: если нашлись
    потенциалы u, v, для которых c[i][j] - u[i] - v[j] >= 0 во всех клетках
    и = 0 во всех клетках с перевозками, то дешевле плана не существует.
    В открытой задаче (запасов больше, чем потребностей) дополнительно
    нужно u[i] <= 0, а у поставщиков с остатком груза u[i] = 0.
    Проверка не зависит от того, каким способом план был получен.
    """
    costs = np.asarray(costs, dtype=np.int64)
    n_rows, n_cols = costs.shape
    shipped = [0] * n_rows
    received = [0] * n_cols
    total = 0
    u, v = plan.row_potentials, plan.col_potentials

    for i, cells in enumerate(plan.shipments):
        for j, amount in cells:
            if amount <= 0:
                raise ValueError(f"Перевозка ({i}, {j}) не положительна: {amount}")
            if int(costs[i, j]) - int(u[i]) - int(v[j]) != 0:
                raise ValueError(f"Перевозка идёт по ненулевой клетке ({i}, {j})")
            shipped[i] += amount
            received[j] += amount
            total += amount * int(costs[i, j])

    if any(s > a for s, a in zip(shipped, supply)):
        raise ValueError("Вывоз от поставщика превышает его запас")
    if received != [int(b) for b in demand]:
        raise ValueError("Потребности удовлетворены не полностью")
    if total != plan.total_cost:
        raise ValueError(f"Стоимость плана {total} не совпадает с заявленной {plan.total_cost}")
    if int((costs - u[:, None] - v[None, :]).min()) < 0:
        raise ValueError("Потенциалы недопустимы: есть клетка с отрицательной оценкой")
    if sum(shipped) < sum(supply):
        if int(u.max()) > 0:
            raise ValueError("В открытой задаче потенциалы поставщиков должны быть <= 0")
        for i, (s, a) in enumerate(zip(shipped, supply)):
            if s < a and int(u[i]) != 0:
                raise ValueError(f"У поставщика {i} остался груз при ненулевом потенциале")


def _total_cost(costs, shipments):
    """Считает стоимость плана точно, в целых числах Python (без переполнения)."""
    rows, cols, amounts = [], [], []
    for i, cells in enumerate(shipments):
        for j, amount in cells:
            rows.append(i)
            cols.append(j)
            amounts.append(amount)
    if not amounts:
        return 0
    prices = costs[rows, cols].tolist()
    return sum(price * amount for price, amount in zip(prices, amounts))


def _group_by_key(sorted_keys, values, n_groups):
    """Раскладывает values по спискам: в k-й список попадают значения с ключом k.

    sorted_keys – неубывающий массив ключей (0..n_groups-1) той же длины.
    """
    bounds = np.searchsorted(sorted_keys, np.arange(n_groups + 1)).tolist()
    values = values.tolist()
    return [values[bounds[k]:bounds[k + 1]] for k in range(n_groups)]


class _PrimalDual:
    """Состояние прямо-двойственного метода для закрытой транспортной задачи.

    Обозначения: строка = поставщик, столбец = потребитель.
    """

    def __init__(self, supply, demand, costs):
        self.n_rows, self.n_cols = costs.shape

        # --- Начальные потенциалы: u[i] = минимум строки, v[j] = минимум
        # столбца после вычитания u. Все приведённые стоимости >= 0, и в
        # каждой строке и каждом столбце есть хотя бы одна нулевая клетка.
        self.u_start = costs.min(axis=1)
        reduced = costs - self.u_start[:, None]
        self.v_start = reduced.min(axis=0)
        reduced -= self.v_start
        self.reduced_flat = reduced.ravel()      # приведённые стоимости на старте

        # На сколько потенциалы изменились с начала работы:
        # u[i] = u_start[i] + row_rise[i], row_rise >= 0 (только растут);
        # v[j] = v_start[j] - col_drop[j], col_drop >= 0 (только убывают).
        self.row_rise = np.zeros(self.n_rows, dtype=np.int64)
        self.col_drop = np.zeros(self.n_cols, dtype=np.int64)

        # --- Клетки-кандидаты: все клетки с начальной приведённой
        # стоимостью <= threshold. Хранятся тремя параллельными массивами.
        self.threshold = self._initial_threshold()
        self._collect_candidates()

        # --- Сеть нулевых клеток (списки смежности).
        zero = np.flatnonzero(self.cand_reduced == 0)
        rows, cols = self.cand_row[zero], self.cand_col[zero]
        self.zero_cols = _group_by_key(rows, cols, self.n_rows)      # строка -> столбцы
        by_col = np.argsort(cols, kind="stable")
        self.zero_rows = _group_by_key(cols[by_col], rows[by_col], self.n_cols)  # столбец -> строки

        # --- Текущий план и остатки.
        self.supply_left = list(supply)          # ещё не вывезено от поставщика
        self.demand_left = list(demand)          # ещё не довезено потребителю
        self.unsent = sum(supply)
        self.inflow = [{} for _ in range(self.n_cols)]    # столбец -> {строка: объём}
        self.served = [set() for _ in range(self.n_rows)]  # строка -> столбцы с перевозкой > 0
        self.sources = [i for i, a in enumerate(supply) if a > 0]
        self.sinks = [j for j, b in enumerate(demand) if b > 0]

        # Расстояния до ближайшего потребителя с дефицитом (в дугах сети).
        self.dist_row = [_UNREACHED] * self.n_rows
        self.dist_col = [_UNREACHED] * self.n_cols

        # Глубина рекурсии поиска в глубину не больше длины цепочки
        # «строка – столбец – строка – ...» без повторов.
        depth = 2 * min(self.n_rows, self.n_cols) + 1000
        if sys.getrecursionlimit() < depth:
            sys.setrecursionlimit(depth)

    # ------------------------------------------------------------------
    # Главный цикл
    # ------------------------------------------------------------------

    def run(self):
        """Чередует шаги «поток» и «потенциалы», пока весь груз не развезён."""
        while True:
            self._max_flow()
            if self.unsent == 0:
                return
            self._dual_step()

    # ------------------------------------------------------------------
    # Шаг «поток»: максимум перевозок по нулевым клеткам (алгоритм Диница)
    # ------------------------------------------------------------------

    def _max_flow(self):
        while self.unsent:
            active = self._label_distances()
            if not active:
                return
            self._blocking_flow(active)

    def _label_distances(self):
        """Поиск в ширину от потребителей с дефицитом «назад» по сети.

        Груз может перейти:
          * из строки i в столбец j, если клетка (i, j) нулевая;
          * из столбца j в строку r, если r уже везёт груз в j (эту
            перевозку можно уменьшить, а груз строки r отправить дальше).
        Для каждой вершины считается число таких переходов до ближайшего
        потребителя с дефицитом. Возвращает поставщиков с остатком груза,
        от которых потребитель с дефицитом достижим.
        """
        zero_rows, served = self.zero_rows, self.served
        dist_row = [_UNREACHED] * self.n_rows
        dist_col = [_UNREACHED] * self.n_cols

        demand_left = self.demand_left
        cols = self.sinks = [j for j in self.sinks if demand_left[j] > 0]
        for j in cols:
            dist_col[j] = 0

        distance = 0
        while cols:
            distance += 1
            rows = []
            for j in cols:
                for i in zero_rows[j]:
                    if dist_row[i] < 0:
                        dist_row[i] = distance
                        rows.append(i)
            distance += 1
            cols = []
            for i in rows:
                for j in served[i]:
                    if dist_col[j] < 0:
                        dist_col[j] = distance
                        cols.append(j)

        self.dist_row, self.dist_col = dist_row, dist_col
        supply_left = self.supply_left
        self.sources = [i for i in self.sources if supply_left[i] > 0]
        return [i for i in self.sources if dist_row[i] > 0]

    def _blocking_flow(self, active):
        """Поиск в глубину: развозит груз по кратчайшим цепочкам.

        Двигаться разрешено только в вершину, которая на единицу ближе
        к потребителю с дефицитом, поэтому поиск не зацикливается, а
        тупиковые дуги каждая вершина пропускает навсегда (указатели
        next_zero и next_feeder).
        """
        zero_cols, inflow, served = self.zero_cols, self.inflow, self.served
        demand_left, dist_row, dist_col = self.demand_left, self.dist_row, self.dist_col
        next_zero = [0] * self.n_rows       # первая ещё не исчерпанная нулевая клетка строки
        feeders = [None] * self.n_cols      # строки, чьи перевозки в столбец можно уменьшать
        next_feeder = [0] * self.n_cols

        def ship_from_row(i, limit):
            """Отправляет из строки i до limit единиц груза; возвращает, сколько удалось."""
            cols = zero_cols[i]
            count = len(cols)
            target = dist_row[i] - 1
            shipped = 0
            k = next_zero[i]
            while k < count:
                j = cols[k]
                if dist_col[j] == target:
                    room = limit - shipped
                    if target == 0:
                        # Столбец j – потребитель с дефицитом: груз остаётся у него.
                        amount = demand_left[j]
                        if amount > room:
                            amount = room
                        demand_left[j] -= amount
                    else:
                        amount = release_from_col(j, room)
                    if amount:
                        suppliers = inflow[j]
                        if i in suppliers:
                            suppliers[i] += amount
                        else:
                            suppliers[i] = amount
                            served[i].add(j)
                        shipped += amount
                        if shipped == limit:
                            next_zero[i] = k
                            return shipped
                k += 1
            next_zero[i] = k
            return shipped

        def release_from_col(j, limit):
            """Освобождает в столбце j место под limit единиц нового груза.

            Для этого перевозки других строк в столбец j уменьшаются, а
            высвободившийся у них груз отправляется дальше по сети.
            """
            suppliers = inflow[j]
            rows = feeders[j]
            if rows is None:
                target = dist_col[j] - 1
                rows = feeders[j] = [r for r in suppliers if dist_row[r] == target]
            count = len(rows)
            released = 0
            k = next_feeder[j]
            while k < count:
                r = rows[k]
                capacity = suppliers[r]          # сколько строка r сейчас везёт в столбец j
                room = limit - released
                amount = ship_from_row(r, capacity if capacity < room else room)
                if amount:
                    released += amount
                    if amount == capacity:
                        del suppliers[r]
                        served[r].discard(j)
                    else:
                        suppliers[r] = capacity - amount
                        if released == limit:
                            # У строки r ещё есть что забрать – вернёмся к ней.
                            next_feeder[j] = k
                            return released
                # Перевозка строки r исчерпана либо строке некуда деть груз.
                k += 1
                if released == limit:
                    break
            next_feeder[j] = k
            return released

        supply_left = self.supply_left
        for i in active:
            shipped = ship_from_row(i, supply_left[i])
            supply_left[i] -= shipped
            self.unsent -= shipped

    # ------------------------------------------------------------------
    # Шаг «потенциалы»
    # ------------------------------------------------------------------

    def _dual_step(self):
        """Сдвигает потенциалы так, чтобы появились новые нулевые клетки.

        Пусть S – вершины, от которых по сети НЕ добраться до потребителя
        с дефицитом (после шага «поток» туда попадают все поставщики
        с остатком груза). Для строк из S потенциал увеличивается на delta,
        для столбцов из S – уменьшается на delta, где delta – минимальная
        приведённая стоимость клеток «строка из S – столбец не из S».
        Тогда:
          * клетки внутри S и клетки вне S не меняются;
          * клетки «строка из S – столбец не из S» дешевеют на delta,
            самые дешёвые из них становятся нулевыми (появляются новые пути);
          * клетки «строка не из S – столбец из S» дорожают на delta;
            перевозок по ним нет, поэтому план остаётся на нулевых клетках.
        """
        in_s_row = np.array(self.dist_row) < 0
        in_s_col = np.array(self.dist_col) < 0

        while True:
            # side = -1: клетка подешевеет; +1: подорожает; 0: не изменится.
            side = in_s_col[self.cand_col].view(np.int8) - in_s_row[self.cand_row].view(np.int8)
            crossing = side < 0
            if crossing.any():
                delta = int(self.cand_reduced[crossing].min())
                # Клетки вне списка кандидатов имеют начальную приведённую
                # стоимость > threshold. Пока рост потенциала любой строки
                # не превышает threshold, такие клетки остаются строго
                # положительными – значит, delta найдено верно и ни одна
                # нулевая клетка не пропущена.
                needed = int(self.row_rise[in_s_row].max()) + delta
                if needed <= self.threshold or self._all_cells_are_candidates():
                    break
            else:
                needed = 0
            self._extend_candidates(needed)

        cand_reduced = self.cand_reduced
        lost = np.flatnonzero((side > 0) & (cand_reduced == 0))
        cand_reduced += side * np.int64(delta)
        born = np.flatnonzero(crossing & (cand_reduced == 0))
        self.row_rise[in_s_row] += delta
        self.col_drop[in_s_col] += delta

        zero_cols, zero_rows = self.zero_cols, self.zero_rows
        for i, j in zip(self.cand_row[lost].tolist(), self.cand_col[lost].tolist()):
            zero_cols[i].remove(j)
            zero_rows[j].remove(i)
        for i, j in zip(self.cand_row[born].tolist(), self.cand_col[born].tolist()):
            zero_cols[i].append(j)
            zero_rows[j].append(i)

    # ------------------------------------------------------------------
    # Список клеток-кандидатов
    # ------------------------------------------------------------------

    def _initial_threshold(self):
        """Подбирает порог так, чтобы кандидатов было около CANDIDATES_PER_NODE на вершину."""
        size = self.reduced_flat.size
        share = min(1.0, CANDIDATES_PER_NODE * max(self.n_rows, self.n_cols) / size)
        sample = self.reduced_flat[:: max(1, size // THRESHOLD_SAMPLE_SIZE)]
        k = int(share * (sample.size - 1))
        return int(np.partition(sample, k)[k])

    def _collect_candidates(self):
        """Собирает клетки с начальной приведённой стоимостью <= threshold."""
        cells = np.flatnonzero(self.reduced_flat <= self.threshold)
        self.cand_row, self.cand_col = np.divmod(cells, self.n_cols)
        # Текущая приведённая стоимость: начальная минус рост u плюс падение v.
        self.cand_reduced = (
            self.reduced_flat[cells] - self.row_rise[self.cand_row] + self.col_drop[self.cand_col]
        )

    def _extend_candidates(self, needed):
        """Поднимает порог (минимум вдвое) и пересобирает список кандидатов.

        Новые клетки не могут быть нулевыми (их начальная стоимость больше
        старого порога, а потенциалы строк выросли не более чем на него),
        поэтому сеть нулевых клеток пересобирать не нужно.
        """
        if self._all_cells_are_candidates():
            # Расширять некуда. Для корректных данных сюда попасть нельзя:
            # пока груз не развезён, клетка «поставщик с остатком –
            # потребитель с дефицитом» всегда существует.
            raise RuntimeError("Внутренняя ошибка: список клеток-кандидатов исчерпан")
        self.threshold = max(2 * self.threshold, needed, 1)
        self._collect_candidates()

    def _all_cells_are_candidates(self):
        return self.cand_row.size == self.reduced_flat.size

    # ------------------------------------------------------------------
    # Результат
    # ------------------------------------------------------------------

    def shipments(self):
        """Возвращает план: для каждой строки – список (столбец, объём) по возрастанию столбца."""
        result = [[] for _ in range(self.n_rows)]
        for j, suppliers in enumerate(self.inflow):
            for i, amount in suppliers.items():
                result[i].append((j, amount))
        return result

    def row_potentials(self):
        return self.u_start + self.row_rise

    def col_potentials(self):
        return self.v_start - self.col_drop
