"""Генератор случайных входных данных для проверки скорости.

Пример:
    python tools/generate.py 1000 1000
создаёт файл in.txt с задачей на 1000 поставщиков и 1000 потребителей.
Суммарные запасы всегда равны суммарным потребностям.

Параметры: N M [--max-volume 999] [--max-cost 999] [--seed 1] [--out in.txt]
"""

import argparse
import random


def balance(rnd, supply, demand):
    """Уравнивает суммы запасов и потребностей, добавляя недостающее по единице."""
    gap = sum(supply) - sum(demand)
    smaller = demand if gap > 0 else supply
    for _ in range(abs(gap)):
        smaller[rnd.randrange(len(smaller))] += 1


def main():
    parser = argparse.ArgumentParser(description="Генератор in.txt для транспортной задачи")
    parser.add_argument("n", type=int, help="число поставщиков")
    parser.add_argument("m", type=int, help="число потребителей")
    parser.add_argument("--max-volume", type=int, default=999, help="запасы и потребности от 1 до max-volume")
    parser.add_argument("--max-cost", type=int, default=999, help="цены перевозок от 1 до max-cost")
    parser.add_argument("--seed", type=int, default=1, help="зерно генератора: с тем же зерном получается тот же файл")
    parser.add_argument("--out", default="in.txt", help="имя выходного файла (по умолчанию in.txt)")
    args = parser.parse_args()

    rnd = random.Random(args.seed)
    supply = [rnd.randint(1, args.max_volume) for _ in range(args.n)]
    demand = [rnd.randint(1, args.max_volume) for _ in range(args.m)]
    balance(rnd, supply, demand)

    with open(args.out, "w", encoding="ascii") as file:
        file.write(f"{args.n} {args.m}\n")
        file.write(" ".join(map(str, supply)) + "\n")
        file.write(" ".join(map(str, demand)) + "\n")
        for _ in range(args.n):
            file.write(" ".join(str(rnd.randint(1, args.max_cost)) for _ in range(args.m)) + "\n")

    print(f"Создан {args.out}: поставщиков {args.n}, потребителей {args.m}, груза {sum(supply)}")


if __name__ == "__main__":
    main()
