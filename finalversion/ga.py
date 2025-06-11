#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GA for TSP
Author : Jimmy Chang

ga_tsp_full.py
==========================
Genetic Algorithm for solving the Symmetric Traveling Salesman Problem (TSP).

- Encoding : permutation of city indices (0-based)
- Crossover: Order Crossover (OX1)
- Mutation : swap mutation
- Selection: truncation (best 50 %) + random mating
- Elitism  : keep the best individual each generation

中文說明：
    本程式利用遺傳演算法求解對稱式 TSP，並額外量測
    執行時間、CPU 時間與記憶體增量。I/O 方面支援：
        (1) 以 tsplib95 讀取 .tsp 檔
        (2) 讀取官方 .opt.tour 檔並比較成本
    全程 **不依賴 NetworkX**，符合題目限制。

Author : Jimmy Chang
Date   : 2025-06-11
"""

import os
import random
import time
from typing import List, Tuple

# --- psutil 為非必需品：若沒安裝則降級為僅量測時間 -------------------------
try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    print("⚠️  找不到 psutil，將僅顯示執行時間。可 `pip install psutil` 取得完整量測。")
    _HAS_PSUTIL = False

import tsplib95


# =============================================================================
#  Genetic Algorithm
# =============================================================================
def genetic_algorithm(cost: List[List[int]],
                      n: int,
                      population_size: int = 100,
                      generations: int = 1000,
                      mutation_rate: float = 0.02,
                      seed: int | None = None
                      ) -> Tuple[List[int], int]:
    """遺傳演算法核心。

    Args
    ----
    cost : NxN 成本矩陣
    n    : 城市數量
    population_size : 族群大小
    generations     : 迭代世代
    mutation_rate   : 突變機率
    seed            : 隨機種子（可重現結果）

    Returns
    -------
    best_tour_overall : 最佳路徑 (list[int], 0-base)
    best_cost_overall : 對應總成本
    """
    if seed is not None:
        random.seed(seed)

    # ---------- helper functions ---------------------------------------------
    def generate_population() -> List[List[int]]:
        """隨機建立初始族群。"""
        return [random.sample(range(n), n) for _ in range(population_size)]

    def fitness(tour: List[int]) -> int:
        """計算路徑總成本（成本越低越佳）。"""
        return sum(cost[tour[i]][tour[(i + 1) % n]] for i in range(n))

    def crossover(p1: List[int], p2: List[int]) -> List[int]:
        """Order Crossover (OX1)。"""
        child = [None] * n
        start, end = sorted(random.sample(range(n), 2))
        # (a) 複製父 1 片段
        child[start:end + 1] = p1[start:end + 1]

        # (b) 從父 2 依序填入其餘城市
        ptr = 0
        for i in range(n):
            if child[i] is None:
                # 找到父 2 中不重複的城市
                while p2[ptr] in child:
                    ptr = (ptr + 1) % n     # 保證不會溢位
                child[i] = p2[ptr]
                ptr = (ptr + 1) % n
        return child

    def mutate(tour: List[int]) -> List[int]:
        """交換突變 (swap mutation)。"""
        if random.random() < mutation_rate:
            i, j = random.sample(range(n), 2)
            tour[i], tour[j] = tour[j], tour[i]
        return tour

    # ---------- main GA loop --------------------------------------------------
    population = generate_population()
    best_tour_overall = min(population, key=fitness)
    best_cost_overall = fitness(best_tour_overall)

    for gen in range(generations):
        # 1) 評分並排序
        population.sort(key=fitness)

        # 2) 更新最佳解
        if fitness(population[0]) < best_cost_overall:
            best_tour_overall   = population[0]
            best_cost_overall   = fitness(population[0])
            #print(f"[Gen {gen}] New Best = {best_cost_overall}")

        # 3) 產生新族群
        new_population = [population[0]]                # 精英保留
        pool = population[:population_size // 2]        # 前 50 % 作選擇池

        for _ in range(population_size - 1):
            parent1, parent2 = random.choices(pool, k=2)
            child = mutate(crossover(parent1, parent2))
            new_population.append(child)

        population = new_population

    return best_tour_overall, best_cost_overall


# =============================================================================
#  I/O 與效能量測
# =============================================================================
def load_optimal_tour(filename: str) -> List[int]:
    """讀取 .opt.tour，回傳 0-based 城市索引串列。"""
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    tour, in_section = [], False
    for line in lines:
        line = line.strip()
        if in_section:
            if line in ('-1', 'EOF') or not line.isdigit():
                break
            tour.append(int(line) - 1)      # 轉 0-base
        elif line == 'TOUR_SECTION':
            in_section = True
    return tour


def calculate_tour_cost(problem: tsplib95.models.Problem, tour: List[int]) -> int:
    """利用 tsplib95 物件計算一條路徑成本（含回到起點）。"""
    total = 0
    for i in range(len(tour)):
        u = tour[i] + 1                     # 轉回 1-based 節點編號
        v = tour[(i + 1) % len(tour)] + 1
        total += problem.get_weight(u, v)
    return total


def measure_performance(cost: List[List[int]], n: int, **ga_kwargs):
    """執行 GA 並量測效能。"""
    if _HAS_PSUTIL:
        proc = psutil.Process(os.getpid())
        start_cpu  = proc.cpu_times()
        start_mem  = proc.memory_info().rss
    start_time = time.perf_counter()

    tour, total_cost = genetic_algorithm(cost, n, **ga_kwargs)

    end_time = time.perf_counter()
    if _HAS_PSUTIL:
        end_cpu   = proc.cpu_times()
        end_mem   = proc.memory_info().rss
        cpu_time  = (end_cpu.user - start_cpu.user) + (end_cpu.system - start_cpu.system)
        mem_usage = (end_mem - start_mem) / (1024 * 1024)
    else:
        cpu_time, mem_usage = float('nan'), float('nan')

    return tour, total_cost, end_time - start_time, cpu_time, mem_usage


# =============================================================================
#  主程式入口
# =============================================================================
def main():
    # ---------------------------------------------------------------------
    # 可自行更換檔名 (同資料夾)
    TSP_FILE      = 'pa561.tsp'
    OPT_TOUR_FILE = 'pa561.opt.tour'
    # GA 參數
    GA_PARAMS = dict(
        population_size = 100,
        generations     = 1500,
        mutation_rate   = 0.02,
        seed            = 0          # 固定隨機種子便於重現
    )
    # ---------------------------------------------------------------------

    if not os.path.exists(TSP_FILE):
        print(f"❌ TSP file '{TSP_FILE}' not found.")
        return

    # 1) 讀取 .tsp
    problem = tsplib95.load(TSP_FILE)
    nodes   = sorted(problem.get_nodes())
    n       = len(nodes)

    # 2) 建立成本矩陣 (0-based)
    cost_matrix = [[problem.get_weight(nodes[i], nodes[j]) for j in range(n)]
                   for i in range(n)]

    # 3) 執行 GA
    print(f"🔄 Solving '{TSP_FILE}' (n={n}) with Genetic Algorithm ...")
    tour, total_cost, wall, cpu, mem = measure_performance(cost_matrix, n, **GA_PARAMS)

    # 4) 結果顯示
    print("\n===== GA Result =====")
    print(f"Best tour (first 10 cities) : {tour[:10]} ...")
    print(f"Total cost (GA)             : {total_cost}")
    print(f"Wall-clock time             : {wall:.4f} s")
    if _HAS_PSUTIL:
        print(f"CPU time                    : {cpu:.4f} s")
        print(f"Memory Δ                    : {mem:.2f} MB")

    # 5) 若有官方最佳路徑，做比較
    if os.path.exists(OPT_TOUR_FILE):
        opt_tour  = load_optimal_tour(OPT_TOUR_FILE)
        opt_cost  = calculate_tour_cost(problem, opt_tour)
        print("\n===== Compare with OPT =====")
        print(f"Official optimal cost       : {opt_cost}")
        if opt_cost > 0:
            gap = (total_cost - opt_cost) / opt_cost * 100
            print(f"Gap to optimal              : {gap:.2f} %")
        else:
            print("⚠️  OPT cost = 0, cannot compute gap.")
    else:
        print(f"\n⚠️  '{OPT_TOUR_FILE}' not found, skipping comparison.")


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
