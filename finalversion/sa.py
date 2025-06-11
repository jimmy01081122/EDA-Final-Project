"""
Simulated Annealing for the Symmetric TSP
Author : <your name>
Date   : 2025-06-11
"""

import random
import math
import time
import psutil
import tsplib95
import os
import numpy as np
import sys
# ------------------------------------------------------------
# Simulated Annealing 核心
# ------------------------------------------------------------
def simulated_annealing(cost_mat, n,
                        iterations: int = 100_000,
                        start_temp: float = 1_000.0,
                        alpha: float = 0.9995):
    """
    使用模擬退火演算法求解對稱 TSP。

    Args:
        cost_mat : 2D 成本矩陣 (list 或 numpy array)。
        n        : 城市數量。
        iterations : SA 總迭代次數。
        start_temp : 初始溫度 T₀。
        alpha      : 冷卻因子 (0 < α < 1)，控制降溫速度。

    Returns:
        best_tour  : 找到的最佳路徑 (list[int])。
        best_cost  : 最佳路徑的總長度 (float)。
    """

    # --- 1. 隨機產生初始解 ---
    # 產生一個 0 到 n-1 的隨機排列作為初始路徑。
    tour = list(range(n))
    random.shuffle(tour)

    def cost_of_tour(seq) -> float:
        """O(n) 迴圈計算巡迴長度。
        注意：這裡的成本計算是 O(n)，但在高效能實作中，2-opt 的成本差可以 O(1) 計算。
        為求清晰，此處保留 O(n) 的完整計算。
        """
        # (i + 1) % n 確保能從最後一個城市回到第一個城市，形成環路。
        return sum(
            cost_mat[seq[i]][seq[(i + 1) % n]]
            for i in range(n)
        )

    # 計算初始解的成本，並將其設為當前解與歷史最佳解。
    # tour[:] 創建一個 tour 的淺拷貝，避免後續修改影響 best_tour。
    current_cost = cost_of_tour(tour)
    best_tour, best_cost = tour[:], current_cost

    # 設定初始溫度。
    T = start_temp

    # --- 演算法主迴圈 ---
    for _ in range(iterations):
        # --- 2. 產生鄰域解 (使用 2-opt) ---
        # 隨機選取兩個不同的索引 i 和 j。
        i, j = sorted(random.sample(range(n), 2))
        # 將 tour[i] 到 tour[j] 之間的片段反轉，產生新路徑。
        # 這是 TSP 的經典鄰域操作：2-opt 交換。
        new_tour = tour[:i] + tour[i:j + 1][::-1] + tour[j + 1:]

        # --- 3. 計算成本差 (Δ) ---
        new_cost = cost_of_tour(new_tour)
        delta = new_cost - current_cost

        # --- 4. Metropolis 接受準則 ---
        # 如果新解更好 (delta < 0)，或者以一定機率接受更差的解。
        # math.exp(-delta / T) 是接受壞解的機率，T 越高機率越大。
        if delta < 0 or random.random() < math.exp(-delta / T):
            # 接受新解作為當前解。
            tour, current_cost = new_tour, new_cost
            # 如果這個解比歷史最佳解還要好，就更新歷史最佳解。
            if current_cost < best_cost:
                best_tour, best_cost = tour[:], current_cost

        # --- 5. 冷卻 ---
        # 將溫度按比例 alpha 降低，為下一次迭代做準備。
        T *= alpha

    return best_tour, best_cost
def calculate_tour_cost(problem, tour):
    """
    依照 tsplib95 的 problem 物件來計算一條路徑的總成本
    tour 必須是 0-based（已經減 1）的節點序列
    """
    total = 0
    n = len(tour)
    for i in range(n):
        # tsplib95.get_weight() 使用 1-based 編號，因此要 +1
        a = tour[i] + 1
        b = tour[(i + 1) % n] + 1
        total += problem.get_weight(a, b)
    return total
# 3. measure_performance()
def measure_performance(cost, n):
    proc = psutil.Process()
    start_wall = time.perf_counter()
    start_cpu = proc.cpu_times()

    # 執行 SA
    tour, best_cost = simulated_annealing(cost, n)

    end_wall = time.perf_counter()
    end_cpu = proc.cpu_times()
    mem_mb = proc.memory_info().rss / (1024 * 1024)

    wall_time = end_wall - start_wall
    cpu_time = ((end_cpu.user - start_cpu.user)
                + (end_cpu.system - start_cpu.system))

    return tour, best_cost, wall_time, cpu_time, mem_mb

# 4. load_optimal_tour()   <<< 這個一定要放這裡，主程式呼叫才找得到
def load_optimal_tour(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    tour = []
    start = False
    for line in lines:
        line = line.strip()
        if start:
            if not line.isdigit() or line in ('-1', 'EOF'):
                break
            tour.append(int(line) - 1)          # 轉成 0-based
        elif line == 'TOUR_SECTION':
            start = True
    return tour
# ------------------------------------------------------------
# 性能量測包裝函式
# ------------------------------------------------------------
def measure_performance(cost_mat, n):
    """封裝效能量測：wall time、CPU time、RSS memory"""
    proc = psutil.Process()

    start_wall = time.perf_counter() # 記錄真實世界流逝時間
    start_cpu  = proc.cpu_times()    # 記錄 CPU 使用時間

    # 執行主演算法
    tour, best_cost = simulated_annealing(cost_mat, n)

    end_wall = time.perf_counter()
    end_cpu  = proc.cpu_times()
    mem_mb   = proc.memory_info().rss / (1024 * 1024) # 常駐記憶體大小 (MB)

    wall_time = end_wall - start_wall
    cpu_time  = ((end_cpu.user - start_cpu.user)
                 + (end_cpu.system - start_cpu.system))

    return tour, best_cost, wall_time, cpu_time, mem_mb
# ---------- 1. 建立距離矩陣 / build cost matrix ----------
def build_cost_matrix(problem):
    """
    Robust  n×n  distance matrix builder.
    * 支援 FULL_MATRIX / UPPER_ROW / LOWER_ROW / 等三角格式
    * 若 (i, j) 查不到就自動改查 (j, i)
    * 回傳純 Python list，可再轉成 NumPy
    """
    n = problem.dimension
    cost = [[0]*n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if i == j:
                continue                              # 對角線保持 0
            a, b = i + 1, j + 1                      # tsplib95 使用 1-based
            try:
                cost[i][j] = problem.get_weight(a, b)
            except IndexError:
                cost[i][j] = problem.get_weight(b, a)

    return cost
# ------------------------------------------------------------
# 主程式
# ------------------------------------------------------------
# ---------- 3. main ----------
if __name__ == "__main__":
    TSP_FILE      = "ulysses16.tsp"
    OPT_TOUR_FILE = "ulysses16.opt.tour"

    if not os.path.exists(TSP_FILE):
        print(f"錯誤：找不到 {TSP_FILE}")
        exit(1)

    problem      = tsplib95.load(TSP_FILE)
    cost_matrix  = build_cost_matrix(problem)   # <<< 換成這行
    n            = problem.dimension

    print(f"正在求解 {TSP_FILE} (n={n}) ...")
    tour, total_cost, wall, cpu, mem = measure_performance(cost_matrix, n)

    print("\n--- SA 結果 ---")
    print(f"前 10 個城市：{tour[:10]} ...")
    print(f"SA 近似成本：{total_cost}")
    print(f"牆鐘時間：{wall:.4f}s | CPU：{cpu:.4f}s | RSS：{mem:.2f}MB")

    # --- 比對最佳解 (若有) ---
    if os.path.exists(OPT_TOUR_FILE):
        opt_tour  = load_optimal_tour(OPT_TOUR_FILE)
        opt_cost  = calculate_tour_cost(problem, opt_tour)
        print("\n--- 與官方 \'.opt.tour\' 比較 ---")
        print(f"官方成本：{opt_cost}")
        print(f"誤差：{((total_cost - opt_cost)/opt_cost)*100:.2f}%")
    else:
        print("\n⚠️ 找不到 .opt.tour，跳過比較")

