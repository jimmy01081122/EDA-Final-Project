"""
DP for TSP
Author : Jimmy Chang

Held-Karp 動態規劃解 TSP + 效能量測．
已修正：
1. 終止條件 (mask 全滿後回起點)
2. 回傳路徑 (parent 陣列重建)
3. measure_performance() 與主程式回傳值數量
4. 加入 load_optimal_tour / calculate_tour_cost 範例實作
"""

from __future__ import annotations
import sys, time, psutil, os
import tsplib95
from functools import lru_cache
from typing import List, Tuple

# ---------- Held-Karp DP ---------- #
def tsp_dp(cost: List[List[int]], n: int) -> Tuple[int, List[int]]:
    """
    回傳 (最短成本, tour list；首尾皆為 0)
    """
    FULL_MASK = (1 << n) - 1
    # dp[mask][pos] = (cost, prev)  用 dict 節省記憶體
    dp: list[dict[int, Tuple[int,int]]] = [dict() for _ in range(1 << n)]
    dp[1][0] = (0, -1)                      # 只在節點 0 時成本 0

    for mask in range(1 << n):
        for pos in list(dp[mask].keys()):
            cost_so_far, _ = dp[mask][pos]
            # 嘗試走向下一個沒訪問過的城市
            for nxt in range(n):
                if mask & (1 << nxt):      # visited
                    continue
                new_mask = mask | (1 << nxt)
                new_cost = cost_so_far + cost[pos][nxt]
                if nxt not in dp[new_mask] or new_cost < dp[new_mask][nxt][0]:
                    dp[new_mask][nxt] = (new_cost, pos)

    # 收尾：回到 0
    best_cost = sys.maxsize
    last = -1
    for pos, (c, prev) in dp[FULL_MASK].items():
        total = c + cost[pos][0]
        if total < best_cost:
            best_cost, last = total, pos

    # reconstruct path
    path = [0] * (n + 1)
    mask, idx = FULL_MASK, last
    for i in range(n - 1, 0, -1):
        path[i] = idx
        _, idx = dp[mask][idx]
        mask ^= 1 << path[i]
    path[-1] = 0  # 回起點

    return best_cost, path


# ---------- 效能量測 ---------- #
def measure_performance(cost, n):
    process = psutil.Process()
    start_wall = time.perf_counter()
    start_cpu  = process.cpu_times()

    total_cost, tour = tsp_dp(cost, n)

    end_wall = time.perf_counter()
    end_cpu  = process.cpu_times()
    mem_mb   = process.memory_info().rss / (1024 ** 2)

    return total_cost, tour, end_wall - start_wall, \
           (end_cpu.user - start_cpu.user) + (end_cpu.system - start_cpu.system), \
           mem_mb


# ---------- 讀最佳解 (若有) ---------- #
def load_optimal_tour(tour_file: str) -> list[int]:
    with open(tour_file) as f:
        raw = [int(x) for x in f.read().split() if x.isdigit()]
    return raw + [raw[0]]       # 令首尾相同

def calculate_tour_cost(problem, tour: list[int]) -> int:
    return sum(problem.get_weight(tour[i], tour[i+1]) for i in range(len(tour)-1))


# ---------- main ---------- #
if __name__ == "__main__":
    TSP_FILE      = "gr17.tsp"
    OPT_TOUR_FILE = "gr17.opt.tour"

    if not os.path.exists(TSP_FILE):
        sys.exit(f"❌ TSP 檔 '{TSP_FILE}' 不存在，請下載後重試。")

    # 1. 建成本矩陣
    prob   = tsplib95.load(TSP_FILE)
    nodes  = sorted(prob.get_nodes())          # 轉成 0..n-1
    idx_of = {node: i for i, node in enumerate(nodes)}
    n      = len(nodes)

    cost_m = [[prob.get_weight(u, v) for v in nodes] for u in nodes]

    # 2. 執行 DP + 量測
    print(f"▶ 使用 Held-Karp DP 求解 '{TSP_FILE}' (n={n})")
    best_cost, best_tour, wall, cpu, mem = measure_performance(cost_m, n)

    # 3. 結果
    print("\n--- 結果 ---")
    print(f"最短成本 = {best_cost}")
    # print(f"完整路徑 = {best_tour}")
    print(f"Wall-clock 時間 = {wall:.4f}s")
    print(f"CPU 時間 = {cpu:.4f}s")
    print(f"常駐記憶體 = {mem:.2f} MB")

    # 4. 與官方最佳比較 (若有)
    if os.path.exists(OPT_TOUR_FILE):
        opt_tour_nodes = load_optimal_tour(OPT_TOUR_FILE)
        # 轉成 0..n-1 index 方便比對
        opt_tour_idx = [idx_of[x] for x in opt_tour_nodes]
        opt_cost     = calculate_tour_cost(prob, opt_tour_nodes)

        print("\n--- 與官方最佳比較 ---")
        print(f"官方最佳成本 = {opt_cost}")
        gap = (best_cost - opt_cost) / opt_cost * 100
        print(f"誤差 = {gap:.2f}%")
    else:
        print("\n⚠️ 未提供最佳解檔，略過比較")
