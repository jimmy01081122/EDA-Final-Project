import numpy as np
import time
import psutil
import tsplib95
from pathlib import Path

"""
ANT for TSP
Author : Jimmy Chang

Ant-Colony-Optimization (ACO) solver for the Traveling Salesman Problem (TSP)
==========================================================================
此模組提供了:
  • compute_distance_matrix(): 從 TSPLIB 實例建立一個 NumPy 距離矩陣，
                               **不**使用 problem.get_graph() / NetworkX。
  • aco_tsp():                 核心的 ACO 元啟發式演算法實現，帶有詳細的註解。
  • measure_performance():     一個方便的包裝函式，執行 aco_tsp() 並記錄效能。
  • CLI 展示:                 (python aco_tsp_with_comments.py <instance.tsp>)
                               同時會嘗試載入對應的 <instance.opt.tour> 並比較成本。
"""

# ---------------------------------------------------------------------------
#  工具函式: 建立距離矩陣 (不使用 NetworkX)
# ---------------------------------------------------------------------------

def compute_distance_matrix(problem: tsplib95.models.Problem) -> np.ndarray:
    """回傳一個 n × n 的 NumPy 邊權重矩陣。"""
    n = problem.dimension
    dist = np.empty((n, n), dtype=float)

    # TSPLIB 的節點索引從 1 開始，我們需要轉換為 0-based
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            dist[i - 1, j - 1] = problem.get_weight(i, j)
    return dist

# ---------------------------------------------------------------------------
#  核心演算法: Ant Colony Optimization for TSP
# ---------------------------------------------------------------------------

def aco_tsp(
    dist: np.ndarray,
    n: int,
    *,
    ants: int = 10,
    iterations: int = 100,
    alpha: float = 1.0,
    beta: float = 2.0,
    rho: float = 0.5,
    Q: float = 1.0,
    random_start: bool = True,
) -> tuple[list[int], float]:
    """使用經典的蟻群演算法 (ACO) 求解 TSP。

    Parameters
    ----------
    dist : np.ndarray
        n × n 的成對距離矩陣。
    n : int
        城市數量。
    ants : int, default 10
        每次迭代中模擬的螞蟻數量。
    iterations : int, default 100
        ACO 迴圈的總迭代次數。
    alpha : float, default 1.0
        費洛蒙 (τ) 在轉移規則中的影響力。
    beta : float, default 2.0
        啟發式資訊 (η = 1 / d) 在轉移規則中的影響力。
    rho : float, default 0.5
        全域費洛蒙蒸發率 (0 < ρ ≤ 1)。
    Q : float, default 1.0
        費洛蒙沉積時使用的常數: Δτ = Q / L_tour。
    random_start : bool, default True
        每隻螞蟻是否從隨機城市出發（推薦），或全部從城市 0 開始。

    Returns
    -------
    best_tour : list[int]
        找到的最佳路徑 (0-based 城市索引，長度為 n)。
    best_cost : float
        best_tour 的成本。
    """

    # --- 1. 初始化 ---
    # 初始費洛蒙濃度：一個小的常數 (τ₀)，這裡設為 1/n。
    pheromone = np.full((n, n), 1.0 / n, dtype=float)

    # 追蹤歷史最佳解
    best_tour: list[int] | None = None
    best_cost = float("inf")

    # 一個快速計算路徑長度的輔助函式
    def tour_length(tour: list[int]) -> float:
        return sum(dist[tour[i], tour[(i + 1) % n]] for i in range(n))

    # 使用 NumPy 的隨機數生成器以獲得更好的效能
    rng = np.random.default_rng()

    # ------------------------ 2. ACO 主迴圈 ----------------------------------
    for _ in range(iterations):
        all_tours: list[list[int]] = [] # 儲存當前世代所有螞蟻的路徑

        # ~~~ 2a. 每隻螞蟻建構一條路徑 ~~~
        for _ in range(ants):
            # 決定螞蟻的起始城市
            start = int(rng.integers(n)) if random_start else 0
            
            tour = [start]
            visited = np.zeros(n, dtype=bool)
            visited[start] = True

            # 螞蟻需要走 n-1 步來訪問所有城市
            for _ in range(1, n):
                current = tour[-1] # 當前所在的城市

                # --- 計算到所有**未訪問**城市的轉移機率 ---
                # 啟發式資訊 (η)，即距離的倒數。距離越短，η 越大。
                heuristic = 1.0 / dist[current]
                heuristic[heuristic == np.inf] = 0  # 處理除以零的情況 (雖然在 TSP 中罕見)

                # 計算機率的分子部分： (費洛蒙^α) * (啟發式資訊^β)
                # `(~visited)` 是一個布林遮罩，只計算未訪問城市的機率。
                probs_raw = (
                    pheromone[current] ** alpha * heuristic ** beta * (~visited)
                )
                total = probs_raw.sum()

                # 數值穩定性檢查：如果螞蟻被困住（不應發生），則隨機選一個
                if total == 0:
                    candidates = np.where(~visited)[0]
                    next_city = int(rng.choice(candidates))
                else:
                    # 歸一化得到機率分佈
                    probs = probs_raw / total
                    # 根據機率分佈輪盤賭選擇下一個城市
                    next_city = int(rng.choice(n, p=probs))

                tour.append(next_city)
                visited[next_city] = True

            all_tours.append(tour)

        # ~~~ 3. 全域費洛蒙更新 (蒸發 + 沉積) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 3a. 蒸發: 所有路徑上的費洛蒙都減少
        pheromone *= (1.0 - rho)

        # 3b. 沉積: 每隻螞蟻根據路徑長度增加費洛蒙
        for tour in all_tours:
            L = tour_length(tour)
            delta = Q / L  # 路徑越短，增加的費洛蒙越多
            for i in range(n):
                a, b = tour[i], tour[(i + 1) % n]
                pheromone[a, b] += delta
                pheromone[b, a] += delta  # 對稱 TSP，雙向更新

            # 同時更新歷史最佳解
            if L < best_cost:
                best_cost = L
                best_tour = tour

    assert best_tour is not None # 給靜態分析器提示，確保 best_tour 不會是 None
    return best_tour, best_cost

# ---------------------------------------------------------------------------
#  效能包裝函式
# ---------------------------------------------------------------------------

def measure_performance(dist: np.ndarray, n: int, **aco_kwargs):
    """執行 aco_tsp() 一次並測量運行時統計數據。"""
    process = psutil.Process()
    start_wall = time.perf_counter()
    start_cpu = process.cpu_times()

    tour, cost = aco_tsp(dist, n, **aco_kwargs)

    end_wall = time.perf_counter()
    end_cpu = process.cpu_times()
    mem_mb = process.memory_info().rss / (1024 * 1024)
    wall_time = end_wall - start_wall
    cpu_time = (end_cpu.user - start_cpu.user) + (end_cpu.system - start_cpu.system)

    return tour, cost, wall_time, cpu_time, mem_mb

# ---------------------------------------------------------------------------
#  輔助函式: 載入官方最佳解以供比較
# ---------------------------------------------------------------------------

def load_opt_tour(instance_path: Path, dist: np.ndarray) -> tuple[list[int], float] | None:
    """嘗試載入 <instance.opt.tour> 檔案。"""
    tour_path = instance_path.with_suffix(".opt.tour")
    if not tour_path.exists():
        return None

    tour_file = tsplib95.load(tour_path)
    # 官方解可能包含多個 tour，我們取第一個
    node_list: list[int] = tour_file.tours[0]  # 1-based 索引
    # 轉換為 0-based 並移除可能的結束符 -1
    node_list = [v - 1 for v in node_list if v > 0]
    cost = sum(dist[node_list[i], node_list[(i + 1) % len(node_list)]] for i in range(len(node_list)))
    return node_list, cost

# ---------------------------------------------------------------------------
#  命令列介面展示
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse, sys

    parser = argparse.ArgumentParser(description="使用蟻群演算法求解 TSPLIB 實例")
    parser.add_argument("instance", help="<name>.tsp 檔案的路徑")
    parser.add_argument("--ants", type=int, default=20, help="每次迭代的螞蟻數量 (預設: 20)")
    parser.add_argument("--iter", type=int, default=250, help="迭代次數 (預設: 250)")
    args = parser.parse_args()

    tsp_path = Path(args.instance)
    if not tsp_path.exists():
        print(f"[錯誤] 檔案 '{tsp_path}' 不存在", file=sys.stderr)
        sys.exit(1)

    problem = tsplib95.load(tsp_path)
    dist_matrix = compute_distance_matrix(problem)
    n_cities = problem.dimension

    tour, cost, wall, cpu, mem = measure_performance(
        dist_matrix,
        n_cities,
        ants=args.ants,
        iterations=args.iter,
    )

    print("────────────────────────────────────────────────────────────────")
    print(f"實例          : {tsp_path.name}  (n = {n_cities})")
    print(f"找到的最佳成本: {cost:,.2f}")
    print(f"執行時間      : {wall:.3f} s  |  CPU: {cpu:.3f} s  |  記憶體: {mem:.1f} MB")

    opt = load_opt_tour(tsp_path, dist_matrix)
    if opt is not None:
        opt_tour, opt_cost = opt
        gap = (cost - opt_cost) / opt_cost * 100
        print(f"已知最佳解    : {opt_cost:,.2f}  (誤差 = {gap:+.2f} %)")
    else:
        print("[資訊] 找不到 .opt.tour 檔案，跳過誤差比較。")

    print("路徑 (前20個城市):", tour[:20], "..." if n_cities > 20 else "")