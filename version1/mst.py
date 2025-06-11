import sys
import heapq
import time
import psutil
from collections import defaultdict
import tsplib95

def mst_tsp(cost, n):
    def prim(graph):
        visited = [False] * n
        parent = [None] * n
        key = [sys.maxsize] * n
        key[0] = 0
        parent[0] = -1
        pq = [(0, 0)]

        while pq:
            u = heapq.heappop(pq)[1]
            visited[u] = True
            for v in range(n):
                if graph[u][v] and not visited[v] and graph[u][v] < key[v]:
                    key[v] = graph[u][v]
                    parent[v] = u
                    heapq.heappush(pq, (key[v], v))

        mst = defaultdict(list)
        for i in range(1, n):
            if parent[i] is not None:
                mst[parent[i]].append(i)
                mst[i].append(parent[i])
        return mst

    def preorder(mst, start):
        visited = [False] * n
        tour = []

        def dfs(node):
            visited[node] = True
            tour.append(node)
            for neighbor in mst[node]:
                if not visited[neighbor]:
                    dfs(neighbor)

        dfs(start)
        return tour

    mst = prim(cost)
    tour = preorder(mst, 0)
    tour.append(0)

    total_cost = 0
    for i in range(len(tour) - 1):
        total_cost += cost[tour[i]][tour[i + 1]]

    return tour, total_cost

def measure_performance(cost, n):
    process = psutil.Process()
    start_time = time.perf_counter()
    start_cpu = process.cpu_times()

    tour, cost_val = mst_tsp(cost, n)

    end_time = time.perf_counter()
    end_cpu = process.cpu_times()
    memory_usage = process.memory_info().rss / (1024 * 1024)

    time_taken = end_time - start_time
    cpu_time = (end_cpu.user - start_cpu.user) + (end_cpu.system - start_cpu.system)

    return tour, cost_val, time_taken, cpu_time, memory_usage

def load_optimal_tour(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    tour = []
    start = False
    for line in lines:
        if line.strip() == 'TOUR_SECTION':
            start = True
            continue
        if not start:
            continue
        city = int(line.strip())
        if city == -1:
            break
        tour.append(city - 1)  # 轉為 0-based
    tour.append(tour[0])
    return tour

def calculate_tour_cost(problem, tour):
    total = 0
    for i in range(len(tour) - 1):
        total += problem.get_weight(tour[i] + 1, tour[i + 1] + 1)  # 注意 tsplib95 是 1-based
    return total

if __name__ == "__main__":
    problem = tsplib95.load('ulysses16.tsp')
    nodes = list(problem.get_nodes())
    nodes.sort()
    n = len(nodes)

    cost_matrix = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            cost_matrix[i][j] = problem.get_weight(nodes[i], nodes[j])

    tour, total_cost, time_taken, cpu_time, memory_usage = measure_performance(cost_matrix, n)

    print(f"TSP 路徑: {tour}")
    print(f"MST 近似總成本: {total_cost}")
    print(f"執行時間: {time_taken:.4f} 秒")
    print(f"CPU 時間: {cpu_time:.4f} 秒")
    print(f"記憶體使用量: {memory_usage:.2f} MB")

    # 加入最佳解比較
    try:
        opt_tour = load_optimal_tour('ch130.opt.tour')
        opt_cost = calculate_tour_cost(problem, opt_tour)
        print(f"\n最佳總成本: {opt_cost}")
        print(f"誤差百分比: {(total_cost - opt_cost) / opt_cost * 100:.2f}%")
    except FileNotFoundError:
        print("\n⚠️ 找不到 ch130.opt.tour，跳過最佳解比較。")