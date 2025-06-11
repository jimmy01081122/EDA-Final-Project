import math

def read_tsp_file(filename):
    coords = {}
    with open(filename, 'r') as file:
        lines = file.readlines()
        start = False
        for line in lines:
            if line.strip() == 'NODE_COORD_SECTION':
                start = True
                continue
            if start:
                if line.strip() == 'EOF':
                    break
                parts = line.strip().split()
                if len(parts) >= 3:
                    node_id = int(parts[0])
                    x, y = float(parts[1]), float(parts[2])
                    coords[node_id] = (x, y)
    return coords

def read_opt_tour_file(filename):
    tour = []
    with open(filename, 'r') as file:
        lines = file.readlines()
        start = False
        for line in lines:
            if line.strip() == 'TOUR_SECTION':
                start = True
                continue
            if start:
                if line.strip() == '-1' or line.strip() == 'EOF':
                    break
                tour.append(int(line.strip()))
    return tour

def euclidean_distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def compute_tour_cost(coords, tour):
    cost = 0.0
    n = len(tour)
    for i in range(n):
        city_a = tour[i]
        city_b = tour[(i + 1) % n]  # wrap around to the first city
        cost += euclidean_distance(coords[city_a], coords[city_b])
    return cost

# 替換成你的檔案路徑
tsp_file = 'ch130.tsp'
opt_tour_file = 'ch130.opt.tour'

coords = read_tsp_file(tsp_file)
tour = read_opt_tour_file(opt_tour_file)
cost = compute_tour_cost(coords, tour)

print(f"Optimal tour cost: {cost:.2f}")
