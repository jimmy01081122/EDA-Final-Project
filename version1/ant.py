import numpy as np
import time
import psutil
import tsplib95
import os
import random
import pandas as pd
import matplotlib.pyplot as plt

def aco_tsp(cost_matrix, n, ants=20, iterations=100, alpha=1.0, beta=2.0, rho=0.5, Q=100):
    """
    Solves the TSP using Ant Colony Optimization (ACO).
    This is an improved and corrected version.
    """
    # [OPTIMIZATION] Use a small epsilon to avoid division by zero.
    epsilon = 1e-10
    
    # Heuristic information (inverse of cost), handling zero costs.
    heuristic = 1 / (np.array(cost_matrix) + epsilon)
    
    # Initialize pheromone matrix.
    pheromone = np.ones((n, n))
    
    # Global best tour and its cost.
    best_tour_overall = None
    best_cost_overall = float('inf')
    
    def get_tour_length(tour):
        """Calculates the total cost of a single tour."""
        length = 0
        for i in range(len(tour) - 1):
            u, v = tour[i], tour[i+1]
            length += cost_matrix[u][v]
        return length

    for _ in range(iterations):
        all_tours = []
        
        # 1. All ants construct their tours.
        for _ in range(ants):
            # [OPTIMIZATION] Each ant starts at a random city.
            start_node = random.randint(0, n - 1)
            tour = [start_node]
            visited = {start_node}

            while len(tour) < n:
                current_node = tour[-1]
                
                # Calculate transition probabilities.
                pheromones = pheromone[current_node, :] ** alpha
                heuristics = heuristic[current_node, :] ** beta
                
                prob = pheromones * heuristics
                
                # Mask visited cities.
                prob[list(visited)] = 0
                
                # Handle the rare case where the sum of probabilities is zero.
                if prob.sum() < epsilon:
                    next_node = random.choice([i for i in range(n) if i not in visited])
                else:
                    probabilities = prob / prob.sum()
                    next_node = np.random.choice(range(n), p=probabilities)
                
                tour.append(next_node)
                visited.add(next_node)
            
            # Close the loop for a complete tour
            tour.append(tour[0])
            all_tours.append(tour)
            
        # 2. Pheromone evaporation.
        pheromone *= (1 - rho)
        
        # 3. Pheromone update & find the best tour in the current iteration.
        for tour in all_tours:
            cost_of_tour = get_tour_length(tour)
            
            # Update pheromones based on tour quality.
            for i in range(n):
                u, v = tour[i], tour[i+1]
                # Deposit pheromones on the path in both directions.
                pheromone[u, v] += Q / cost_of_tour
                pheromone[v, u] += Q / cost_of_tour
            
            # Update the overall best tour.
            if cost_of_tour < best_cost_overall:
                best_cost_overall = cost_of_tour
                best_tour_overall = tour
    
    return best_tour_overall, best_cost_overall

def measure_performance(algorithm_func, **kwargs):
    """Measures the performance of a given TSP algorithm."""
    process = psutil.Process(os.getpid())
    start_time = time.perf_counter()
    start_cpu = process.cpu_times()

    tour, cost_val = algorithm_func(**kwargs)

    end_time = time.perf_counter()
    end_cpu = process.cpu_times()
    memory_usage_mb = process.memory_info().rss / (1024 * 1024)

    time_taken_s = end_time - start_time
    cpu_time_s = (end_cpu.user - start_cpu.user) + (end_cpu.system - start_cpu.system)

    return tour, cost_val, time_taken_s, cpu_time_s, memory_usage_mb

def load_optimal_tour(filename):
    """Loads an optimal tour from a .opt.tour file (Robust Version)."""
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
        tour = []
        tour_section_started = False
        for line in lines:
            line_strip = line.strip()
            if line_strip == 'TOUR_SECTION':
                tour_section_started = True
                continue
            if not tour_section_started or not line_strip or line_strip == '-1' or line_strip == 'EOF':
                if tour_section_started: break
                continue
            if line_strip.isdigit():
                tour.append(int(line_strip) - 1)
        if tour:
            tour.append(tour[0]) # Close the loop
        return tour
    except FileNotFoundError:
        return None

def calculate_tour_cost(problem, tour):
    """Calculates the total cost of a given tour (expects a closed tour)."""
    if not tour or len(tour) < 2:
        return 0
    total_cost = 0
    for i in range(len(tour) - 1):
        total_cost += problem.get_weight(tour[i] + 1, tour[i + 1] + 1)
    return total_cost

def process_problem(tsp_filepath):
    """Processes a single TSP problem file and returns a dictionary of results."""
    problem_name = os.path.basename(tsp_filepath).replace('.tsp', '')
    print(f"\n--- Processing: {problem_name} ---")
    
    try:
        problem = tsplib95.load(tsp_filepath)
        nodes = sorted(list(problem.get_nodes()))
        n = len(nodes)
        
        cost_matrix = [[problem.get_weight(nodes[i], nodes[j]) for j in range(n)] for i in range(n)]
        
        # Set ACO parameters here
        aco_params = {'cost_matrix': cost_matrix, 'n': n, 'ants': 30, 'iterations': 150}
        
        print(f"Solving with ACO (n={n}, ants={aco_params['ants']}, iterations={aco_params['iterations']})...")
        tour, cost, time_taken, cpu_time, mem_usage = measure_performance(aco_tsp, **aco_params)
        
        opt_tour_filepath = tsp_filepath.replace('.tsp', '.opt.tour')
        opt_tour = load_optimal_tour(opt_tour_filepath)
        opt_cost = calculate_tour_cost(problem, opt_tour) if opt_tour else None
        
        error_pct = ((cost - opt_cost) / opt_cost * 100) if opt_cost and opt_cost > 0 else None

        print(f"  ACO cost found: {cost}")
        if opt_cost:
            print(f"  Optimal cost: {opt_cost}")
            if error_pct is not None:
                print(f"  Approximation error: {error_pct:.2f}%")
        else:
            print(f"  ⚠️ Could not find or parse optimal tour file: {os.path.basename(opt_tour_filepath)}")

        return {
            "Problem": problem_name, "N_Cities": n, "Found_Tour": ' '.join(map(str, tour)),
            "Found_Cost": cost, "Optimal_Cost": opt_cost, "Error_Pct": error_pct,
            "Time_Taken_s": time_taken, "CPU_Time_s": cpu_time, "Memory_Usage_MB": mem_usage
        }
    except Exception as e:
        import traceback
        print(f"An unexpected error occurred while processing {problem_name}: {e}")
        traceback.print_exc()
        return None

def plot_performance_trends(df):
    """Plots ACO performance metrics across different problems."""
    if df.empty:
        print("No data available for plotting.")
        return
        
    df_sorted = df.sort_values(by='N_Cities')

    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('ACO Performance Analysis', fontsize=20)

    axs[0, 0].bar(df_sorted['Problem'], df_sorted['Error_Pct'], color='skyblue')
    axs[0, 0].set_title('Approximation Error (%)'); axs[0, 0].set_ylabel('Error (%)')
    axs[0, 0].tick_params(axis='x', rotation=45)

    axs[0, 1].plot(df_sorted['Problem'], df_sorted['Time_Taken_s'], marker='o', color='coral')
    axs[0, 1].set_title('Execution Time (seconds)'); axs[0, 1].set_ylabel('Time (s)')
    axs[0, 1].grid(True); axs[0, 1].tick_params(axis='x', rotation=45)

    axs[1, 0].plot(df_sorted['Problem'], df_sorted['CPU_Time_s'], marker='s', color='lightgreen')
    axs[1, 0].set_title('CPU Time (seconds)'); axs[1, 0].set_xlabel('TSPLIB Problem'); axs[1, 0].set_ylabel('CPU Time (s)')
    axs[1, 0].grid(True); axs[1, 0].tick_params(axis='x', rotation=45)
    
    axs[1, 1].plot(df_sorted['Problem'], df_sorted['Memory_Usage_MB'], marker='x', color='plum')
    axs[1, 1].set_title('Memory Usage (MB)'); axs[1, 1].set_xlabel('TSPLIB Problem'); axs[1, 1].set_ylabel('Memory (MB)')
    axs[1, 1].grid(True); axs[1, 1].tick_params(axis='x', rotation=45)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('aco_performance_trends.png')
    print("\n📈 Performance trend plot saved to aco_performance_trends.png")
    plt.show()

def plot_algorithm_comparison(problem_name, results_dict):
    """Plots a comparison of different algorithms for a single problem."""
    df = pd.DataFrame(results_dict).T
    
    fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=False)
    fig.suptitle(f'Algorithm Performance Comparison on "{problem_name}"', fontsize=20)

    df['Cost'].plot(kind='bar', ax=axs[0], color=['#1f77b4', '#ff7f0e', '#2ca02c']); axs[0].set_title('Cost Comparison'); axs[0].set_ylabel('Total Cost'); axs[0].tick_params(axis='x', rotation=15)
    df['Time'].plot(kind='bar', ax=axs[1], color=['#1f77b4', '#ff7f0e', '#2ca02c']); axs[1].set_title('Execution Time Comparison'); axs[1].set_ylabel('Time (seconds)'); axs[1].tick_params(axis='x', rotation=15)
    df['Memory'].plot(kind='bar', ax=axs[2], color=['#1f77b4', '#ff7f0e', '#2ca02c']); axs[2].set_title('Memory Usage Comparison'); axs[2].set_ylabel('Memory (MB)'); axs[2].tick_params(axis='x', rotation=15)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    output_filename = f'comparison_on_{problem_name}.png'
    plt.savefig(output_filename)
    print(f"📊 Algorithm comparison plot saved to {output_filename}")
    plt.show()

if __name__ == "__main__":
    # ❗ Please fill in the paths to your TSPLIB files here.
    tsp_files = [
        'ch130.tsp', 'a280.tsp', 'gr666.tsp', 'pa561.tsp','att48.tsp', 
        'kroA100.tsp', 'ch150.tsp', 'brg180.tsp'
    ]
    
    all_results = []
    for tsp_file in tsp_files:
        if not os.path.exists(tsp_file):
            print(f"\n--- Processing: {os.path.basename(tsp_file).replace('.tsp', '')} ---")
            print(f"⚠️ File not found: '{tsp_file}'. Skipping.")
            continue
        result = process_problem(tsp_file)
        if result:
            all_results.append(result)

    if all_results:
        results_df = pd.DataFrame(all_results)
        log_filename = 'aco_tsp_log.csv'
        results_df.to_csv(log_filename, index=False)
        print(f"\n✅ All results have been logged to {log_filename}")

        plot_performance_trends(results_df)

        first_problem_result = results_df.iloc[0]
        problem_to_compare = first_problem_result['Problem']
        
        comparison_data = {
            'ACO Algorithm': {
                'Cost': first_problem_result['Found_Cost'],
                'Time': first_problem_result['Time_Taken_s'],
                'Memory': first_problem_result['Memory_Usage_MB']
            },
        }
        
        if pd.notna(first_problem_result['Optimal_Cost']):
             comparison_data['Optimal Solution'] = {
                'Cost': first_problem_result['Optimal_Cost'], 'Time': 0, 'Memory': 0
             }
        
        plot_algorithm_comparison(problem_to_compare, comparison_data)
    else:
        print("\nNo valid files were processed. Program finished.")