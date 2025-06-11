import random
import math
import time
import psutil
import tsplib95
import os
import pandas as pd
import matplotlib.pyplot as plt

def simulated_annealing(cost_matrix, n, iterations=50000, start_temp=1000, alpha=0.999):
    """
    Solves the TSP using Simulated Annealing (SA) with an efficient O(1) delta calculation.
    """
    # Generate an initial random tour
    current_tour = list(range(n))
    random.shuffle(current_tour)

    # Calculate the cost of the initial tour
    def get_tour_cost(tour):
        # This version of the tour is an open path, so we use % n to close the loop for cost calculation
        return sum(cost_matrix[tour[i]][tour[(i + 1) % n]] for i in range(n))

    current_cost = get_tour_cost(current_tour)
    best_tour = list(current_tour)
    best_cost = current_cost
    temp = start_temp

    for _ in range(iterations):
        # Generate a neighbor solution using a 2-opt swap
        i, j = sorted(random.sample(range(n), 2))
        
        # --- [CORE OPTIMIZATION]: Calculate cost difference (delta) in O(1) ---
        # Old edges to be removed: (i-1 -> i) and (j -> j+1)
        # New edges to be added: (i-1 -> j) and (i -> j+1)
        
        # Get node indices from the tour
        prev_i_node = current_tour[i - 1]
        i_node = current_tour[i]
        j_node = current_tour[j]
        next_j_node = current_tour[(j + 1) % n]
        
        # Calculate costs of edges being removed and added
        cost_removed = cost_matrix[prev_i_node][i_node] + cost_matrix[j_node][next_j_node]
        cost_added = cost_matrix[prev_i_node][j_node] + cost_matrix[i_node][next_j_node]
        
        delta = cost_added - cost_removed
        
        # Metropolis acceptance criterion
        if delta < 0 or random.random() < math.exp(-delta / temp):
            # Accept the new solution: perform the 2-opt swap and update cost
            current_tour[i:j+1] = current_tour[i:j+1][::-1]
            current_cost += delta
            
            # If the new solution is the best so far, save it
            if current_cost < best_cost:
                best_cost = current_cost
                best_tour = list(current_tour)
        
        # Cool down the temperature
        temp *= alpha
        
    # Return the best tour found as a closed loop
    final_tour = list(best_tour)
    final_tour.append(final_tour[0])

    return final_tour, best_cost

# --- Standardized helper functions ---

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
        
        # Set Simulated Annealing parameters here
        sa_params = {
            'cost_matrix': cost_matrix, 'n': n, 
            'iterations': 80000, 'start_temp': 10000, 'alpha': 0.9995
        }
        
        print(f"Solving with Simulated Annealing (n={n}, iterations={sa_params['iterations']})...")
        tour, cost, time_taken, cpu_time, mem_usage = measure_performance(simulated_annealing, **sa_params)
        
        opt_tour_filepath = tsp_filepath.replace('.tsp', '.opt.tour')
        opt_tour = load_optimal_tour(opt_tour_filepath)
        opt_cost = calculate_tour_cost(problem, opt_tour) if opt_tour else None
        
        error_pct = ((cost - opt_cost) / opt_cost * 100) if opt_cost and opt_cost > 0 else None

        print(f"  SA cost found: {cost:.2f}")
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
    """Plots SA performance metrics across different problems."""
    if df.empty:
        print("No data available for plotting.")
        return
        
    df_sorted = df.sort_values(by='N_Cities')

    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Simulated Annealing Performance Analysis', fontsize=20)

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
    plt.savefig('sa_performance_trends.png')
    print("\n📈 Performance trend plot saved to sa_performance_trends.png")
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
        'gr202.tsp',
        'ch150.tsp',
        'kroA100.tsp',
        'ch130.tsp',
        # 'a280.tsp', # Note: Larger problems will take longer for SA
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
        log_filename = 'sa_tsp_log.csv'
        results_df.to_csv(log_filename, index=False)
        print(f"\n✅ All results have been logged to {log_filename}")

        plot_performance_trends(results_df)

        first_problem_result = results_df.iloc[0]
        problem_to_compare = first_problem_result['Problem']
        
        comparison_data = {
            'Simulated Annealing': {
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