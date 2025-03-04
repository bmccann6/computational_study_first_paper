import json
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint, pformat
import time
import argparse
from matplotlib.ticker import FuncFormatter
import pickle
from gurobipy import Model, GRB, quicksum
import entropy_plot_input_variables
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

def validate_data():
    for year, drugs in resource_sets.items():
        for drug in drugs:
            if drug not in item_vals:
                raise ValueError(f"Drug '{drug}' in year '{year}' is not a valid item in item_vals")

def calculate_backloading_stack_values(resource_set_dict, item_vals, capacities):
    hider_resources_sorted = sorted(resource_set_dict.items(), key=lambda x: item_vals[x[0]])
    local_copy_capacities = capacities.copy()
    backloading_stack_values = {i: 0 for i in range(len(capacities))}
    i = 0
    while local_copy_capacities and hider_resources_sorted:
        if local_copy_capacities[0] == 0:
            local_copy_capacities.pop(0)
            i += 1
            continue
        if hider_resources_sorted[0][1] == 0:
            hider_resources_sorted.pop(0)
            continue
        assignment_amount = min(local_copy_capacities[0], hider_resources_sorted[0][1])
        item_name = hider_resources_sorted[0][0]
        item_value = item_vals[item_name]
        backloading_stack_values[i] += assignment_amount * item_value
        local_copy_capacities[0] -= assignment_amount
        hider_resources_sorted[0] = (hider_resources_sorted[0][0], hider_resources_sorted[0][1] - assignment_amount)
    return backloading_stack_values

def calculate_breakpoints(backloading_stack_values, capacities):
    n = len(capacities)
    i_0 = 0
    l = 0
    breakpoints = [i_0]
    while True:
        current_max = float('-inf')
        best_k = None
        for k in range(breakpoints[l], n):
            segment_sum = sum(backloading_stack_values[j] for j in range(breakpoints[l], k + 1))
            segment_length = k + 1 - breakpoints[l]
            avg = segment_sum / segment_length
            if avg >= current_max:
                current_max = avg
                best_k = k + 1
        breakpoints.append(best_k)
        if breakpoints[l+1] == n:
            return breakpoints
        else:
            l += 1

def calculate_expected_value_under_equilibrium_each_node(resource_set_dict, item_vals, capacities):  
    backloading_stack_values = calculate_backloading_stack_values(resource_set_dict, item_vals, capacities)
    breakpoints = calculate_breakpoints(backloading_stack_values, capacities)    
    expected_value_under_equilibrium_each_node = {}
    for node in range(len(backloading_stack_values)):
        relevant_indices = [
            i for i in range(len(breakpoints) - 1)
            if breakpoints[i] <= node < breakpoints[i + 1]
        ]
        expected_value_under_equilibrium_each_node[node] = np.mean([
            backloading_stack_values[j]
            for j in range(breakpoints[relevant_indices[0]], breakpoints[relevant_indices[0] + 1])
        ])
    return expected_value_under_equilibrium_each_node, breakpoints

def compute_all_years_equilibrium_values(resource_sets, item_vals, sizes_hiding_locations_each_year):
    year_to_equilibrium_node_values = {}
    for year, resource_set_dict in resource_sets.items():
        eq_vals, _ = calculate_expected_value_under_equilibrium_each_node(resource_set_dict, item_vals, sizes_hiding_locations_each_year[year])
        year_to_equilibrium_node_values[year] = eq_vals
    return year_to_equilibrium_node_values

def compute_resource_sets_info_for_json():
    resource_sets_info = []
    for year, resource_set_dict in resource_sets.items():
        eq_vals, breakpoints = calculate_expected_value_under_equilibrium_each_node(resource_set_dict, item_vals, sizes_hiding_locations_each_year[year])
        resource_sets_info.append({
            "year": year,
            "breakpoints": breakpoints,
            "expected value of items at each node in equilibrium": eq_vals
        })
    return resource_sets_info

def calculate_expected_value_each_node_this_prob_dist(prob_dist, precomputed_equilib_vals, NUM_HIDING_LOCATIONS):
    expected_value_each_node_under_this_prob_dist = {i: 0 for i in range(NUM_HIDING_LOCATIONS)}
    for year, eq_vals in precomputed_equilib_vals.items():
        for node in range(NUM_HIDING_LOCATIONS):
            expected_value_each_node_under_this_prob_dist[node] += prob_dist[year] * eq_vals[node]
    return expected_value_each_node_under_this_prob_dist

def get_optimal_list_detectors_this_prob_dist(budget, NUM_HIDING_LOCATIONS, detectors, expected_value_each_node_under_this_prob_dist):
    nodes = range(NUM_HIDING_LOCATIONS)
    m = Model("integer_program")
    m.setParam('OutputFlag', 0)
    x = m.addVars(detectors, nodes, vtype=GRB.BINARY, name="x")
    objective = quicksum(expected_value_each_node_under_this_prob_dist[node] * detectors[detector]["accuracy"] * x[detector, node]
                         for detector in detectors for node in nodes)
    m.setObjective(objective, GRB.MAXIMIZE)
    m.addConstr(quicksum(detectors[detector]["cost"] * x[detector, node]
                         for detector in detectors for node in nodes) <= budget, "budget_constraint")
    for node in nodes:
        m.addConstr(quicksum(x[detector, node] for detector in detectors) <= 1, f"selection_constraint_{node}")
    m.optimize()
    optimal_list = [(detector, detectors[detector]['accuracy'])
                    for detector in detectors for node in nodes if x[detector, node].x > 0]
    return optimal_list

def calculate_expected_and_total_values_detected_this_prob_dist(prob_dist, precomputed_equilib_vals, NUM_HIDING_LOCATIONS, detectors, budget):
    expected_value_each_node = calculate_expected_value_each_node_this_prob_dist(prob_dist, precomputed_equilib_vals, NUM_HIDING_LOCATIONS)
    optimal_list = get_optimal_list_detectors_this_prob_dist(budget, NUM_HIDING_LOCATIONS, detectors, expected_value_each_node)
    detector_accuracies = sorted([accuracy for _, accuracy in optimal_list], reverse=True) + [0] * (NUM_HIDING_LOCATIONS - len(optimal_list))
    expected_value_detected = sum(detector_accuracies[i] * expected_value_each_node[i] for i in range(len(expected_value_each_node)))
    expected_total_value = sum(expected_value_each_node.values())
    return expected_value_detected, expected_total_value

def plot_entropy_vs_final_fraction_value_detected(bins_data):
    x_labels = list(bins_data.keys())
    y_values = [bins_data[k]["final_fraction_value_detected"] for k in bins_data]
    data_range = max(y_values) - min(y_values)
    y_buffer = data_range * 0.05
    y_min = min(y_values) - y_buffer
    y_max = max(y_values) + y_buffer
    y_ticks = np.linspace(y_min, y_max, 10)
    def format_y_value(y, pos):
        return f'{y:.4f}'
    plt.figure(figsize=(15, 8))
    plt.title("Normalized Entropy vs. Fraction Detected", fontsize=14)
    plt.bar(x_labels, y_values)
    plt.xticks(rotation=45, fontsize=11)
    plt.xlabel("\nEntropy Bins", fontsize=14)
    plt.ylabel("Average Fraction Detected\n", fontsize=14)
    plt.yticks(y_ticks, [format_y_value(y, None) for y in y_ticks], fontsize=11)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(format_y_value))
    plt.ylim(y_min, y_max)
    plt.subplots_adjust(bottom=0.2, top=0.9)
    plt.savefig(f"entropy_plots/entropy_vs_fraction_detected_plots/entropy_vs_fraction_detected_budget_{budget}_and_{NUM_SAMPLES_PER_BIN}_samples_per_bin_and_NUM_BINS_{NUM_BINS}.png")

def plot_entropy_vs_final_expected_value_detected(bins_data):
    x_labels = list(bins_data.keys())
    y_values = [bins_data[k]["final_expected_value_detected"] for k in bins_data]
    data_range = max(y_values) - min(y_values)
    y_buffer = data_range * 0.05
    y_min = min(y_values) - y_buffer
    y_max = max(y_values) + y_buffer
    y_ticks = np.linspace(y_min, y_max, 10)
    def format_y_value(y, pos):
        if y >= 1e9:
            return f'${y/1e9:.2f}B'
        elif y >= 1e6:
            return f'${y/1e6:.2f}M'
        elif y >= 1e3:
            return f'${y/1e3:.2f}K'
        return f'${y:.2f}'
    plt.figure(figsize=(15, 8))
    plt.title("Normalized Entropy vs. Expected Value Detected (in billions USD)", fontsize=14)
    plt.bar(x_labels, y_values)
    plt.xticks(rotation=45, fontsize=11)
    plt.xlabel("\nEntropy Bins", fontsize=14)
    plt.ylabel("Average Value Detected\n", fontsize=14)
    plt.yticks(y_ticks, [format_y_value(y, None) for y in y_ticks], fontsize=11)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(format_y_value))
    plt.ylim(y_min, y_max)
    plt.subplots_adjust(bottom=0.2, top=0.9)
    plt.savefig(f"entropy_plots/entropy_vs_value_detected_plots/entropy_vs_value_detected_budget_{budget}_and_{NUM_SAMPLES_PER_BIN}_samples_per_bin_and_NUM_BINS_{NUM_BINS}.png")

def process_prob_dist(entropy_range, entry, precomputed_equilib_vals, NUM_HIDING_LOCATIONS, detectors, budget):
    try:
        prob_dist = entry["prob_dist"]
        result = calculate_expected_and_total_values_detected_this_prob_dist(prob_dist,
                                                                             precomputed_equilib_vals,
                                                                             NUM_HIDING_LOCATIONS,
                                                                             detectors,
                                                                             budget)
        bin_key = f"{entropy_range[0]:.2f}-{entropy_range[1]:.2f}"
        return (bin_key, result)
    except Exception as e:
        print(f"Error in worker: {e}")
        raise

def main():
    start_time = time.time()
    bins_data = {f"{i/NUM_BINS:.2f}-{(i+1)/NUM_BINS:.2f}": {"expected_values_detected": [], "expected_total_values": []}
                 for i in range(NUM_BINS)}
    tasks = []
    for entropy_range, entries in prob_distributions_dict.items():
        for entry in entries:
            tasks.append((entropy_range, entry))
    total_tasks = len(tasks)
    completed = 0
    # Use a spawn context to avoid Gurobi issues with forked processes.
    with ProcessPoolExecutor(mp_context=mp.get_context("spawn")) as executor:
        futures = [executor.submit(process_prob_dist, entropy_range, entry,
                                   precomputed_equilib_vals, NUM_HIDING_LOCATIONS, detectors, budget)
                   for entropy_range, entry in tasks]
        for future in as_completed(futures):
            try:
                bin_key, (ev_detected, tot_val) = future.result()
                bins_data[bin_key]["expected_values_detected"].append(ev_detected)
                bins_data[bin_key]["expected_total_values"].append(tot_val)
            except Exception as e:
                print(f"Worker raised exception: {e}")
            completed += 1
            print(f"\rNumber of scenarios done: {completed}/{total_tasks}. Time elapsed: {time.time() - start_time:.2f} seconds", end="")
    print()
    for key in bins_data:
        total_detected = sum(bins_data[key]["expected_values_detected"])
        total_all = sum(bins_data[key]["expected_total_values"])
        bins_data[key]["final_fraction_value_detected"] = total_detected / total_all if total_all != 0 else 0
        bins_data[key]["final_expected_value_detected"] = total_detected / NUM_SAMPLES_PER_BIN
    return bins_data

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Load configuration for entropy plot calculations.")
    parser.add_argument('-config', type=str, required=True, help='Path to the JSON configuration file.')
    parser.add_argument('-prob_distributions', type=str, required=True, help='Path to the Pickle file of probability distributions')
    args = parser.parse_args()
    item_vals, resource_sets, _, hiding_locations, NUM_HIDING_LOCATIONS, sizes_hiding_locations_each_year, detectors, budget, NUM_SAMPLES_PER_BIN, NUM_BINS = entropy_plot_input_variables.get_configuration(args.config)
    validate_data()
    precomputed_equilib_vals = compute_all_years_equilibrium_values(resource_sets, item_vals, sizes_hiding_locations_each_year)
    with open(args.prob_distributions, 'rb') as file:
        prob_distributions_dict = pickle.load(file)
    bins_at_end = main()
    resource_sets_info = compute_resource_sets_info_for_json()
    json_data = {"resource_sets_info": resource_sets_info, "bins_at_end": bins_at_end}
    print("Creating json...")
    with open(f"output_data_entropy_plots/output_data_entropy_plot_budget_{budget}_and_{NUM_SAMPLES_PER_BIN}_samples_per_bin_and_NUM_BINS_{NUM_BINS}.json", "w") as f:
        json.dump(json_data, f, indent=4)
    print("Finished creating json.")
    plot_entropy_vs_final_fraction_value_detected(bins_at_end)
    plot_entropy_vs_final_expected_value_detected(bins_at_end)
