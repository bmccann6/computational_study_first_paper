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


def validate_data():
    """Ensure the input variables are consistent."""
    for year, drugs in resource_sets.items():
        for drug in drugs:
            if drug not in item_vals:
                raise ValueError(f"Drug '{drug}' in year '{year}' is not a valid item in item_vals")


# NOTE: These are the A_i values in the algorithm mathieu wrote. NOTE! They are not the expected value of items at each node i.
def calculate_backloading_stack_values(resource_set_dict, item_vals, capacities):
    hider_resources_sorted = sorted(resource_set_dict.items(), key=lambda x: item_vals[x[0]])
    local_copy_capacities = capacities.copy()          # We make a local copy because we will be modifying the sizes of the hiding locations as we fill them up. But we don't want to change capacities because we need that in later calls.
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

def compute_all_years_equilibrium_values_at_each_node(resource_sets, item_vals, sizes_hiding_locations_each_year):
    """Returns dict: year -> {node: eq_value}, also captures breakpoints if needed."""
    year_to_equilibrium_node_values = {}
    for year, resource_set_dict in resource_sets.items():
        eq_vals, _ = calculate_expected_value_under_equilibrium_each_node(resource_set_dict, item_vals, sizes_hiding_locations_each_year[year])
        year_to_equilibrium_node_values[year] = eq_vals
        
    return year_to_equilibrium_node_values

def compute_resource_sets_info_for_json():
    """
    Since resource_sets_info is deterministic and does not depend on
    the probability distribution, compute it exactly once here.
    """
    resource_sets_info = []
    for year, resource_set_dict in resource_sets.items():
        expected_value_items_each_node_this_prob_dist_specific_resource_set, breakpoints = calculate_expected_value_under_equilibrium_each_node(resource_set_dict, item_vals, sizes_hiding_locations_each_year[year])
        resource_sets_info.append({
            "year": year,
            "breakpoints": breakpoints,
            "expected value of items at each node in equilibrium": expected_value_items_each_node_this_prob_dist_specific_resource_set
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
    m.setParam('OutputFlag', 0)    # Suppress the Gurobi output to console
    
    x = m.addVars(detectors, nodes, vtype=GRB.BINARY, name="x")  # Binary decision variables
    objective = quicksum(expected_value_each_node_under_this_prob_dist[node] * detectors[detector_type]["accuracy"] * x[detector_type, node] for detector_type in detectors for node in nodes)
    m.setObjective(objective, GRB.MAXIMIZE)

    m.addConstr(quicksum(detectors[detector_type]["cost"] * x[detector_type, node] for detector_type in detectors for node in nodes) <= budget, "budget_constraint")

    # Each item can be selected at most once across all t
    for node in nodes:
        m.addConstr(quicksum(x[detector_type, node] for detector_type in detectors) <= 1, f"selection_constraint_{node}")

    m.optimize()
            
    # Create list of tuples with (detector_type_name, accuracy, x_value)
    optimal_list_detectors_this_prob_dist = [(detector_type, detectors[detector_type]['accuracy'])
                                                for detector_type in detectors for node in nodes if x[detector_type, node].x > 0 ]
        
    return optimal_list_detectors_this_prob_dist
    
def calculate_expected_and_total_values_detected_this_prob_dist(prob_dist, precomputed_equilib_vals, NUM_HIDING_LOCATIONS, detectors, budget):
    
    expected_value_each_node_this_prob_dist = calculate_expected_value_each_node_this_prob_dist(prob_dist, precomputed_equilib_vals, NUM_HIDING_LOCATIONS)
    optimal_list_detectors_this_prob_dist = get_optimal_list_detectors_this_prob_dist(budget, NUM_HIDING_LOCATIONS, detectors, expected_value_each_node_this_prob_dist)
    
    # print("\nThis is optimal_list_detectors_this_prob_dist:")
    # pprint(optimal_list_detectors_this_prob_dist)
    
    detector_accuracies = sorted([accuracy for _, accuracy in optimal_list_detectors_this_prob_dist], reverse=True) + [0] * (NUM_HIDING_LOCATIONS - len(optimal_list_detectors_this_prob_dist))     # We sort the detector accuracies just in case (probably not needed because the way the MIP works but whatever), and then add any remaining null-detectors to ensure the number of detectors we have is equal to the number of hiding locations (makes the code easier later if we require that)
    # print(f"This is detector_accuracies: {detector_accuracies}")
 
    expected_value_detected_this_prob_dist = sum(detector_accuracies[i] * expected_value_each_node_this_prob_dist[i] for i in range(len(expected_value_each_node_this_prob_dist)))
    expected_total_value_this_prob_dist = sum(expected_value_each_node_this_prob_dist.values())
    
    return expected_value_detected_this_prob_dist, expected_total_value_this_prob_dist

# def calculate_expected_and_total_values_detected_this_prob_dist(prob_dist):
#     expected_value_detected_this_prob_dist = 0
#     expected_total_value_this_prob_dist = 0

#     for year, resource_set_dict in resource_sets.items():       
#         expected_value_items_each_node_this_prob_dist_specific_resource_set, _ = calculate_expected_value_under_equilibrium_each_node(resource_set_dict, item_vals, sizes_hiding_locations_each_year)

#         Will need to change this (and maybe can use the function calculate_expected_value_each_node_this_prob_dist I created). Maybe I can get rid of some lines below:
#         expected_value_detected_this_prob_dist_specific_resource_set = sum(detector_accuracies[i] * expected_value_items_each_node_this_prob_dist_specific_resource_set[i] for i in range(len(expected_value_items_each_node_this_prob_dist_specific_resource_set)))        
#         expected_total_value_this_prob_dist_specific_resource_set = sum(expected_value_items_each_node_this_prob_dist_specific_resource_set.values())
        
#         expected_value_detected_this_prob_dist += prob_dist[year] * expected_value_detected_this_prob_dist_specific_resource_set
#         expected_total_value_this_prob_dist += prob_dist[year] * expected_total_value_this_prob_dist_specific_resource_set

#     return expected_value_detected_this_prob_dist, expected_total_value_this_prob_dist

# def update_bin_dict(bins_data, entropy_range, expected_value_detected_this_prob_dist, expected_total_value_this_prob_dist):
#     """
#     For the bin containing 'entropy', append the given sample to its lists,
#     and if the bin has reached the required number of samples, compute the
#     final value and store it in the 'final_fraction_value_detected' key.
#     """
    
#     if "final_fraction_value_detected" not in bins_data[entropy_range]:
#         # Collect new sample
#         bins_data[entropy_range]["expected_values_detected"].append(expected_value_detected_this_prob_dist)
#         bins_data[entropy_range]["expected_total_values"].append(expected_total_value_this_prob_dist)
        
#         # If we've reached the required number of samples, compute the final value
#         if len(bins_data[entropy_range]["expected_values_detected"]) == NUM_SAMPLES_PER_BIN:
#             total_detected = sum(bins_data[entropy_range]["expected_values_detected"])
#             total_all = sum(bins_data[entropy_range]["expected_total_values"])
#             bins_data[entropy_range]["final_fraction_value_detected"] = total_detected / total_all
#             bins_data[entropy_range]["final_expected_value_detected"] = total_detected / NUM_SAMPLES_PER_BIN
            

#     return bins_data

def plot_entropy_vs_final_fraction_value_detected(bins_data):
    """
    Plot the (final) fraction detected as a function of entropy bins.
    """       
        
    x_labels = [k for k in bins_data]
    y_values = [bins_data[k]["final_fraction_value_detected"] for k in bins_data]

    data_range = max(y_values) - min(y_values)
    y_axis_buffer_adjust_amount = data_range * 0.05      # This will be used for the y ticks to ensure that there isn't a bar going to the top y-tick. Its just for presentation sake.

    y_min = min(y_values) - y_axis_buffer_adjust_amount
    y_max = max(y_values) + y_axis_buffer_adjust_amount
    
    num_ticks = 10
    y_ticks = np.linspace(y_min, y_max, num_ticks)

    def format_y_value(y, pos):
        return f'{y:.4f}'
    
    plt.figure(figsize=(15, 8)) 
    plt.title("Normalized Entropy vs. Fraction Detected", fontsize=14)
    plt.bar(x_labels, y_values)
    plt.xticks(rotation=45, fontsize=11)
    plt.xlabel("\nEntropy Bins", fontsize=14)
    plt.ylabel("Average Fraction Detected\n", fontsize=14)
    plt.yticks(y_ticks, [format_y_value(y, None) for y in y_ticks], fontsize=11)  # Using the formatter for the labels
    
    formatter = FuncFormatter(format_y_value)
    plt.gca().yaxis.set_major_formatter(formatter)    
    
    plt.ylim(y_min, y_max)
    plt.subplots_adjust(bottom=0.2, top=0.9)  # Adjust subplot margins to avoid cut-off    
    plt.savefig(f"entropy_plots/entropy_vs_fraction_detected_plots/entropy_vs_fraction_detected_budget_{budget}_and_{NUM_SAMPLES_PER_BIN}_samples_per_bin_and_NUM_BINS_{NUM_BINS}.png")
    # plt.show()

def plot_entropy_vs_final_expected_value_detected(bins_data):
    """
    Plot the (final) expected value detected as a function of entropy bins.
    """
        
    x_labels = [k for k in bins_data]
    y_values = [bins_data[k]["final_expected_value_detected"] for k in bins_data]
    
    data_range = max(y_values) - min(y_values)
    y_axis_buffer_adjust_amount = data_range * 0.05      # This will be used for the y ticks to ensure that there isn't a bar going to the top y-tick. Its just for presentation sake.

    y_min = min(y_values) - y_axis_buffer_adjust_amount
    y_max = max(y_values) + y_axis_buffer_adjust_amount

    # Calculate the y-ticks to include a tick at the min and max, and evenly spaced in between
    num_ticks = 10  # Define number of ticks you want (including min and max)
    y_ticks = np.linspace(y_min, y_max, num_ticks)
        
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
    plt.yticks(y_ticks, [format_y_value(y, None) for y in y_ticks], fontsize=11)  # Using the formatter for the labels 
    formatter = FuncFormatter(format_y_value)
    plt.gca().yaxis.set_major_formatter(formatter)      
    plt.ylim(y_min, y_max)
    plt.subplots_adjust(bottom=0.2, top=0.9)  # Adjust subplot margins to avoid cut-off    
    plt.savefig(f"entropy_plots/entropy_vs_value_detected_plots/entropy_vs_value_detected_budget_{budget}_and_{NUM_SAMPLES_PER_BIN}_samples_per_bin_and_NUM_BINS_{NUM_BINS}.png")
    # plt.show()

def main():
    
    start_time = time.time()
   
    bins_data = {}
    for i in range(NUM_BINS):
        bin_key_str = f"{i/NUM_BINS:.2f}-{(i+1)/NUM_BINS:.2f}"
        bins_data[bin_key_str] = {"expected_values_detected": [], "expected_total_values": []}

    num_scenarios_done = 0
    for entropy_range, entries in prob_distributions_dict.items():
        key_str = f"{entropy_range[0]:.2f}-{entropy_range[1]:.2f}"
        for entry in entries:
            prob_dist = entry["prob_dist"]
            expected_value_detected_this_prob_dist, expected_total_value_this_prob_dist = calculate_expected_and_total_values_detected_this_prob_dist(prob_dist, precomputed_equilib_vals, NUM_HIDING_LOCATIONS, detectors, budget)

            bins_data[key_str]["expected_values_detected"].append(expected_value_detected_this_prob_dist)
            bins_data[key_str]["expected_total_values"].append(expected_total_value_this_prob_dist)        
            
            num_scenarios_done += 1
            print(f"\rNumber of scenarios done: {num_scenarios_done}. Time elapsed: {time.time() - start_time} seconds", end="")
            
    for entropy_range, _ in prob_distributions_dict.items():        
        key_str = f"{entropy_range[0]:.2f}-{entropy_range[1]:.2f}"        
        total_detected = sum(bins_data[key_str]["expected_values_detected"])
        total_all = sum(bins_data[key_str]["expected_total_values"])
        bins_data[key_str]["final_fraction_value_detected"] = total_detected / total_all
        bins_data[key_str]["final_expected_value_detected"] = total_detected / NUM_SAMPLES_PER_BIN        
        
        
    return bins_data



#\ici 
# Get final results now. 
# Then, once done, start commenting things in code. Go through all comments and make sure everything is correct. Make sure to explain the logic of everything too, in case I need to look at my code in a few weeks. 
# Its important for my own efficiency later.

if __name__=="__main__":

    parser = argparse.ArgumentParser(description="Load configuration for entropy plot calculations.")
    parser.add_argument('-config', type=str, required=True, help='Path to the JSON configuration file.')
    parser.add_argument('-prob_distributions', type=str, required=True, help='Path to the Pickle file of probability distributions')
    args = parser.parse_args()
    item_vals, resource_sets, _, hiding_locations, NUM_HIDING_LOCATIONS, sizes_hiding_locations_each_year, detectors, budget, NUM_SAMPLES_PER_BIN, NUM_BINS = entropy_plot_input_variables.get_configuration(args.config)

    validate_data()
    precomputed_equilib_vals = compute_all_years_equilibrium_values_at_each_node(resource_sets, item_vals, sizes_hiding_locations_each_year) 

    with open(args.prob_distributions, 'rb') as file:
        prob_distributions_dict = pickle.load(file)    
    
    bins_at_end = main()
    
    resource_sets_info = compute_resource_sets_info_for_json()
    json_data = {"resource_sets_info": resource_sets_info}    
    json_data["bins_at_end"] = bins_at_end

    print("Creating json...")
    with open(f"output_data_entropy_plots/output_data_entropy_plot_budget_{budget}_and_{NUM_SAMPLES_PER_BIN}_samples_per_bin_and_NUM_BINS_{NUM_BINS}.json", "w") as f:
        json.dump(json_data, f, indent=4)    
    print("Finished creating json.")
    
    
    plot_entropy_vs_final_fraction_value_detected(bins_at_end)
    plot_entropy_vs_final_expected_value_detected(bins_at_end) 