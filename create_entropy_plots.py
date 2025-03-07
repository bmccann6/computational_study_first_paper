"""
Overview: 
    This module creates the entropy plots we want (expected fraction detected vs entropy, and also average value detected vs entropy).  

Inputs:
    The user must pass in two files:
        1. Configuration json file
            -These files give data on the value of each hider's items, their possible resource sets, the hiding locations and their full original capacities, 
            the detectors (including their accuracies and costs), the budget, the fraction of containers that come from drug regions of the world, 
            the fraction of those containers which can then be accessed by the drug cartel, and the fraction of cargo containers which are in general stopped for inspection.   
                
        2. Probability distributions generated pickle file.
            -These files contain probability distributions that we generated. These files have a certain number of entropy bins, and the number of probability distribution samples generated for each bin.
            By generating these apriori, we save time by not having to generate them here everytime we want to make a plot.

Description:
    Once we have the inputs, we precompute the equilibrium values at each node for each resource set (using the function compute_each_resource_sets_equilibrium_values_at_each_node). 
    This saves time because these values will not change for a given probability distribution, so we do not want to calculate these values every single time we are calculating the 
    expected fraction of value detected and averge value detected for a probability distribution.

    The code then runs generate_entropy_plot_data(), which calculates the expected fraction of value detected and the average expected value detected for each probability distribution in our probability distributions generated pickle file.


Outputs:
    -The code creates the output json, and the plots. The output json has a tag for information about the resource sets (including the name of the resource set, the breakpoints, and expected value of 
     items at each node under equilibrium for that resource set). There is also a tag for each entropy bin which contains lists for the expected values detected and expected total values detected for each 
     of the probability distributions in our probability distributions generated pickle file. Finally, we also have tags for the expected fraction of value detected across all probability distributions, and the 
     average expected value detected across all probability distributions.

    -Entropy vs expected fraction detected plot.
    
    -Entropy vs average value detected plot.

"""


import json
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint, pformat
import time
import argparse
from matplotlib.ticker import FuncFormatter
import datetime
from gurobipy import Model, GRB, quicksum
import setup_data


def validate_data():
    """Ensures the input variables are consistent."""
    for resource_set_name, drugs in resource_sets.items():
        for drug in drugs:
            if drug not in item_vals:
                raise ValueError(f"Drug '{drug}' in resource_set_name '{resource_set_name}' is not a valid item in item_vals")

def calculate_backloading_stack_values(resource_set, item_vals, capacities):
    """
    This calculates the expected value of items at each node when we do the initial backloading. 
    These values are important to calculate the breakpoints. 
    These values are the same as the A_i values at the end of the document mathieu wrote in his iPad years ago.
    NOTE! They are not the expected value of items at each node i.
    
    The core of this function is a while-loop. In this while-loop, we iteratively fill up the locations via backloading. Every time we allocate resources to a location, we remove 
    those resources from hider_resources_sorted and decrease the capacity at the location we assigned the resources too.
    Then in the next iteration, if that location has its capacity filled, we remove that location then continue on to the next iteration.
    Also, if a resource type is depleted, we remove that resource type then continue on to the next iteration.
    We repeat until we have completed the backloading.
    """
    hider_resources_sorted = sorted(resource_set.items(), key=lambda x: item_vals[x[0]])
    local_copy_capacities = capacities.copy()          # We make a local copy because we will be modifying the sizes of the hiding locations as we fill them up. But we don't want to globally change the capacities list object we passed in because we need that list in later calls.
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
        hider_resources_sorted[0] = (hider_resources_sorted[0][0], hider_resources_sorted[0][1] - assignment_amount)
        local_copy_capacities[0] -= assignment_amount
        
    return backloading_stack_values

def calculate_breakpoints(backloading_stack_values, capacities):
    """   
    This function uses the backloaded stack values (by stack values I mean the total value at each node as a result of backloading the hider items), and iteratively calculates the breakpoints.
    Recall that breakpoints are based on if we were to average the value of the stacks for the nodes between the last breakpoint i_{\ell} and the 
    next candidate breakpoint node i_{\ell+1} (i.e. randomize all the items positioned nodes between node i_{\ell} + 1 and node i_{\ell+1} so that each node gets the same expected value at it).
    If that average value is greater than any average values of stacks from i_{\ell} + 1 up to any node past this last breakpoint, then we add i_{\ell+1} to our list of breakpoints.
    Also, we also always add n (the number of locations) to be the final breakpoint. Note that this is 1 more than the last index of the capacities (since python counts from 0).
    
    The mathematical algorithm for this is at the end of the document mathieu wrote in his iPad years ago. Note that the A_i values in mathieu's document are the stack values from pure backloading.
    """
    
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

def calculate_expected_value_under_equilibrium_each_node(resource_set, item_vals, capacities):  
    """
    This function simply uses the backloaded stack values and the breakpoints. 
    For each pair of consecutive breakpoints, it calculates the average of the backloaded stack values for nodes between that pair.
    This average is the expected value of items that will be at each node between that pair in equilibrium.
    """
    backloading_stack_values = calculate_backloading_stack_values(resource_set, item_vals, capacities)
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

def compute_each_resource_sets_equilibrium_values_at_each_node(resource_sets, item_vals, sizes_hiding_locations):
    """
    Computes the equilibrium values for each node for each resource set.
    """
    resource_set_to_equilibrium_node_values = {}
    for resource_set_name, resource_set in resource_sets.items():
        eq_vals, _ = calculate_expected_value_under_equilibrium_each_node(resource_set, item_vals, sizes_hiding_locations)
        resource_set_to_equilibrium_node_values[resource_set_name] = eq_vals
        
    return resource_set_to_equilibrium_node_values

def compute_resource_sets_info_for_json():
    """
    Since the information regarding the resource sets is deterministic and does not depend on the probability distribution, we compute it exactly once here and add it to the json.
    """
    resource_sets_info = []
    for resource_set_name, resource_set in resource_sets.items():
        expected_value_items_each_node_this_prob_dist_specific_resource_set, breakpoints = calculate_expected_value_under_equilibrium_each_node(resource_set, item_vals, sizes_hiding_locations)
        resource_sets_info.append({
            "resource_set_name": resource_set_name,
            "breakpoints": breakpoints,
            "expected value of items at each node in equilibrium": expected_value_items_each_node_this_prob_dist_specific_resource_set
        })
    return resource_sets_info

def calculate_expected_value_each_node_this_prob_dist(prob_dist, precomputed_equilib_vals, NUM_HIDING_LOCATIONS):
    """
    This function calculates the overall expected value at each node under a prob_dist belief of what the hider's resources are. 
    To calculate the "overall expected value at each node", for each resource set and node we simply multiply the equilibrium stack value at that node under 
    that resource set, by the probability that resource set occurs. We then sum up these values.
    This is needed for the budget problem we ultimately solve. These values are the E_{\omega}[A^{\omega}_i] values in the budget MIP given on slide 15 of my "Informs 2024 Presentation".
    """
    expected_value_each_node_under_this_prob_dist = {i: 0 for i in range(NUM_HIDING_LOCATIONS)}
    for resource_set_name, eq_vals in precomputed_equilib_vals.items():
        for node in range(NUM_HIDING_LOCATIONS):
            expected_value_each_node_under_this_prob_dist[node] += prob_dist[resource_set_name] * eq_vals[node]
            
    return expected_value_each_node_under_this_prob_dist

def get_optimal_list_detectors_this_prob_dist(budget, NUM_HIDING_LOCATIONS, detectors, expected_value_each_node_under_this_prob_dist):
    """
    This function solves the budget problem for a given probability distribution over the resource sets.
    """
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
            
    # Create list of tuples with (detector_type_name, accuracy) each time that detector type was purchased to be positioned at a node.
    optimal_list_detectors_this_prob_dist = [(detector_type, detectors[detector_type]['accuracy'])
                                                for detector_type in detectors for node in nodes if x[detector_type, node].x > 0 ]
        
    return optimal_list_detectors_this_prob_dist
    
def calculate_expected_and_total_values_detected_this_prob_dist(prob_dist, precomputed_equilib_vals, NUM_HIDING_LOCATIONS, detectors, budget):
    """
    This function is the core of this module. 
    It first calculates the expected value of items at each node under a given prob_dist.
    Then it gets the optimal list of detectors for this prob_dist. 
    Then it sorts the accuracies of that optimal list of detectors (and adds some null detectors to make some later calculations easier).
    Then it calculates the expected value DETECTED under this prob_dist.
    Then it also calculates the expected total value of items under this prob_dist.
    It then returns both the expected value detected and expected total value of items under this prob_dist.
    These values are crucial to make the plots.
    """
    expected_value_each_node_this_prob_dist = calculate_expected_value_each_node_this_prob_dist(prob_dist, precomputed_equilib_vals, NUM_HIDING_LOCATIONS)
    optimal_list_detectors_this_prob_dist = get_optimal_list_detectors_this_prob_dist(budget, NUM_HIDING_LOCATIONS, detectors, expected_value_each_node_this_prob_dist)
        
    detector_accuracies = sorted([accuracy for _, accuracy in optimal_list_detectors_this_prob_dist], reverse=True) + [0] * (NUM_HIDING_LOCATIONS - len(optimal_list_detectors_this_prob_dist))     # We sort the detector accuracies just in case (probably not needed because the way the MIP works but whatever), and then add any remaining null-detectors to ensure the number of detectors we have is equal to the number of hiding locations (makes the code easier later if we require that)
 
    expected_value_detected_this_prob_dist = sum(detector_accuracies[i] * expected_value_each_node_this_prob_dist[i] for i in range(len(expected_value_each_node_this_prob_dist)))
    expected_total_value_this_prob_dist = sum(expected_value_each_node_this_prob_dist.values())
    
    return expected_value_detected_this_prob_dist, expected_total_value_this_prob_dist

def plot_entropy_vs_final_fraction_value_detected(bins_data, path_to_save_fig):
    """
    For each entropy bin, we plot the expected fraction of value detected over all probability distributions generated whose entropy falls within this entropy bin.
    """       
        
    x_labels = [k for k in bins_data]
    y_values = [bins_data[k]["expected_fraction_value_detected_across_all_prob_distributions_this_bin"] for k in bins_data]

    data_range = max(y_values) - min(y_values)
    y_axis_buffer_adjust_amount = data_range * 0.05      # This will be used for the y-ticks to ensure that there isn't a bar going to the top y-tick. Its just for presentation sake.

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
    plt.yticks(y_ticks, [format_y_value(y, None) for y in y_ticks], fontsize=11)  
    
    formatter = FuncFormatter(format_y_value)
    plt.gca().yaxis.set_major_formatter(formatter)    
    
    plt.ylim(y_min, y_max)
    plt.subplots_adjust(bottom=0.2, top=0.9)  # Adjust subplot margins to avoid cut-off    
    plt.savefig(path_to_save_fig)
    # plt.show()

def plot_entropy_vs_final_expected_value_detected(bins_data, path_to_save_fig):
    """
    For each entropy bin, we plot the average expected value detected over all probability distributions generated whose entropy falls within this entropy bin.
    """
        
    x_labels = [k for k in bins_data]
    y_values = [bins_data[k]["average_expected_value_detected_across_all_prob_distributions_this_bin"] for k in bins_data]
    
    data_range = max(y_values) - min(y_values)
    y_axis_buffer_adjust_amount = data_range * 0.05      # This will be used for the y-ticks to ensure that there isn't a bar going to the top y-tick. Its just for presentation sake.

    y_min = min(y_values) - y_axis_buffer_adjust_amount
    y_max = max(y_values) + y_axis_buffer_adjust_amount

    num_ticks = 10  
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
    plt.title("Normalized Entropy vs. Average Expected Value Detected", fontsize=14)
    plt.bar(x_labels, y_values)
    plt.xticks(rotation=45, fontsize=11)
    plt.xlabel("\nEntropy Bins", fontsize=14)
    plt.ylabel("USD\n", fontsize=14)
    plt.yticks(y_ticks, [format_y_value(y, None) for y in y_ticks], fontsize=11)  
    formatter = FuncFormatter(format_y_value)
    plt.gca().yaxis.set_major_formatter(formatter)      
    plt.ylim(y_min, y_max)
    plt.subplots_adjust(bottom=0.2, top=0.9)  # Adjust subplot margins to avoid cut-off    
    plt.savefig(path_to_save_fig)
    # plt.show()

def create_output_json(bins_at_end):
    """
    This creates a json file which contains information regarding the resource sets, the expected values detected and expected total values for each probability distribution, the 
    expected fraction of value detected across all probability distributions for a given entropy bin, and the average expected value detected across all probability distributions for a given entropy bin.
    """
    resource_sets_info = compute_resource_sets_info_for_json()
    json_data = {"resource_sets_info": resource_sets_info}    
    json_data["bins_at_end"] = bins_at_end

    print("Creating json...")
    with open(f"output_data_entropy_plots/output_data_entropy_plot_budget_{budget}_and_{NUM_SAMPLES_PER_BIN}_samples_per_bin_and_NUM_BINS_{NUM_BINS}_timestamp_{time_run_starts}.json", "w") as f:
        json.dump(json_data, f, indent=4)    
    print("Finished creating json.")
        
def generate_entropy_plot_data():
    """
    This function serves as the main driver of the module. It operates in two primary stages using for-loops:
    1. It constructs bins for each entropy range and calculates for each bin the expected value detected and the expected total values for each probability distribution within these ranges.
    2. Using the results of the above stage, it computes, for each bin, the expected fraction of value detected and the average expected value detected ACROSS ALL probability distributions in that bin.
    """
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
        bins_data[key_str]["expected_fraction_value_detected_across_all_prob_distributions_this_bin"] = total_detected / total_all
        bins_data[key_str]["average_expected_value_detected_across_all_prob_distributions_this_bin"] = total_detected / NUM_SAMPLES_PER_BIN        
        
    return bins_data


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Load configuration for entropy plot calculations.")
    parser.add_argument('-config_path', type=str, required=True, help='Path to the JSON configuration file.')
    parser.add_argument('-prob_distributions_path', type=str, required=True, help='Path to the Pickle file of probability distributions')
    args = parser.parse_args()
    item_vals, resource_sets, _, hiding_locations, NUM_HIDING_LOCATIONS, sizes_hiding_locations, detectors, budget = setup_data.get_configuration(args.config_path)
    prob_distributions_dict, NUM_SAMPLES_PER_BIN, NUM_BINS = setup_data.get_prob_distributions_dict(args.prob_distributions_path)
    time_run_starts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")     # This is useful for naming our output files. We want all the output files to have a time-stamp to make it easier if we need to match them up later.
    
    validate_data()
    precomputed_equilib_vals = compute_each_resource_sets_equilibrium_values_at_each_node(resource_sets, item_vals, sizes_hiding_locations) 
    bins_at_end = generate_entropy_plot_data()
    
    create_output_json(bins_at_end)
    plot_entropy_vs_final_fraction_value_detected(bins_at_end, path_to_save_fig=f"entropy_plots/entropy_vs_fraction_detected_plots/entropy_vs_fraction_detected_budget_{budget}_and_{NUM_SAMPLES_PER_BIN}_samples_per_bin_and_NUM_BINS_{NUM_BINS}_timestamp_{time_run_starts}.png")
    plot_entropy_vs_final_expected_value_detected(bins_at_end, path_to_save_fig=f"entropy_plots/entropy_vs_value_detected_plots/entropy_vs_value_detected_budget_{budget}_and_{NUM_SAMPLES_PER_BIN}_samples_per_bin_and_NUM_BINS_{NUM_BINS}_timestamp_{time_run_starts}.png") 