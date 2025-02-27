import json
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint, pformat
import time
import curses
import argparse
from matplotlib.ticker import FuncFormatter
from gurobipy import Model, GRB, quicksum
# from entropy_plot_input_variables import item_vals, resource_sets, num_resource_sets, sizes_hiding_locations, detector_accuracies, NUM_SAMPLES_NEEDED_PER_BIN, NUM_BINS
import entropy_plot_input_variables


def validate_data():
    """Ensure the input variables are consistent."""
    for year, drugs in resource_sets.items():
        for drug in drugs:
            if drug not in item_vals:
                raise ValueError(f"Drug '{drug}' in year '{year}' is not a valid item in item_vals")

def generate_probability_distribution(power):
    random_numbers = np.power(np.random.random(num_resource_sets), power)
    probability_values = random_numbers / random_numbers.sum()
    return {year: prob for year, prob in zip(resource_sets.keys(), probability_values)}

def calculate_normalized_entropy(prob_dist):
    probs = np.array(list(prob_dist.values()))
    nonzero_probs = probs[probs > 0]        # Only take log of non-zero entries just in case we get division by 0 errors.
    entropy = -np.sum(nonzero_probs * np.log2(nonzero_probs)) / np.log2(num_resource_sets)
    return entropy

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

def compute_resource_sets_info_for_json():
    """
    Since resource_sets_info is deterministic and does not depend on
    the probability distribution, compute it exactly once here.
    """
    resource_sets_info = []
    for year, resource_set_dict in resource_sets.items():
        expected_value_items_each_node_this_prob_dist_specific_resource_set, breakpoints = calculate_expected_value_under_equilibrium_each_node(resource_set_dict, item_vals, sizes_hiding_locations)
        resource_sets_info.append({
            "year": year,
            "breakpoints": breakpoints,
            "expected value of items at each node in equilibrium": expected_value_items_each_node_this_prob_dist_specific_resource_set
        })
    return resource_sets_info

def calculate_expected_value_each_node_this_prob_dist(prob_dist, resource_sets, hiding_locations):
    expected_value_each_node_under_this_prob_dist = {i: 0 for i in range(len(hiding_locations))}
    for year, resource_set_dict in resource_sets.items():
        expected_value_under_equilibrium_each_node, _ = calculate_expected_value_under_equilibrium_each_node(resource_set_dict, item_vals, sizes_hiding_locations)
        for node in range(len(hiding_locations)):
            expected_value_each_node_under_this_prob_dist[node] += prob_dist[year] * expected_value_under_equilibrium_each_node[node]
            
    return expected_value_each_node_under_this_prob_dist

def get_optimal_list_detectors_this_prob_dist(budget, NUM_HIDING_LOCATIONS, detectors, expected_value_each_node_under_this_prob_dist):
    # expected_value_each_node_under_this_prob_dist = calculate_expected_value_each_node_this_prob_dist(prob_dist, resource_sets, hiding_locations)

    nodes = range(NUM_HIDING_LOCATIONS)

    m = Model("integer_program")
    x = m.addVars(detectors, nodes, vtype=GRB.BINARY, name="x")  # Binary decision variables
    objective = quicksum(expected_value_each_node_under_this_prob_dist[node] * detectors[detector_type]["accuracy"] * x[detector_type, node] for detector_type in detectors for node in nodes)
    m.setObjective(objective, GRB.MAXIMIZE)

    # Budget constraint
    m.addConstr(quicksum(detectors[detector_type]["cost"] * x[detector_type, node] for detector_type in detectors for node in nodes) <= budget, "budget_constraint")

    # Each item can be selected at most once across all t
    for node in nodes:
        m.addConstr(quicksum(x[detector_type, node] for detector_type in detectors) <= 1, f"selection_constraint_{node}")

    m.optimize()
            
    # Create list of tuples with (detector_type_name, accuracy, x_value)
    optimal_list_detectors_this_prob_dist = [(detector_type, detectors[detector_type]['accuracy'])
                                                for detector_type in detectors for node in nodes if x[detector_type, node].x > 0 ]
    
    
    
    return optimal_list_detectors_this_prob_dist
    
def calculate_expected_and_total_values_detected_this_prob_dist_across_resource_sets(prob_dist, resource_sets, hiding_locations, detectors, budget):
    
    expected_value_each_node_this_prob_dist = calculate_expected_value_each_node_this_prob_dist(prob_dist, resource_sets, hiding_locations)
    optimal_list_detectors_this_prob_dist = get_optimal_list_detectors_this_prob_dist(budget, NUM_HIDING_LOCATIONS, detectors, expected_value_each_node_this_prob_dist)
    
    print("\nThis is optimal_list_detectors_this_prob_dist:")
    pprint(optimal_list_detectors_this_prob_dist)
    
    detector_accuracies = sorted([accuracy for _, accuracy in optimal_list_detectors_this_prob_dist], reverse=True) + [0] * (NUM_HIDING_LOCATIONS - len(optimal_list_detectors_this_prob_dist))     # We sort the detector accuracies just in case (probably not needed because the way the MIP works but whatever), and then add any remaining null-detectors to ensure the number of detectors we have is equal to the number of hiding locations (makes the code easier later if we require that)
    print(f"This is detector_accuracies: {detector_accuracies}")
 
    expected_value_detected_this_prob_dist = sum(detector_accuracies[i] * expected_value_each_node_this_prob_dist[i] for i in range(len(expected_value_each_node_this_prob_dist)))
    expected_total_value_this_prob_dist = sum(expected_value_each_node_this_prob_dist.values())
    
    return expected_value_detected_this_prob_dist, expected_total_value_this_prob_dist

# def calculate_expected_and_total_values_detected_this_prob_dist_across_resource_sets(prob_dist):
#     expected_value_detected_this_prob_dist = 0
#     expected_total_value_this_prob_dist = 0

#     for year, resource_set_dict in resource_sets.items():       
#         expected_value_items_each_node_this_prob_dist_specific_resource_set, _ = calculate_expected_value_under_equilibrium_each_node(resource_set_dict, item_vals, sizes_hiding_locations)

#         Will need to change this (and maybe can use the function calculate_expected_value_each_node_this_prob_dist I created). Maybe I can get rid of some lines below:
#         expected_value_detected_this_prob_dist_specific_resource_set = sum(detector_accuracies[i] * expected_value_items_each_node_this_prob_dist_specific_resource_set[i] for i in range(len(expected_value_items_each_node_this_prob_dist_specific_resource_set)))        
#         expected_total_value_this_prob_dist_specific_resource_set = sum(expected_value_items_each_node_this_prob_dist_specific_resource_set.values())
        
#         expected_value_detected_this_prob_dist += prob_dist[year] * expected_value_detected_this_prob_dist_specific_resource_set
#         expected_total_value_this_prob_dist += prob_dist[year] * expected_total_value_this_prob_dist_specific_resource_set

#     return expected_value_detected_this_prob_dist, expected_total_value_this_prob_dist

def update_bin_dict(bins_data, entropy, expected_value_detected_this_prob_dist, expected_total_value_this_prob_dist):
    """
    For the bin containing 'entropy', append the given sample to its lists,
    and if the bin has reached the required number of samples, compute the
    final value and store it in the 'final_fraction_value_detected' key.
    """
    for (lower_bound, upper_bound), bin_data in bins_data.items():
        # Skip bins that do not include the current entropy
        if not (lower_bound <= entropy < upper_bound):
            continue
        
        # If 'final_fraction_value_detected' is already computed, we do nothing for this bin
        if "final_fraction_value_detected" in bin_data:
            break  # Because we found our bin, no need to continue iterating
        
        # Collect new sample
        bin_data["expected_values_detected"].append(expected_value_detected_this_prob_dist)
        bin_data["expected_total_values"].append(expected_total_value_this_prob_dist)
        
        # If we've reached the required number of samples, compute the final value
        if len(bin_data["expected_values_detected"]) == NUM_SAMPLES_NEEDED_PER_BIN:
            total_detected = sum(bin_data["expected_values_detected"])
            total_all = sum(bin_data["expected_total_values"])
            bin_data["final_fraction_value_detected"] = total_detected / total_all
            bin_data["final_expected_value_detected"] = total_detected / NUM_SAMPLES_NEEDED_PER_BIN
            
        break

    return bins_data

def plot_entropy_vs_final_fraction_value_detected(bins_data):
    """
    Plot the (final) fraction detected as a function of entropy bins.
    """
        
    x_labels = [f"{k[0]:.2f}-{k[1]:.2f}" for k in bins_data]
    y_values = [bins_data[k]["final_fraction_value_detected"] for k in bins_data]

    y_min = np.floor(min(y_values) * 100) / 100

    plt.figure(figsize=(15, 8)) 
    plt.title("Normalized Entropy vs. Fraction Detected", fontsize=14)
    plt.bar(x_labels, y_values)
    plt.xticks(rotation=45, fontsize=11)
    plt.xlabel("\nEntropy Bins", fontsize=14)
    plt.ylabel("Average Fraction Detected\n", fontsize=14)
    plt.yticks(fontsize=11)
    plt.ylim(bottom=y_min)
    plt.subplots_adjust(bottom=0.2, top=0.9)  # Adjust subplot margins to avoid cut-off    
    plt.savefig("entropy_plots/fraction_detected_plots/entropy_vs_fraction_detected.png")
    # plt.show()

def plot_entropy_vs_final_expected_value_detected(bins_data):
    """
    Plot the (final) expected value detected as a function of entropy bins.
    """
        
    x_labels = [f"{k[0]:.2f}-{k[1]:.2f}" for k in bins_data]
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
    plt.savefig("entropy_plots/value_detected_plots/entropy_vs_value_detected.png")
    # plt.show()

def main(stdscr):
    # Clear screen
    stdscr.clear()
    
    start_time = time.time()
    validate_data()
    power = 1
    power_step_factor = 1.5        # From experimenting, I found that this value makes the code run decently fast
    bins_data = {(i/NUM_BINS, (i+1)/NUM_BINS): {"expected_values_detected": [], "expected_total_values": []} for i in range(NUM_BINS)}
    max_num_while_loop_iterations_before_increasing_power = NUM_SAMPLES_NEEDED_PER_BIN * NUM_BINS
    num_iterations_while_loop_at_current_power = 0
    num_iterations = 0
 
    # We only store resource_sets_info once at the beginning
    global json_data
    resource_sets_info = compute_resource_sets_info_for_json()
    json_data = [
        {"resource_sets_info": resource_sets_info}
    ] 
 
        
    for i in range(NUM_BINS-1, -1, -1):
        while ("final_fraction_value_detected" not in bins_data[(i/NUM_BINS, (i+1)/NUM_BINS)] and len(bins_data[(i/NUM_BINS, (i+1)/NUM_BINS)]["expected_values_detected"]) < NUM_SAMPLES_NEEDED_PER_BIN):
            prob_dist = generate_probability_distribution(power)
            entropy = calculate_normalized_entropy(prob_dist)
            expected_value_detected_this_prob_dist, expected_total_value_this_prob_dist = calculate_expected_and_total_values_detected_this_prob_dist_across_resource_sets(prob_dist, resource_sets, hiding_locations, detectors, budget)
            bins_data = update_bin_dict(bins_data, entropy, expected_value_detected_this_prob_dist, expected_total_value_this_prob_dist)
 
            # Log just prob_dist and entropy each iteration
            json_data.append({"prob_dist": prob_dist, "entropy": entropy})
            
            # The following is for printing progress to the terminal.
            stdscr.addstr(0, 0, "Bins data sample counts:\n{}".format(pformat({k: len(v["expected_values_detected"]) if "final_fraction_value_detected" not in v else "All samples obtained." for k, v in bins_data.items()})))
            stdscr.addstr(NUM_BINS + 1, 0, "Power: {}".format(power))
            stdscr.addstr(NUM_BINS + 2, 0, "Iterations: {}".format(num_iterations))
            stdscr.addstr(NUM_BINS + 3, 0, "Time Elapsed: {:.2f} seconds".format(time.time() - start_time))
            stdscr.refresh()  # Refresh the screen            
            
            num_iterations += 1
            num_iterations_while_loop_at_current_power += 1

            all_higher_bins_done = all( "final_fraction_value_detected" in bins_data[(j/NUM_BINS, (j+1)/NUM_BINS)] for j in range(i, NUM_BINS) )            
            if all_higher_bins_done or (num_iterations_while_loop_at_current_power > max_num_while_loop_iterations_before_increasing_power):     # If the higher bins have been filled or we have done more than the maximum number of while loop iterations allowed
                power *= power_step_factor
                num_iterations_while_loop_at_current_power = 0

    return bins_data


if __name__=="__main__":

    parser = argparse.ArgumentParser(description="Load configuration for entropy plot calculations.")
    parser.add_argument('-config', type=str, required=True, help='Path to the JSON configuration file.')
    args = parser.parse_args()
    item_vals, resource_sets, num_resource_sets, hiding_locations, NUM_HIDING_LOCATIONS, fraction_cargo_containers_storing_drugs, sizes_hiding_locations, detectors, budget, NUM_SAMPLES_NEEDED_PER_BIN, NUM_BINS = entropy_plot_input_variables.get_configuration(args.config)

    # Store all runs here before/while writing them out
    json_data = []
    bins_at_end = curses.wrapper(main)

    print("Creating json...")
    with open("breaking_indices_jsons/breaking_indices_log.json", "w") as f:
        json.dump(json_data, f, indent=4)    
    print("Finished creating json.")
    
    plot_entropy_vs_final_fraction_value_detected(bins_at_end)
    plot_entropy_vs_final_expected_value_detected(bins_at_end) 