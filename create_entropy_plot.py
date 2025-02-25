import json
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint, pformat
import time
import curses
import argparse
# from entropy_plot_input_variables import item_vals, resource_sets, num_resource_sets, sizes_hiding_locations, detector_accuracies, NUM_SAMPLES_NEEDED_PER_BIN, NUM_BINS
import entropy_plot_input_variables

def validate_data():
    """Ensure the input variables are consistent."""
    if len(sizes_hiding_locations) != len(detector_accuracies):
        raise ValueError("sizes_hiding_locations and detector_accuracies must have the same length")

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

def calculate_expected_value_under_equilibrium_each_node(backloading_stack_values, breakpoints):
    expected_values = {}
    for node in range(len(backloading_stack_values)):
        relevant_indices = [
            i for i in range(len(breakpoints) - 1)
            if breakpoints[i] <= node < breakpoints[i + 1]
        ]
        expected_values[node] = np.mean([
            backloading_stack_values[j]
            for j in range(breakpoints[relevant_indices[0]], breakpoints[relevant_indices[0] + 1])
        ])
        
    return expected_values

def compute_resource_sets_info_for_json():
    """
    Since resource_sets_info is deterministic and does not depend on
    the probability distribution, compute it exactly once here.
    """
    resource_sets_info = []
    for year, resource_set_dict in resource_sets.items():
        backloading_stack_values = calculate_backloading_stack_values(resource_set_dict, item_vals, sizes_hiding_locations)
        breakpoints = calculate_breakpoints(backloading_stack_values, sizes_hiding_locations)
        expected_value_items_each_node_this_prob_dist_specific_resource_set = calculate_expected_value_under_equilibrium_each_node(backloading_stack_values, breakpoints)
        resource_sets_info.append({
            "year": year,
            "breakpoints": breakpoints,
            "expected value of items at each node in equilibrium": expected_value_items_each_node_this_prob_dist_specific_resource_set
        })
    return resource_sets_info

def calculate_expected_and_total_values_detected_this_prob_dist_across_resource_sets(prob_dist):
    # We’ll collect each year’s breakpoints in a structure to save
    expected_value_detected_this_prob_dist = 0
    expected_total_value_this_prob_dist = 0

    for year, resource_set_dict in resource_sets.items():       
        backloading_stack_values = calculate_backloading_stack_values(resource_set_dict, item_vals, sizes_hiding_locations)    
        breakpoints = calculate_breakpoints(backloading_stack_values, sizes_hiding_locations)
        expected_value_items_each_node_this_prob_dist_specific_resource_set = calculate_expected_value_under_equilibrium_each_node(backloading_stack_values, breakpoints)        

        expected_value_detected_this_prob_dist_specific_resource_set = sum(detector_accuracies[i] * expected_value_items_each_node_this_prob_dist_specific_resource_set[i] for i in range(len(expected_value_items_each_node_this_prob_dist_specific_resource_set)))        
        expected_total_value_this_prob_dist_specific_resource_set = sum(expected_value_items_each_node_this_prob_dist_specific_resource_set.values())
        
        expected_value_detected_this_prob_dist += prob_dist[year] * expected_value_detected_this_prob_dist_specific_resource_set
        expected_total_value_this_prob_dist += prob_dist[year] * expected_total_value_this_prob_dist_specific_resource_set

    return expected_value_detected_this_prob_dist, expected_total_value_this_prob_dist

def update_bin_dict(bins_data, entropy, expected_value_detected_this_prob_dist, expected_total_value_this_prob_dist):
    """
    For the bin containing 'entropy', append the given sample to its lists,
    and if the bin has reached the required number of samples, compute the
    final value and store it in the 'final_value' key.
    """
    for (lower_bound, upper_bound), bin_data in bins_data.items():
        # Skip bins that do not include the current entropy
        if not (lower_bound <= entropy < upper_bound):
            continue
        
        # If 'final_value' is already computed, we do nothing for this bin
        if "final_value" in bin_data:
            break  # Because we found our bin, no need to continue iterating
        
        # Collect new sample
        bin_data["expected_values_detected"].append(expected_value_detected_this_prob_dist)
        bin_data["expected_total_values"].append(expected_total_value_this_prob_dist)
        
        # If we've reached the required number of samples, compute the final value
        if len(bin_data["expected_values_detected"]) == NUM_SAMPLES_NEEDED_PER_BIN:
            total_detected = sum(bin_data["expected_values_detected"])
            total_all = sum(bin_data["expected_total_values"])
            bin_data["final_value"] = total_detected / total_all
            
        break

    return bins_data

def plot_entropy_chart(bins_data):
    """
    Plot the (final) fraction detected as a function of entropy bins.
    """
        
    x_labels = [f"{k[0]:.2f}-{k[1]:.2f}" for k in bins_data]
    y_values = [bins_data[k]["final_value"] for k in bins_data]

    y_min = np.floor(min(y_values) * 100) / 100

    plt.bar(x_labels, y_values)
    plt.xticks(rotation=45)
    plt.xlabel("\nEntropy Bins", fontsize=14)
    plt.ylabel("Average Fraction Detected\n", fontsize=14)
    plt.title("Entropy vs. Fraction Detected")
    plt.ylim(y_min)
    plt.savefig("entropy_vs_fraction_detected.png")
    plt.show()

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
        while ("final_value" not in bins_data[(i/NUM_BINS, (i+1)/NUM_BINS)] and len(bins_data[(i/NUM_BINS, (i+1)/NUM_BINS)]["expected_values_detected"]) < NUM_SAMPLES_NEEDED_PER_BIN):
            prob_dist = generate_probability_distribution(power)
            entropy = calculate_normalized_entropy(prob_dist)
            expected_value_detected_this_prob_dist, expected_total_value_this_prob_dist = calculate_expected_and_total_values_detected_this_prob_dist_across_resource_sets(prob_dist)
            bins_data = update_bin_dict(bins_data, entropy, expected_value_detected_this_prob_dist, expected_total_value_this_prob_dist)
 
            # Log just prob_dist and entropy each iteration
            json_data.append({"prob_dist": prob_dist, "entropy": entropy})
            
            # The following is for printing progress to the terminal.
            stdscr.addstr(0, 0, "Bins data sample counts:\n{}".format(pformat({k: len(v["expected_values_detected"]) if "final_value" not in v else "All samples obtained." for k, v in bins_data.items()})))
            stdscr.addstr(NUM_BINS + 1, 0, "Power: {}".format(power))
            stdscr.addstr(NUM_BINS + 2, 0, "Iterations: {}".format(num_iterations))
            stdscr.addstr(NUM_BINS + 3, 0, "Time Elapsed: {:.2f} seconds".format(time.time() - start_time))
            stdscr.refresh()  # Refresh the screen            
            
            num_iterations += 1
            num_iterations_while_loop_at_current_power += 1

            all_higher_bins_done = all( "final_value" in bins_data[(j/NUM_BINS, (j+1)/NUM_BINS)] for j in range(i, NUM_BINS) )            
            if all_higher_bins_done or (num_iterations_while_loop_at_current_power > max_num_while_loop_iterations_before_increasing_power):     # If the higher bins have been filled or we have done more than the maximum number of while loop iterations allowed
                power *= power_step_factor
                num_iterations_while_loop_at_current_power = 0

    return bins_data


if __name__=="__main__":

    parser = argparse.ArgumentParser(description="Load configuration for entropy plot calculations.")
    parser.add_argument('-config', type=str, required=True, help='Path to the JSON configuration file.')
    args = parser.parse_args()
    item_vals, resource_sets, num_resource_sets, hiding_locations, fraction_cargo_containers_storing_drugs, sizes_hiding_locations, detector_accuracies, NUM_SAMPLES_NEEDED_PER_BIN, NUM_BINS = entropy_plot_input_variables.get_configuration(args.config)

    # Store all runs here before/while writing them out
    json_data = []
    bins_at_end = curses.wrapper(main)

    print("Creating json...")
    with open("breaking_indices_log.json", "w") as f:
        json.dump(json_data, f, indent=4)    
    print("Finished creating json.")
    
    plot_entropy_chart(bins_at_end)
