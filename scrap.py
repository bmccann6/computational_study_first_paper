import json
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint, pformat
import time
import curses
from entropy_plot_input_variables import item_vals, resource_sets, num_resource_sets, sizes_hiding_locations, detector_accuracies, NUM_SAMPLES_NEEDED_PER_BIN


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
    entropy = -np.sum(probs * np.log2(probs)) / np.log2(num_resource_sets)
    return entropy

# NOTE: These are the A_i values in the algorithm mathieu wrote. NOTE! They are not the expected value of items at each node i.
def calculate_backloading_stack_values(resource_set_dict):
    hider_resources_sorted = sorted(resource_set_dict.items(), key=lambda x: item_vals[x[0]])
    local_copy_sizes_hiding_locations = sizes_hiding_locations.copy()          # We make a local copy because we will be modifying the sizes of the hiding locations as we fill them up. But we don't want to change sizes_hiding_locations because we need that in later calls.
    backloading_stack_values = {i: 0 for i in range(len(sizes_hiding_locations))}
    i = 0
    
    while local_copy_sizes_hiding_locations and hider_resources_sorted:
        if local_copy_sizes_hiding_locations[0] == 0:
            local_copy_sizes_hiding_locations.pop(0)
            i += 1
            continue
        
        if hider_resources_sorted[0][1] == 0:
            hider_resources_sorted.pop(0)
            continue
        
        assignment_amount = min(local_copy_sizes_hiding_locations[0], hider_resources_sorted[0][1])
        item_name = hider_resources_sorted[0][0]
        item_value = item_vals[item_name]
        backloading_stack_values[i] += assignment_amount * item_value
        local_copy_sizes_hiding_locations[0] -= assignment_amount
        hider_resources_sorted[0] = (hider_resources_sorted[0][0], hider_resources_sorted[0][1] - assignment_amount)
        
    return backloading_stack_values

def calculate_breakpoints(backloading_stack_values):
    n = len(sizes_hiding_locations)
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


def calculate_expected_and_total_values_detected_this_prob_dist_across_resource_sets(prob_dist):
    # We’ll collect each year’s breakpoints in a structure to save
    resource_sets_info = []
    expected_value_detected_this_prob_dist = 0
    expected_total_value_this_prob_dist = 0

    for year, resource_set_dict in resource_sets.items():       
        backloading_stack_values = calculate_backloading_stack_values(resource_set_dict)    
        breakpoints = calculate_breakpoints(backloading_stack_values)
        expected_value_items_each_node = calculate_expected_value_under_equilibrium_each_node(backloading_stack_values, breakpoints)        
        
        expected_value_detected_this_prob_dist_specific_resource_set = sum(detector_accuracies[i] * expected_value_items_each_node[i] for i in range(len(expected_value_items_each_node)))        
        expected_total_value_this_prob_dist_specific_resource_set = sum(expected_value_items_each_node[i] for i in range(len(expected_value_items_each_node)))
        resource_sets_info.append({
            "year": year,
            "breakpoints": breakpoints
        })
        expected_value_detected_this_prob_dist += prob_dist[year] * expected_value_detected_this_prob_dist_specific_resource_set
        expected_total_value_this_prob_dist += prob_dist[year] * expected_total_value_this_prob_dist_specific_resource_set

    # Log to JSON each time we compute for these prob_dist
    log_entry = {
        "prob_dist": prob_dist,
        "resource_sets": resource_sets_info
    }
    all_calculations.append(log_entry)
    
    # Write out after each iteration so you can see updates in real time
    with open("breaking_indices_log.json", "w") as f:
        json.dump(all_calculations, f, indent=4)

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
    Plot the (final) fraction undetected as a function of entropy bins.
    """
    x_labels = [f"{k[0]:.1f}-{k[1]:.1f}" for k in bins_data]
    y_values = [bins_data[k]["final_value"] for k in bins_data]

    y_min = np.floor(min(y_values) * 100) / 100

    plt.bar(x_labels, y_values)
    plt.xlabel("\nEntropy Bins", fontsize=14)
    plt.ylabel("Average Fraction Undetected\n", fontsize=14)
    plt.title("Entropy vs. Fraction Undetected")
    plt.ylim(y_min)
    plt.savefig("entropy_vs_fraction_undetected.png")
    plt.show()

def main(stdscr):
    # Clear screen
    stdscr.clear()
    
    start_time = time.time()
    validate_data()
    power = 1
    power_step_factor = 1.75        # From experimenting, I found that this value makes the code run decently fast
    bins_data = {(i/10, (i+1)/10): {"expected_values_detected": [], "expected_total_values": []} for i in range(10)}
    max_num_while_loop_iterations_before_increasing_power = NUM_SAMPLES_NEEDED_PER_BIN * 10
    num_iterations_while_loop_at_current_power = 0
    num_iterations = 0
    
    for i in range(9, -1, -1):
        while ("final_value" not in bins_data[(i/10, (i+1)/10)] and len(bins_data[(i/10, (i+1)/10)]["expected_values_detected"]) < NUM_SAMPLES_NEEDED_PER_BIN):
            prob_dist = generate_probability_distribution(power)
            entropy = calculate_normalized_entropy(prob_dist)
            expected_value_detected_this_prob_dist, expected_total_value_this_prob_dist = calculate_expected_and_total_values_detected_this_prob_dist_across_resource_sets(prob_dist)
            bins_data = update_bin_dict(bins_data, entropy, expected_value_detected_this_prob_dist, expected_total_value_this_prob_dist)
            
            # The following is for printing progress to the terminal.
            stdscr.addstr(0, 0, "Bins data sample counts:\n{}".format(pformat({k: len(v["expected_values_detected"]) if "final_value" not in v else "All samples obtained." for k, v in bins_data.items()})))
            stdscr.addstr(11, 0, "Power: {}".format(power))
            stdscr.addstr(12, 0, "Iterations: {}".format(num_iterations))
            stdscr.addstr(13, 0, "Time Elapsed: {:.2f} seconds".format(time.time() - start_time))
            stdscr.refresh()  # Refresh the screen            
            
            num_iterations += 1
            num_iterations_while_loop_at_current_power += 1

            all_higher_bins_done = all( "final_value" in bins_data[(j/10, (j+1)/10)] for j in range(i, 10) )            
            if all_higher_bins_done or (num_iterations_while_loop_at_current_power > max_num_while_loop_iterations_before_increasing_power):     # If the higher bins have been filled or we have done more than the maximum number of while loop iterations allowed
                power *= power_step_factor
                num_iterations_while_loop_at_current_power = 0

    return bins_data


if __name__=="__main__":
    # Store all runs here before/while writing them out
    all_calculations = []
    bins_at_end = curses.wrapper(main)
    print("These are the bins and their values:")
    pprint(bins_at_end)
    plot_entropy_chart(bins_at_end)
