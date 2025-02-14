import json
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint, pformat
import time
import curses
from entropy_plot_input_variables import item_values, resource_sets, r, sizes_hiding_locations, detector_accuracies, required_num_samples_per_bin


def validate_data():
    if len(sizes_hiding_locations) != len(detector_accuracies):
        raise ValueError("sizes_hiding_locations and detector_accuracies must have the same length")

    for year, drugs in resource_sets.items():
        for drug in drugs:
            if drug not in item_values:
                raise ValueError(f"Drug '{drug}' in year '{year}' is not a valid item in item_values")

def generate_probabilities(power):
    random_numbers = np.power(np.random.random(r), power)
    probability_values = random_numbers / random_numbers.sum()
    return {key: prob for key, prob in zip(resource_sets.keys(), probability_values)}

def calculate_normalized_entropy(probabilities):
    probs = np.array(list(probabilities.values()))
    entropy = -np.sum(probs * np.log2(probs)) / np.log2(r)
    return entropy

# NOTE: These are the A_i values in the algorithm mathieu wrote. NOTE! They are not the expected value of items at each node i.
def calculate_backloading_stack_values(resource_set_dict):
    hider_resources_sorted = sorted(resource_set_dict.items(), key=lambda x: item_values[x[0]])
    local_copy_sizes_hiding_locations = sizes_hiding_locations.copy()
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
        item_value = item_values[item_name]
        backloading_stack_values[i] += assignment_amount * item_value
        local_copy_sizes_hiding_locations[0] -= assignment_amount
        hider_resources_sorted[0] = (hider_resources_sorted[0][0], hider_resources_sorted[0][1] - assignment_amount)
        
    return backloading_stack_values

def calculate_breaking_indices(backloading_stack_values):
    n = len(sizes_hiding_locations)
    i_0 = 0
    l = 0
    breaking_indices = [i_0]
    
    while True:
        current_max = float('-inf')
        best_k = None
        
        for k in range(breaking_indices[l], n):
            segment_sum = sum(backloading_stack_values[j] for j in range(breaking_indices[l], k + 1))
            segment_length = k + 1 - breaking_indices[l]
            avg = segment_sum / segment_length
            
            if avg >= current_max:
                current_max = avg
                best_k = k + 1
        
        breaking_indices.append(best_k)
        
        if breaking_indices[l+1] == n:
            return breaking_indices
        else:
            l += 1

def calculate_expected_value_items_at_each_node_under_equilibrium(backloading_stack_values, breaking_indices):
    expected_values = {}
    for node in range(len(backloading_stack_values)):
        relevant_indices = [
            i for i in range(len(breaking_indices) - 1)
            if breaking_indices[i] <= node < breaking_indices[i + 1]
        ]
        expected_values[node] = np.mean([
            backloading_stack_values[j]
            for j in range(breaking_indices[relevant_indices[0]], breaking_indices[relevant_indices[0] + 1])
        ])
        
    return expected_values

def calculate_expected_value_item_each_node_and_breaking_indices_specific_resource_set(resource_set_dict):
    backloading_stack_values = calculate_backloading_stack_values(resource_set_dict)    
    breaking_indices = calculate_breaking_indices(backloading_stack_values)
    expected_value_items_each_node = calculate_expected_value_items_at_each_node_under_equilibrium(backloading_stack_values, breaking_indices)
    
    return expected_value_items_each_node, breaking_indices

def calculate_expected_and_total_values_detected_this_run_across_resource_sets(probabilities):
    # We’ll collect each year’s breaking_indices in a structure to save
    resource_sets_info = []
    expected_value_detected_this_run = 0
    expected_total_value_this_run = 0

    for year, resource_set_dict in resource_sets.items():
        expected_value_items_each_node, breaking_indices = calculate_expected_value_item_each_node_and_breaking_indices_specific_resource_set(resource_set_dict)
        expected_value_detected_this_run_specific_resource_set = sum(detector_accuracies[i] * expected_value_items_each_node[i] for i in range(len(expected_value_items_each_node)))        
        expected_total_value_this_run_specific_resource_set = sum(expected_value_items_each_node[i] for i in range(len(expected_value_items_each_node)))
        resource_sets_info.append({
            "year": year,
            "breaking_indices": breaking_indices
        })
        expected_value_detected_this_run += probabilities[year] * expected_value_detected_this_run_specific_resource_set
        expected_total_value_this_run += probabilities[year] * expected_total_value_this_run_specific_resource_set

    # Log to JSON each time we compute for these probabilities
    log_entry = {
        "probabilities": probabilities,
        "resource_sets": resource_sets_info
    }
    all_calculations.append(log_entry)
    
    # Write out after each iteration so you can see updates in real time
    with open("breaking_indices_log.json", "w") as f:
        json.dump(all_calculations, f, indent=4)

    return expected_value_detected_this_run, expected_total_value_this_run

def update_bin_dict(dict_bins_and_values, entropy, expected_value_detected_this_run, expected_total_value_this_run):
    """
    For the bin containing 'entropy', append the given sample to its lists,
    and if the bin has reached the required number of samples, compute the
    final value and store it in the 'final_value' key.
    """
    for (lower_bound, upper_bound), bin_data in dict_bins_and_values.items():
        # Skip bins that do not include the current entropy
        if not (lower_bound <= entropy < upper_bound):
            continue
        
        # If 'final_value' is already computed, we do nothing for this bin
        if "final_value" in bin_data:
            break  # Because we found our bin, no need to continue iterating
        
        # Collect new sample
        bin_data["expected_values_detected"].append(expected_value_detected_this_run)
        bin_data["expected_total_values"].append(expected_total_value_this_run)
        
        # If we've reached the required number of samples, compute the final value
        if len(bin_data["expected_values_detected"]) == required_num_samples_per_bin:
            total_detected = sum(bin_data["expected_values_detected"])
            total_all = sum(bin_data["expected_total_values"])
            bin_data["final_value"] = total_detected / total_all
            
        break

    return dict_bins_and_values



def plot_entropy_bar_chart(dict_bins_and_values):
    x_labels = [f"{k[0]:.1f}-{k[1]:.1f}" for k in dict_bins_and_values]
    y_values = [dict_bins_and_values[k] for k in dict_bins_and_values]
 
    # Calculate the minimum of y_values rounded down to the nearest 2 decimal places
    y_min = np.floor(min(y_values) * 100) / 100 
    
    plt.bar(x_labels, y_values)
    plt.xlabel("Entropy Bins")
    plt.ylabel("Average Fraction Undetected")
    plt.title("Entropy vs. Fraction Undetected")
    plt.ylim(y_min)  # Set the lower limit of the y-axis    
    plt.savefig("entropy_vs_fraction_undetected.png")
    plt.show()

def main(stdscr):
    # Clear screen
    stdscr.clear()
    
    start_time = time.time()
    validate_data()
    power = 1
    power_step_factor = 2
    dict_bins_and_values = {(i/10, (i+1)/10): {"expected_values_detected": [], "expected_total_values": []} for i in range(10)}
    max_num_while_loop_iterations_before_increasing_power = required_num_samples_per_bin * 10
    num_iterations_while_loop_at_current_power = 0
    num_iterations = 0
    
    for i in range(9, -1, -1):
        while ("final_value" not in dict_bins_and_values[(i/10, (i+1)/10)] and len(dict_bins_and_values[(i/10, (i+1)/10)]["expected_values_detected"]) < required_num_samples_per_bin):
            probabilities = generate_probabilities(power)
            entropy = calculate_normalized_entropy(probabilities)
            expected_value_detected_this_run, expected_total_value_this_run = calculate_expected_and_total_values_detected_this_run_across_resource_sets(probabilities)
            dict_bins_and_values = update_bin_dict(dict_bins_and_values, entropy, expected_value_detected_this_run, expected_total_value_this_run)
            
            # The following is for printing progress to the terminal.
            stdscr.addstr(0, 0, "Dict_bins_and_values: {}".format(pformat({k: len(v) if isinstance(v, list) else "All samples obtained." for k, v in dict_bins_and_values.items()})))
            stdscr.addstr(10, 0, "Power: {}".format(power))
            stdscr.addstr(11, 0, "Iterations: {}".format(num_iterations))
            stdscr.addstr(12, 0, "Time Elapsed: {:.2f} seconds".format(time.time() - start_time))
            stdscr.refresh()  # Refresh the screen
            
            num_iterations += 1
            num_iterations_while_loop_at_current_power += 1

            all_higher_bins_done = all( "final_value" in dict_bins_and_values[(j/10, (j+1)/10)] for j in range(i, 10) )            
            if all_higher_bins_done or (num_iterations_while_loop_at_current_power > max_num_while_loop_iterations_before_increasing_power):     # If the higher bins have been filled or we have done more than the maximum number of while loop iterations allowed
                power *= power_step_factor
                num_iterations_while_loop_at_current_power = 0


    print("These are the bins and their values:")
    pprint(dict_bins_and_values)
    plot_entropy_bar_chart(dict_bins_and_values)

if __name__=="__main__":
    # Store all runs here before/while writing them out
    all_calculations = []
    curses.wrapper(main)
