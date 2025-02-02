import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
import time
from entropy_plot_input_variables import item_values, resource_sets, r, sizes_hiding_locations, detector_accuracies, required_num_samples_per_bin


def error_checking():
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

def calculate_A_i_values(resource_set_dict):
    """
    This function calculates the A_i values from just backloading. Like the A_i values in the writeup mathieu gave for calculating the breaking indices.
    """
    
    hider_resources_sorted = sorted(resource_set_dict.items(), key=lambda x: item_values[x[0]])     # Creates a list of tuples (drug, quantity) from resource_set_dict and sorts the tuples based on the value of the drug, from low to high.
    local_copy_sizes_hiding_locations = sizes_hiding_locations.copy()       # We make a copy so that we don't modify sizes_hiding_locations outside this function
    A_i_values = {i: 0 for i in range(len(sizes_hiding_locations))}
    i = 0
    
    while local_copy_sizes_hiding_locations and hider_resources_sorted:     # While both are not empty lists
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
        A_i_values[i] += assignment_amount * item_value
        local_copy_sizes_hiding_locations[0] -= assignment_amount
        hider_resources_sorted[0] = (hider_resources_sorted[0][0], hider_resources_sorted[0][1] - assignment_amount)
        
    return A_i_values
        
def calculate_breaking_indices(A_i_values):
    n = len(sizes_hiding_locations)   
    i_0 = 0
    l = 0
    breaking_indices = [i_0]
    
    while True:
        current_max = float('-inf')
        best_k = None
        
        # Calculate the maximum average for segments starting from i_l
        for k in range(breaking_indices[l], n):
            # Calculate sum and average using dictionary keys
            segment_sum = sum(A_i_values[j] for j in range(breaking_indices[l], k + 1))   
            segment_length = k + 1 - breaking_indices[l]
            avg = segment_sum / segment_length
            
            if avg >= current_max:
                current_max = avg
                best_k = k + 1
        
        breaking_indices.append(best_k)
        
        # Check if we reach the end
        if breaking_indices[l+1] == n:
            return breaking_indices
        else:
            l += 1
        
def calculate_expected_value_items_at_each_node_under_equilibrium(A_i_values, breaking_indices):
    expected_values = {}
    for node in range(len(A_i_values)):
        relevant_indices = [i for i in range(len(breaking_indices) - 1) if breaking_indices[i] <= node < breaking_indices[i + 1]]
        expected_values[node] = np.mean([A_i_values[j] for j in range(breaking_indices[relevant_indices[0]], breaking_indices[relevant_indices[0] + 1])])
        
    return expected_values

def calculate_expected_fraction_undetected_value_specific_resource_set(resource_set_dict):
    A_i_values = calculate_A_i_values(resource_set_dict)      #NOTE: We note that the A_i values here are the A_i values in the algorithm of mathieu's writeup for the algorithm at the end to find the breaking indices. It is just the initial A_i values from backloading. Not the A_i values in equilibrium. I don't want you to get the A_i values here confused with A(sigma^2, i) in my paper.
    breaking_indices = calculate_breaking_indices(A_i_values)        
    expected_value_items_each_node = calculate_expected_value_items_at_each_node_under_equilibrium(A_i_values, breaking_indices)
    total_value = sum(expected_value_items_each_node.values())
    amount_value_detected = sum(detector_accuracies[i] * expected_value_items_each_node[i] for i in range(len(expected_value_items_each_node)))
    expected_fraction_undetected_value_specific_resource_set  = (total_value - amount_value_detected) / total_value
    
    return expected_fraction_undetected_value_specific_resource_set
    
def calculate_expected_fraction_undetected_value_across_resource_sets(probabilities):
    expected_fraction_undetected_value = 0 
    for year, resource_set_dict in resource_sets.items():
        expected_fraction_undetected_value += probabilities[year] * calculate_expected_fraction_undetected_value_specific_resource_set(resource_set_dict)
        
    return expected_fraction_undetected_value
    
def update_bin_dict(dict_bins_and_values, entropy, fraction_value_detected):
    for key in dict_bins_and_values:
        if key[0] <= entropy < key[1]:
            if isinstance(dict_bins_and_values[key], list) and (len(dict_bins_and_values[key]) < required_num_samples_per_bin):
                dict_bins_and_values[key].append(fraction_value_detected)
                if len(dict_bins_and_values[key]) == required_num_samples_per_bin:     # If we have now hit required_num_samples_per_bin once we just added fraction_value_detected, then we are done with this bin and we will take the average of the values associated with that bin of the fraction of undetected value.
                    dict_bins_and_values[key] = np.mean(dict_bins_and_values[key])
            break
    return dict_bins_and_values

def plot_entropy_bar_chart(dict_bins_and_values):
    x_labels = [f"{k[0]:.1f}-{k[1]:.1f}" for k in dict_bins_and_values]
    y_values = [dict_bins_and_values[k] for k in dict_bins_and_values]
    
    plt.bar(x_labels, y_values)
    plt.xlabel("Entropy Bins")
    plt.ylabel("Average Fraction Undetected")
    plt.title("Entropy vs. Fraction Undetected")
    plt.show()


def main():    
    start_time = time.time()
    error_checking()   
    power = 1
    power_step_size = 0.5
    dict_bins_and_values = {(i/10, (i+1)/10): [] for i in range(10)}   # This will be used to store the average fraction of undetected values for the probability distributions generated which fall into each of these entropy bins
    max_num_while_loop_iterations_before_increasing_power = required_num_samples_per_bin * 100      # This is a value which signals that will be helpful to signal that we are not generating enough samples with the desired entropy range, so we should increase the power.
    num_iterations_while_loop_at_current_power = 0
        
    for i in range(9, -1, -1):
        while isinstance(dict_bins_and_values[(i/10, (i+1)/10)], list) and len(dict_bins_and_values[(i/10, (i+1)/10)]) < required_num_samples_per_bin:
            probabilities = generate_probabilities(power)
            entropy = calculate_normalized_entropy(probabilities)
            expected_fraction_value_detected = calculate_expected_fraction_undetected_value_across_resource_sets(probabilities)
            dict_bins_and_values = update_bin_dict(dict_bins_and_values, entropy, expected_fraction_value_detected)
            print("Time Elapsed: ", time.time() - start_time)
            num_iterations_while_loop_at_current_power += 1
            if num_iterations_while_loop_at_current_power > max_num_while_loop_iterations_before_increasing_power:
                power += power_step_size
                num_iterations_while_loop_at_current_power = 0
            
        
    plot_entropy_bar_chart(dict_bins_and_values)


    
if __name__=="__main__":
    main()