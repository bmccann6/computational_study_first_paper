import numpy as np
from pprint import pprint, pformat
import time
import curses
import argparse
import pickle
import entropy_plot_input_variables


def generate_probability_distribution(power):
    random_numbers = np.power(np.random.random(num_resource_sets), power)
    probability_values = random_numbers / random_numbers.sum()
    return {year: prob for year, prob in zip(resource_sets.keys(), probability_values)}

def calculate_normalized_entropy(prob_dist):
    probs = np.array(list(prob_dist.values()))
    nonzero_probs = probs[probs > 0]        # Only take log of non-zero entries just in case we get division by 0 errors.
    entropy = -np.sum(nonzero_probs * np.log2(nonzero_probs)) / np.log2(num_resource_sets)
    return entropy

def update_bin_dict(bins_data, prob_dist, entropy):
    """
    For the bin containing 'entropy', append the given sample to its lists, and if the bin has reached the required number of samples, stop.
    """
    for (lower_bound, upper_bound), bin_data in bins_data.items():
        # Skip bins that do not include the current entropy
        if not (lower_bound <= entropy < upper_bound):
            continue
        
        if len(bin_data) == NUM_SAMPLES_NEEDED_PER_BIN:
            break
        
        # Collect new sample
        bin_data.append(prob_dist)
        
    return bins_data


def main(stdscr):
    # Clear screen
    stdscr.clear()
    
    start_time = time.time()
    power = 1
    power_step_factor = 1.25        # From experimenting, I found that this value makes the code run decently fast
    bins_data = {(i/NUM_BINS, (i+1)/NUM_BINS): [] for i in range(NUM_BINS)}
    max_num_while_loop_iterations_before_increasing_power = NUM_SAMPLES_NEEDED_PER_BIN * NUM_BINS
    num_iterations_while_loop_at_current_power = 0
    num_iterations = 0
         
    for i in range(NUM_BINS-1, -1, -1):
        while len(bins_data[(i/NUM_BINS, (i+1)/NUM_BINS)]) < NUM_SAMPLES_NEEDED_PER_BIN:
            prob_dist = generate_probability_distribution(power)
            entropy = calculate_normalized_entropy(prob_dist)
            bins_data = update_bin_dict(bins_data, prob_dist, entropy)
            
            # The following is for printing progress to the terminal.
            stdscr.addstr(0, 0, "Bins data sample counts:\n{}".format(pformat({k: len(v) if len(v) != NUM_SAMPLES_NEEDED_PER_BIN else "All samples obtained." for k, v in bins_data.items()})))
            stdscr.addstr(NUM_BINS + 1, 0, "Power: {}".format(power))
            stdscr.addstr(NUM_BINS + 2, 0, "Iterations: {}".format(num_iterations))
            stdscr.addstr(NUM_BINS + 3, 0, "Time Elapsed: {:.2f} seconds".format(time.time() - start_time))
            stdscr.refresh()  # Refresh the screen            
            
            num_iterations += 1
            num_iterations_while_loop_at_current_power += 1

            all_higher_bins_done = all(  len(bins_data[(j/NUM_BINS, (j+1)/NUM_BINS)]) == NUM_SAMPLES_NEEDED_PER_BIN for j in range(i, NUM_BINS) )            
            if all_higher_bins_done or (num_iterations_while_loop_at_current_power > max_num_while_loop_iterations_before_increasing_power):     # If the higher bins have been filled or we have done more than the maximum number of while loop iterations allowed
                power *= power_step_factor
                num_iterations_while_loop_at_current_power = 0

    return bins_data


if __name__=="__main__":

    parser = argparse.ArgumentParser(description="Load configuration for entropy plot calculations.")           # We just need the config file to get the resource sets
    parser.add_argument('-config', type=str, required=True, help='Path to the JSON configuration file.')
    args = parser.parse_args()
    item_vals, resource_sets, num_resource_sets, hiding_locations, NUM_HIDING_LOCATIONS, sizes_hiding_locations_each_year, detectors, budget, NUM_SAMPLES_NEEDED_PER_BIN, NUM_BINS = entropy_plot_input_variables.get_configuration(args.config)

    # Store all runs here before/while writing them out
    bins_at_end = curses.wrapper(main)
    
    print("Creating pickle file...")
    with open(f"generated_prob_dist_data/prob_distributions_generated_NUM_SAMPLES_PER_BIN_{NUM_SAMPLES_NEEDED_PER_BIN}_and_NUM_BINS_{NUM_BINS}.pkl", "wb") as file:
        pickle.dump(bins_at_end, file)
    print("Finished creating pickle file.")
    

