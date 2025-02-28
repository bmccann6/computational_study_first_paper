import numpy as np
from pprint import pprint, pformat
import time
import curses
import argparse
import pickle
from multiprocessing import Pool, cpu_count
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
def worker_task(resource_sets, num_resource_sets, power, batch_size):
    results = []
    for _ in range(batch_size):
        random_numbers = np.power(np.random.random(num_resource_sets), power)
        probability_values = random_numbers / random_numbers.sum()
        prob_dist = {year: prob for year, prob in zip(resource_sets.keys(), probability_values)}

        probs = np.array(list(prob_dist.values()))
        nonzero_probs = probs[probs > 0]
        entropy = -np.sum(nonzero_probs * np.log2(nonzero_probs)) / np.log2(num_resource_sets)

        results.append((prob_dist, entropy))
    return results



def main(stdscr):
    # Clear screen
    stdscr.clear()
 
    # We can choose how many processes and how large each batch is
    num_processes = cpu_count()  # or manually set fewer if you like
    batch_size_per_proc = 1000   # how many samples each process should do per chunk (tweak to find your sweet spot) 
    
    start_time = time.time()
    power = 1
    power_step_factor = 1.25        # From experimenting, I found that this value makes the code run decently fast
    bins_data = {(i/NUM_BINS, (i+1)/NUM_BINS): [] for i in range(NUM_BINS)}
    max_num_while_loop_iterations_before_increasing_power = NUM_SAMPLES_NEEDED_PER_BIN * NUM_BINS
    num_iterations_while_loop_at_current_power = 0
    num_iterations = 0
         
    for i in range(NUM_BINS - 1, -1, -1):
        bin_key = (i/NUM_BINS, (i+1)/NUM_BINS)
        while len(bins_data[bin_key]) < NUM_SAMPLES_NEEDED_PER_BIN:
            # ----- Parallel batch sampling -----
            # Create tasks for each process
            tasks = [(resource_sets, num_resource_sets, power, batch_size_per_proc) for _ in range(num_processes)]

            with Pool(processes=num_processes) as pool:
                # results is a list of lists; flatten it afterwards
                batch_results = pool.starmap(worker_task, tasks)
            # Flatten
            all_pairs = []
            for sublist in batch_results:
                all_pairs.extend(sublist)

            # Update bins_data using all samples from this batch
            for prob_dist, entropy in all_pairs:
                bins_data = update_bin_dict(bins_data, prob_dist, entropy)
                num_iterations += 1
                num_iterations_while_loop_at_current_power += 1

            # Print progress to the terminal using curses
            bin_counts = {k: (len(v) if len(v) != NUM_SAMPLES_NEEDED_PER_BIN else "All samples obtained.") for k, v in bins_data.items()}
            stdscr.addstr(0, 0, "Bins data sample counts:\n{}".format(pformat(bin_counts)))
            stdscr.addstr(NUM_BINS + 1, 0, f"Power: {power}")
            stdscr.addstr(NUM_BINS + 2, 0, f"Iterations: {num_iterations}")
            stdscr.addstr(NUM_BINS + 3, 0, "Time Elapsed: {:.2f} seconds".format(time.time() - start_time))
            stdscr.refresh()

            # If the bin at 'i' plus all bins above it are filled or we've done too many iterations, increase power
            all_higher_bins_done = all( len(bins_data[(j/NUM_BINS, (j+1)/NUM_BINS)]) == NUM_SAMPLES_NEEDED_PER_BIN for j in range(i, NUM_BINS)
            )
            if all_higher_bins_done or (num_iterations_while_loop_at_current_power > max_num_while_loop_iterations_before_increasing_power):
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
    
