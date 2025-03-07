"""
Overview:
    This module generates probability distributions for a given NUM_BINS_TO_HAVE and NUM_SAMPLES_TO_GENERATE_PER_BIN.
    It saves the generated probability distributions in a pickle file, and we organize the probability distributions by the entropy bin they fall into.
    It is useful to have these probability distributions saved so that we don't have to generate them everytime we run create_entropy_plots.py. 
    Having the probability distributions already generated and saved in a file saves a lot of time.

Features:
    -We parallelize the generation of probability distributions. This saves a lot of time.
    -Everytime the current highest unfilled bin becomes filled, we increase the exponent on the random variables drawn. 
     This speeds up the generation of lower entropy probability distributions, as raising to a larger exponent increases the spread of the generated random variables.
"""


from pprint import pformat
import numpy as np
import time
import curses
import argparse
import pickle
from multiprocessing import Pool, cpu_count
import setup_data


def generate_probability_distribution(power, resource_sets, NUM_RESOURCE_SETS):
    """
    Generates a probability distribution based on a given power.

    This function generates a set of random numbers, raises them to the specified power,
    and then normalizes them to create a probability distribution. The resulting probabilities
    are mapped to the keys of the `resource_sets` dictionary.
    
    Args:
        power (float): The power to which each random number is raised.

    Returns:
        dict: A dictionary where the keys are from `resource_sets` and the values are the 
              corresponding probability values.
    """
    random_numbers = np.power(np.random.random(NUM_RESOURCE_SETS), power)
    probability_values = random_numbers / random_numbers.sum()
    return {year: prob for year, prob in zip(resource_sets.keys(), probability_values)}

def calculate_normalized_entropy(prob_dist, NUM_RESOURCE_SETS):
    probs = np.array(list(prob_dist.values()))
    nonzero_probs = probs[probs > 0]        # Only take log of non-zero entries just in case we get division by 0 errors.
    entropy = -np.sum(nonzero_probs * np.log2(nonzero_probs)) / np.log2(NUM_RESOURCE_SETS)
    return entropy

def update_bin_dict(bins_data, prob_dist, entropy, NUM_SAMPLES_TO_GENERATE_PER_BIN):
    """
    For the bin containing 'entropy', append to that bin's list a dictionary where the keys are the prob_dist and entropy of that prob_dist.
    If the bin has reached the required number of samples, stop.
    """
    for (lower_bound, upper_bound), bin_data in bins_data.items():
        # Skip bins that do not include the current entropy
        if not (lower_bound <= entropy < upper_bound):
            continue
        
        if len(bin_data) == NUM_SAMPLES_TO_GENERATE_PER_BIN:
            break
        
        prob_dist_and_entropy = {"prob_dist": prob_dist, "entropy": entropy}
        # Collect new sample
        bin_data.append(prob_dist_and_entropy)
        
    return bins_data

def worker_task(resource_sets, NUM_RESOURCE_SETS, power, batch_size_per_proc):
    """
    Processes a batch of samples to generate probability distributions and compute their normalized entropies.

    This function is intended to run in parallel across multiple processes. Each parallel process is given a number of probability distributions to generate (batch_size_per_proc). 
    For each sample in the batch, it generates a probability distribution by:
      - Drawing `NUM_RESOURCE_SETS` random numbers.
      - Raising these random numbers to the specified `power`.
      - Normalizing the results so that they sum to 1, mapping them to the keys in `resource_sets`.
      - Calculating the normalized entropy of the resulting distribution.

    Args:
        resource_sets (dict): Dictionary representing the resource sets. The keys are used to map probability values.
        NUM_RESOURCE_SETS (int): The number of random values to generate, corresponding to the number of resource sets.
        power (float): The exponent applied to each random number to adjust the spread of the generated values.
        batch_size_per_proc (int): The number of samples to generate in a batch. 

    Returns:
        list: A list of tuples, where each tuple contains:
              - dict: A generated probability distribution with keys from `resource_sets`.
              - float: The normalized entropy of the probability distribution.
    """    
    
    results = []
    for _ in range(batch_size_per_proc):
        random_numbers = np.power(np.random.random(NUM_RESOURCE_SETS), power)
        probability_values = random_numbers / random_numbers.sum()
        prob_dist = {year: prob for year, prob in zip(resource_sets.keys(), probability_values)}

        probs = np.array(list(prob_dist.values()))
        nonzero_probs = probs[probs > 0]
        entropy = -np.sum(nonzero_probs * np.log2(nonzero_probs)) / np.log2(NUM_RESOURCE_SETS)

        results.append((prob_dist, entropy))
        
    return results


# def main(stdscr):
def main(stdscr, resource_sets, NUM_RESOURCE_SETS, NUM_SAMPLES_TO_GENERATE_PER_BIN, NUM_BINS_TO_HAVE):
    """
    This function executes a loop for generating probability distributions and categorizing them by normalized entropy.

    This function runs within a curses wrapper to provide real-time terminal updates. It uses
    multiprocessing to generate batches of probability distributions, computes their normalized entropies,
    and organizes them into bins defined by entropy intervals. The power applied to the random numbers
    is dynamically adjusted based on the progress of sample collection in the bins.

    The process involves:
      1. Setting up multiprocessing parameters such as the number of processes and batch size per process.
      2. Initializing variables for tracking time, current power, and the entropy bins (bins_data).
      3. Iterating over entropy bins in reverse order to fill higher entropy bins first.
      4. For each bin:
         - Generating probability distributions and calculating their entropies in parallel using the worker_task function.
         - Flattening and processing the results (from the parallel worker_task calls) to update the bins with the new samples.
         - Displaying current progress on the terminal, including the number of samples per bin,
           current power, iteration count, and elapsed time.
         - Adjusting the power if the current bin and all higher bins are filled or if too many iterations
           have occurred without filling the bin.
    
    Note that the argument stdscr (curses.window) is the curses window object used for displaying progress updates in the terminal.
    
    Returns:
        dict: A dictionary where the keys are tuples representing the lower and upper bounds of entropy bins,
              and the values are lists of the sample dictionaries, where here a sample dictionary includes a generated
              probability distribution and its corresponding normalized entropy.
    """    
    
    # Clear screen
    stdscr.clear()
 
    num_processes = cpu_count()  
    batch_size_per_proc = 1000      # Defines the number of samples each process will generate per batch
    
    start_time = time.time()
    power = 1           # Initializes the exponent used to modify the spread of generated random numbers. We raise each generated random number to this power.
    power_step_factor = 1.25        # From experimenting, I found that this value makes the code run decently fast
    bins_data = {(i/NUM_BINS_TO_HAVE, (i+1)/NUM_BINS_TO_HAVE): [] for i in range(NUM_BINS_TO_HAVE)}     # Keys are tuples representing entropy intervals. We will populate the empty list value with sample dictionaries as described in the docstring for this function.
    max_num_while_loop_iterations_before_increasing_power = NUM_SAMPLES_TO_GENERATE_PER_BIN * NUM_BINS_TO_HAVE      # Useful to prevent us getting stuck in a while loop for a power that is not generating samples fast enough to fill up the highest entropy bin remaining unfilled.
    num_iterations_while_loop_at_current_power = 0
    num_iterations = 0
         
    # Loop over entropy bins in reverse order (starting with the highest entropy bin). We do this because when the power is lower, the generated probability distributions will most likely have an entropy value which falls in a higher bin. So those higher entropy bins will fill up faster.
    for i in range(NUM_BINS_TO_HAVE - 1, -1, -1):
        bin_key = (i/NUM_BINS_TO_HAVE, (i+1)/NUM_BINS_TO_HAVE)
        
        # Continue generating samples until the current bin is filled with the required number of samples        
        while len(bins_data[bin_key]) < NUM_SAMPLES_TO_GENERATE_PER_BIN:
            tasks = [(resource_sets, NUM_RESOURCE_SETS, power, batch_size_per_proc) for _ in range(num_processes)]    # Create tasks for parallel processing; each task contains parameters for generating a batch of samples
            
            # Use a multiprocessing Pool to execute worker_task concurrently across all tasks
            with Pool(processes=num_processes) as pool:
                batch_results = pool.starmap(worker_task, tasks)     # results is a list of lists; flatten it afterwards
                
            # Flatten the list of lists from the parallel processes into a single list of (prob_dist, entropy) tuples
            all_pairs = []
            for sublist in batch_results:
                all_pairs.extend(sublist)

            # Update the appropriate bin in bins_data and update iteration counters
            for prob_dist, entropy in all_pairs:
                bins_data = update_bin_dict(bins_data, prob_dist, entropy, NUM_SAMPLES_TO_GENERATE_PER_BIN)
                num_iterations += 1
                num_iterations_while_loop_at_current_power += 1

            # Print progress to the terminal using curses
            bin_counts = {k: (len(v) if len(v) != NUM_SAMPLES_TO_GENERATE_PER_BIN else "All samples obtained.") for k, v in bins_data.items()}
            stdscr.addstr(0, 0, "Bins data sample counts:\n{}".format(pformat(bin_counts)))
            stdscr.addstr(NUM_BINS_TO_HAVE + 1, 0, f"Power: {power}")
            stdscr.addstr(NUM_BINS_TO_HAVE + 2, 0, f"Iterations: {num_iterations}")
            stdscr.addstr(NUM_BINS_TO_HAVE + 3, 0, "Time Elapsed: {:.2f} seconds".format(time.time() - start_time))
            stdscr.refresh()

            # In the next part, we check if the current bin and all higher bins are filled or if the iteration limit at the current power is exceeded
            all_higher_bins_done = all( len(bins_data[(j/NUM_BINS_TO_HAVE, (j+1)/NUM_BINS_TO_HAVE)]) == NUM_SAMPLES_TO_GENERATE_PER_BIN for j in range(i, NUM_BINS_TO_HAVE) )
            if all_higher_bins_done or (num_iterations_while_loop_at_current_power > max_num_while_loop_iterations_before_increasing_power):
                power *= power_step_factor                  # Increase the power to help generate distributions that fill the remaining (lower entropy) bins faster
                num_iterations_while_loop_at_current_power = 0      # Reset the iteration counter for the current power level

    # Return the completed dictionary of bins containing generated samples
    return bins_data        


if __name__=="__main__":

    parser = argparse.ArgumentParser(description="Load configuration for entropy plot calculations.")           # We just need the config file to get the resource sets
    parser.add_argument('-config_path', type=str, required=True, help='Path to the JSON configuration file.')
    args = parser.parse_args()
    _, resource_sets, NUM_RESOURCE_SETS, fraction_cargo_containers_inbound_to_US_storing_drugs, _, _, _, _, _= setup_data.get_configuration(args.config_path)

    NUM_SAMPLES_TO_GENERATE_PER_BIN = 200
    NUM_BINS_TO_HAVE = 20

    # Run the main function within a curses wrapper to handle terminal display; capture the generated bins data
    bins_at_end = curses.wrapper(lambda stdscr: main(stdscr, resource_sets, NUM_RESOURCE_SETS, NUM_SAMPLES_TO_GENERATE_PER_BIN, NUM_BINS_TO_HAVE))
    
    
    
    print("Creating pickle file...")
    with open(f"generated_prob_dist_data/prob_distributions_generated_NUM_SAMPLES_PER_BIN_{NUM_SAMPLES_TO_GENERATE_PER_BIN}_and_NUM_BINS_{NUM_BINS_TO_HAVE}.pkl", "wb") as file:
        pickle.dump(bins_at_end, file)
    print("Finished creating pickle file.")
    
