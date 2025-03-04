#!/usr/bin/env python3
import json
import os
import subprocess
import argparse
import entropy_plot_input_variables

def main():
    # Load the original configuration
    with open(args.config_path, "r") as f:
        config = json.load(f)
    
    # Iterate over some budgets
    for budget in budgets:
        print(f"Running for budget: {budget}")
        # <-- Modified: Update the budget value in the config
        config["budget"] = budget
        
        # Write the modified config to a temporary file
        temp_config_file = f"config_files/temp_config.json"
        with open(temp_config_file, "w") as f:
            json.dump(config, f, indent=4)
        
        # Run create_entropy_plots.py with the temporary config file
        result = subprocess.run(["python", "create_entropy_plots.py", "-config_path", temp_config_file, "-prob_distributions_path", args.prob_distributions_path])
        if result.returncode != 0:
            print(f"Error running create_entropy_plots.py for budget {budget}")
            continue
        
        
        # Rename the output plot to include the budget value in the filename
        src_fraction_plot = "entropy_plots/entropy_vs_fraction_detected_plots/entropy_vs_fraction_detected.png"
        dst_fraction_plot = f"entropy_plots/entropy_vs_fraction_detected_plots/entropy_vs_fraction_detected_budget_{budget}_and_{NUM_SAMPLES_PER_BIN}_samples_per_bin_and_NUM_BINS_{NUM_BINS}.png"
        src_value_plot = "entropy_plots/entropy_vs_value_detected_plots/entropy_vs_value_detected.png"
        dst_value_plot = f"entropy_plots/entropy_vs_value_detected_plots/entropy_vs_value_detected_budget_{budget}_and_{NUM_SAMPLES_PER_BIN}_samples_per_bin_and_NUM_BINS_{NUM_BINS}.png"
        if os.path.exists(src_fraction_plot):
            os.replace(src_fraction_plot, dst_fraction_plot)
        if os.path.exists(src_value_plot):
            os.replace(src_value_plot, dst_value_plot)
                
        # Remove the temporary config file
        os.remove(temp_config_file)

        print(f"Finished running for budget: {budget}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load configuration for entropy plot calculations.")
    parser.add_argument('-config_path', type=str, required=True, help='Path to the JSON configuration file.')
    parser.add_argument('-prob_distributions_path', type=str, required=True, help='Path to the Pickle file of probability distributions')    
    args = parser.parse_args()
    
    _, _, _, _, NUM_HIDING_LOCATIONS, _, detectors, budget = entropy_plot_input_variables.get_configuration(args.config_path)
    prob_distributions_dict, NUM_SAMPLES_PER_BIN, NUM_BINS = entropy_plot_input_variables.get_prob_distributions_dict(args.prob_distributions_path)


    # Calculate the smallest difference between any pair of detectors
    detector_costs = [detector["cost"] for detector in detectors.values()]  
    budget_minimum = min(detector_costs) * 3        # This is a decent lower bound.
    budget_maximum =  max(detector_costs) * NUM_HIDING_LOCATIONS
    # budget_step_size = min(abs(detector_costs[i] - detector_costs[j]) for i in range(len(detector_costs)) for j in range(i + 1, len(detector_costs))) 
    budget_step_size = 50000
    # budgets = range(budget_minimum, budget_maximum + 1, budget_step_size)
    budgets = [1000000]
    
        
    main()
