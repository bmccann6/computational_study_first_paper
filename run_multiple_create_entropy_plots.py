#!/usr/bin/env python3
import json
import os
import subprocess
import argparse
import entropy_plot_input_variables

def main():
    # Load the original configuration
    with open(args.config, "r") as f:
        config = json.load(f)
    
    # Iterate over some budgets
    # for budget in range(100000, 2300000 + 1, 25000):
    for budget in [110000]:
        print(f"Running for budget: {budget}")
        # <-- Modified: Update the budget value in the config
        config["budget"] = budget
        
        # Write the modified config to a temporary file
        temp_config_file = f"config_files/temp_config.json"
        with open(temp_config_file, "w") as f:
            json.dump(config, f, indent=4)
        
        # Run create_entropy_plots.py with the temporary config file
        result = subprocess.run(["python", "create_entropy_plots.py", "-config", temp_config_file, "-prob_distributions", args.prob_distributions])
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load configuration for entropy plot calculations.")
    parser.add_argument('-config', type=str, required=True, help='Path to the JSON configuration file.')
    parser.add_argument('-prob_distributions', type=str, required=True, help='Path to the Pickle file of probability distributions')    
    args = parser.parse_args()
    
    item_vals, resource_sets, num_resource_sets, hiding_locations, NUM_HIDING_LOCATIONS, sizes_hiding_locations_each_year, detectors, budget, NUM_SAMPLES_PER_BIN, NUM_BINS = entropy_plot_input_variables.get_configuration(args.config)
    main()
