"""
This module is used to recreate plots from existing json output data. It can be helpful so that we don't have to rerun create_entropy_plots.py.
Also, we have a function in here for creating a custom plot, should we want to make a plot different than the ones in create_entropy_plots.py using the output json data.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
from matplotlib.ticker import FuncFormatter
import datetime
import setup_data
from create_entropy_plots import plot_entropy_vs_final_fraction_value_detected
from create_entropy_plots import plot_entropy_vs_final_expected_value_detected


def create_CUSTOM_plot_entropy_vs_final_expected_value_detected(data):
    """
    This is for a custom plot. You can adjust anything needed in here.
    """
        
    x_labels = [bin_range for bin_range in data]
    y_values = [(3.8/2) * data[bin_range]["average_expected_value_detected_across_all_prob_distributions_this_bin"] for bin_range in data]
    
    y_min = 250e6
    y_max = 264e6

    # Calculate the y-ticks to include a tick at the min and max, and evenly spaced in between
    y_ticks = np.arange(250e6, 264e6 + 2e6, 2e6)
        
    plt.figure(figsize=(15, 8)) 
    plt.title("Normalized Entropy vs. Average Expected Value Detected", fontsize=14)
    plt.bar(x_labels, y_values)
    plt.xticks(rotation=45, fontsize=11)
    plt.xlabel("\nEntropy Bins", fontsize=14)
    plt.ylabel("USD (Millions)\n", fontsize=14)
    plt.yticks(y_ticks, [f'${int(y/1e6)}M' for y in y_ticks], fontsize=11)
    plt.ylim(y_min, y_max)
    plt.subplots_adjust(bottom=0.2, top=0.9)  # Adjust subplot margins to avoid cut-off    
    plt.savefig(f"entropy_plots/recreated_plots/CUSTOM_created_entropy_vs_value_detected.png")
    # plt.show()

def format_data_for_plot_recreations(data_path):
    with open(data_path, 'r') as file:
        raw_data = json.load(file)

    data = {}
    for bin_range, values in raw_data['bins_at_end'].items():
        data[bin_range] = {
            "expected_fraction_value_detected_across_all_prob_distributions_this_bin": values["expected_fraction_value_detected_across_all_prob_distributions_this_bin"],
            "average_expected_value_detected_across_all_prob_distributions_this_bin": values["average_expected_value_detected_across_all_prob_distributions_this_bin"]
        }
    
    return data

if __name__=="__main__":

    parser = argparse.ArgumentParser(description="Load configuration for entropy plot calculations.")
    parser.add_argument('-config_path', type=str, required=True, help='Path to the JSON configuration file.')
    parser.add_argument('-data_path', type=str, required=True, help='Path to the output JSON data of some create_entropy_plots.py run. This is the output data for which you want to recreate the plots.')
    args = parser.parse_args()
    data = format_data_for_plot_recreations(args.data_path)    
    item_vals, resource_sets, _, fraction_cargo_containers_inbound_to_US_storing_drugs, hiding_locations, NUM_HIDING_LOCATIONS, sizes_hiding_locations, detectors, budget = setup_data.get_configuration(args.config_path)
    time_run_starts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


    plot_entropy_vs_final_fraction_value_detected(data, path_to_save_fig=f"entropy_plots/recreated_plots/recreated_entropy_vs_fraction_detected.png")
    plot_entropy_vs_final_expected_value_detected(data, path_to_save_fig=f"entropy_plots/recreated_plots/recreated_entropy_vs_value_detected.png") 
    create_CUSTOM_plot_entropy_vs_final_expected_value_detected(data)