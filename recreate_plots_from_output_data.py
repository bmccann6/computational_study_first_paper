import json
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint, pformat
import time
import argparse
from matplotlib.ticker import FuncFormatter
import datetime
from gurobipy import Model, GRB, quicksum
import entropy_plot_input_variables



def recreate_plot_entropy_vs_final_fraction_value_detected(data):
    """
    Plot the (final) fraction detected as a function of entropy bins.
    """       
        
    x_labels = [bin_range for bin_range in data]
    y_values = [data[bin_range]["final_fraction_value_detected"] for bin_range in data]

    data_range = max(y_values) - min(y_values)
    y_axis_buffer_adjust_amount = data_range * 0.05      # This will be used for the y ticks to ensure that there isn't a bar going to the top y-tick. Its just for presentation sake.

    y_min = min(y_values) - y_axis_buffer_adjust_amount
    y_max = max(y_values) + y_axis_buffer_adjust_amount
    
    num_ticks = 10
    y_ticks = np.linspace(y_min, y_max, num_ticks)

    def format_y_value(y, pos):
        return f'{y:.4f}'
    
    plt.figure(figsize=(15, 8)) 
    plt.title("Normalized Entropy vs. Fraction Detected", fontsize=14)
    plt.bar(x_labels, y_values)
    plt.xticks(rotation=45, fontsize=11)
    plt.xlabel("\nEntropy Bins", fontsize=14)
    plt.ylabel("Average Fraction Detected\n", fontsize=14)
    plt.yticks(y_ticks, [format_y_value(y, None) for y in y_ticks], fontsize=11)  # Using the formatter for the labels
    
    formatter = FuncFormatter(format_y_value)
    plt.gca().yaxis.set_major_formatter(formatter)    
    
    plt.ylim(y_min, y_max)
    plt.subplots_adjust(bottom=0.2, top=0.9)  # Adjust subplot margins to avoid cut-off    
    plt.savefig(f"entropy_plots/recreated_plots/recreated_entropy_vs_fraction_detected.png")
    # plt.show()

def recreate_plot_entropy_vs_final_expected_value_detected(data):
    """
    Plot the (final) expected value detected as a function of entropy bins.
    """
        
    x_labels = [bin_range for bin_range in data]
    y_values = [data[bin_range]["final_expected_value_detected"] for bin_range in data]
    
    data_range = max(y_values) - min(y_values)
    y_axis_buffer_adjust_amount = data_range * 0.05      # This will be used for the y ticks to ensure that there isn't a bar going to the top y-tick. Its just for presentation sake.

    y_min = min(y_values) - y_axis_buffer_adjust_amount
    y_max = max(y_values) + y_axis_buffer_adjust_amount

    # Calculate the y-ticks to include a tick at the min and max, and evenly spaced in between
    num_ticks = 10  # Define number of ticks you want (including min and max)
    y_ticks = np.linspace(y_min, y_max, num_ticks)
        
    def format_y_value(y, pos):
        if y >= 1e9:
            return f'${y/1e9:.2f}B'
        elif y >= 1e6:
            return f'${y/1e6:.2f}M'
        elif y >= 1e3:
            return f'${y/1e3:.2f}K'
        return f'${y:.2f}'
    
    plt.figure(figsize=(15, 8)) 
    plt.title("Normalized Entropy vs. Average Expected Value Detected", fontsize=14)
    plt.bar(x_labels, y_values)
    plt.xticks(rotation=45, fontsize=11)
    plt.xlabel("\nEntropy Bins", fontsize=14)
    plt.ylabel("USD (Millions)\n", fontsize=14)
    plt.yticks(y_ticks, [format_y_value(y, None) for y in y_ticks], fontsize=11)  # Using the formatter for the labels 
    formatter = FuncFormatter(format_y_value)
    plt.gca().yaxis.set_major_formatter(formatter)      
    plt.ylim(y_min, y_max)
    plt.subplots_adjust(bottom=0.2, top=0.9)  # Adjust subplot margins to avoid cut-off    
    plt.savefig(f"entropy_plots/recreated_plots/recreated_entropy_vs_value_detected.png")
    # plt.show()

def recreate_CUSTOM_plot_entropy_vs_final_expected_value_detected(data):
    """
    This is for a custom plot. You can adjust anything needed in here.
    """
        
    x_labels = [bin_range for bin_range in data]
    y_values = [(3.8/2) * data[bin_range]["final_expected_value_detected"] for bin_range in data]
    
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
    plt.savefig(f"entropy_plots/recreated_plots/recreated_entropy_vs_value_detected.png")
    # plt.show()

def format_data_for_plot_recreations(data_path):
    with open(data_path, 'r') as file:
        raw_data = json.load(file)

    data = {}
    for bin_range, values in raw_data['bins_at_end'].items():
        data[bin_range] = {
            "final_fraction_value_detected": values["final_fraction_value_detected"],
            "final_expected_value_detected": values["final_expected_value_detected"]
        }
    
    return data

if __name__=="__main__":

    parser = argparse.ArgumentParser(description="Load configuration for entropy plot calculations.")
    parser.add_argument('-config_path', type=str, required=True, help='Path to the JSON configuration file.')
    # parser.add_argument('-prob_distributions_path', type=str, required=True, help='Path to the Pickle file of probability distributions')
    parser.add_argument('-data_path', type=str, required=True, help='Path to the output data of some create_entropy_plots.py run. This is the output data for which you want to recreate the plots.')
    args = parser.parse_args()
    data = format_data_for_plot_recreations(args.data_path)    
    item_vals, resource_sets, _, hiding_locations, NUM_HIDING_LOCATIONS, sizes_hiding_locations, detectors, budget = entropy_plot_input_variables.get_configuration(args.config_path)
    # prob_distributions_dict, NUM_SAMPLES_PER_BIN, NUM_BINS = entropy_plot_input_variables.get_prob_distributions_dict(args.prob_distributions_path)
    time_run_starts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    recreate_plot_entropy_vs_final_fraction_value_detected(data)
    # recreate_plot_entropy_vs_final_expected_value_detected(data) 
    recreate_CUSTOM_plot_entropy_vs_final_expected_value_detected(data)