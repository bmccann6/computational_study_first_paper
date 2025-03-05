import json 
import pickle
import re
from pprint import pprint as pprint

def get_configuration(config_path):
    with open(config_path, 'r') as file:
        config_data = json.load(file)
    
    item_vals = config_data["item_vals"]
    resource_sets = config_data["resource_sets"]
    NUM_RESOURCE_SETS = len(resource_sets)
    hiding_locations = config_data["hiding_locations"]
    NUM_HIDING_LOCATIONS = len(hiding_locations)
    
    fraction_cargo_containers_inspected = config_data['fraction_cargo_containers_inspected']
    detectors = config_data["detectors"]
    for detector in detectors.values():
        detector['accuracy'] *= fraction_cargo_containers_inspected     # Having the detector accuracies be multiplied by fraction_cargo_containers_inspected is equivalent to only inspecting fraction_cargo_containers_inspected of containers at a port by any given detector.
            
    budget = config_data["budget"]
    fraction_containers_from_drug_regions = config_data['fraction_containers_from_drug_regions']
    accessible_fraction_containers_from_drug_regions = config_data['accessible_fraction_containers_from_drug_regions']

    # num_items_per_year = {year: sum(resources.values()) for year, resources in resource_sets.items()}
    # max_num_items_across_years = max(num_items_per_year.values())
    # total_original_capacity = sum(hiding_locations.values())
    # fraction_cargo_containers_inbound_to_US_storing_drugs = max_num_items_across_years/total_original_capacity   # We will multiply all the capacities in hiding_locations by this value in other modules. It is useful because we will never need to have the total capacity be greater than the max capacity needed across all years. Also, by having a fraction we will multiply all capacities by later in other modules, we make the changes to capacities more uniform across locations.
    # fraction_cargo_containers_inbound_to_US_storing_drugs = 0.035
    # fraction_cargo_containers_inbound_to_US_storing_drugs = (10/3)/100
    fraction_cargo_containers_inbound_to_US_storing_drugs = fraction_containers_from_drug_regions * accessible_fraction_containers_from_drug_regions    
    
    #NOTE: Once we have found good values for the capacities, then change sizes_hiding_locations here so that in create_entropy_plots, we use the capacities we found.
    # sizes_hiding_locations = [int(size * fraction_cargo_containers_inbound_to_US_storing_drugs) for size in sorted(hiding_locations.values(), reverse=True)]        # At each index i, the size of hiding location i is the value.
    # sizes_hiding_locations = {year: [int(size * num_items_per_year[year]/total_original_capacity) for size in sorted(hiding_locations.values(), reverse=True)]  for year in resource_sets.keys()}       # At each index i of a year, the size of hiding location i is the value.
        
    # sizes_hiding_locations = {year: [int(size * fraction_cargo_containers_inbound_to_US_storing_drugs) for size in sorted(hiding_locations.values(), reverse=True)] for year in resource_sets.keys()}
    sizes_hiding_locations = [int(size * fraction_cargo_containers_inbound_to_US_storing_drugs) for size in sorted(hiding_locations.values(), reverse=True)]

    return item_vals, resource_sets, NUM_RESOURCE_SETS, hiding_locations, NUM_HIDING_LOCATIONS, sizes_hiding_locations, \
            detectors, budget

def get_prob_distributions_dict(prob_distributions_path):
    with open(prob_distributions_path, 'rb') as file:
        prob_distributions_dict = pickle.load(file)      

    NUM_SAMPLES_PER_BIN = int(re.search(r'NUM_SAMPLES_PER_BIN_(\d+)_and', prob_distributions_path).group(1))
    NUM_BINS = int(re.search(r'NUM_BINS_(\d+)\.pkl', prob_distributions_path).group(1))

    print(f"This is NUM_SAMPLES_PER_BIN: {NUM_SAMPLES_PER_BIN}")
    print(f"This is NUM_BINS: {NUM_BINS}")

    return prob_distributions_dict, NUM_SAMPLES_PER_BIN, NUM_BINS

