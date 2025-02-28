import json 

def get_configuration(config_path):

    with open(config_path, 'r') as file:
        config_data = json.load(file)
    
    item_vals = config_data["item_vals"]
    resource_sets = config_data["resource_sets"]
    num_resource_sets = len(resource_sets)
    hiding_locations = config_data["hiding_locations"]
    NUM_HIDING_LOCATIONS = len(hiding_locations)
    detectors = config_data["detectors"]
    budget = config_data["budget"]

    num_items_per_year = {year: sum(resources.values()) for year, resources in resource_sets.items()}
    max_num_items_across_years = max(num_items_per_year.values())
    total_original_capacity = sum(hiding_locations.values())
    # fraction_cargo_containers_storing_drugs = max_num_items_across_years/total_original_capacity   # We will multiply all the capacities in hiding_locations by this value in other modules. It is useful because we will never need to have the total capacity be greater than the max capacity needed across all years. Also, by having a fraction we will multiply all capacities by later in other modules, we make the changes to capacities more uniform across locations.
    # print(f"This is fraction_cargo_containers_storing_drugs: {fraction_cargo_containers_storing_drugs}")

    #NOTE: Once we have found good values for the capacities, then change sizes_hiding_locations here so that in create_entropy_plots, we use the capacities we found.
    # sizes_hiding_locations = [int(size * fraction_cargo_containers_storing_drugs) for size in sorted(hiding_locations.values(), reverse=True)]        # At each index i, the size of hiding location i is the value.
    sizes_hiding_locations_each_year = {year: [int(size * num_items_per_year[year]/total_original_capacity) for size in sorted(hiding_locations.values(), reverse=True)]  for year in resource_sets.keys()}       # At each index i of a year, the size of hiding location i is the value.

    NUM_SAMPLES_NEEDED_PER_BIN = 10
    NUM_BINS = 20

    return item_vals, resource_sets, num_resource_sets, hiding_locations, NUM_HIDING_LOCATIONS, sizes_hiding_locations_each_year, \
            detectors, budget, NUM_SAMPLES_NEEDED_PER_BIN, NUM_BINS

# Comment out necessary things in order to do testing. Dont use real-world data for testing.