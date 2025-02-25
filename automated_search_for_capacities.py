import random
from deap import base, creator, tools, algorithms
import argparse
from pprint import pprint as pprint
import json
# from entropy_plot_input_variables import resource_sets, hiding_locations, fraction_cargo_containers_storing_drugs, total_real_detectors, null_detectors_count, detector_accuracies, max_num_items_across_years
import entropy_plot_input_variables
from create_entropy_plot import calculate_expected_value_under_equilibrium_each_node



def compute_expected_fraction_detected_each_year(individual):
    """
    For each year, we calculate the expected fraction detected under the capacities being the variable individual from the genetic algorithm
    """
    expected_fraction_detected_each_year = {}
    for year, resource_set_dict in resource_sets.items():    
        expected_value_items_each_node_this_year, _ = calculate_expected_value_under_equilibrium_each_node(resource_set_dict, item_vals, capacities=individual)  
        total_value_items_this_year = sum(expected_value_items_each_node_this_year.values())
        expected_fraction_detected_each_year[year] = sum(detector_accuracies[i] * expected_value_items_each_node_this_year[i] for i in range(len(expected_value_items_each_node_this_year))) / total_value_items_this_year      
        
    return expected_fraction_detected_each_year

def diversity_of_expected_fraction_detected(individual):
    """
    This function returns the sum of pairwise differences among the expected_fraction_detected_each_year.
    """
    expected_fraction_detected_each_year_dict = compute_expected_fraction_detected_each_year(individual)
    expected_fraction_detected_each_year = list(expected_fraction_detected_each_year_dict.values())    # Extract all payoff values in a list (order doesn’t matter)
    
    total_diff = 0.0
    for i in range(len(expected_fraction_detected_each_year)):
        for j in range(i + 1, len(expected_fraction_detected_each_year)):
            total_diff += abs(expected_fraction_detected_each_year[i] - expected_fraction_detected_each_year[j])
    
    return (total_diff,)


# ---------------------------------------------
# Create custom "Individual" and "FitnessMax"
# ---------------------------------------------
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

def random_capacity_in_range(cap):
    """
    Return an integer capacity within ±tolerance_percent_deviation_real_capacity of the real capacity.
    """
    min_cap = int(cap * (1 - tolerance_percent_deviation_real_capacity/100))
    max_cap = int(cap * (1 + tolerance_percent_deviation_real_capacity/100))
    
    return random.randint(min_cap, max_cap)

def init_individual(icls):
    """
    Construct one individual with each location's capacity in ±tolerance_percent_deviation_real_capacity of real value.
    """
    capacities = [random_capacity_in_range(caps_normalized[i]) for i in range(NUM_LOCATIONS)]
    
    return icls(capacities)

def mutate_individual(individual, indpb=0.1):
    """
    With probability indpb, re-draw the capacity from ±tolerance_percent_deviation_real_capacity range of real capacity.
    """
    for i in range(len(individual)):
        if random.random() < indpb:
            individual[i] = random_capacity_in_range(caps_normalized[i])
    return (individual,)

def run_genetic_algorithm():
    toolbox.register("individual", init_individual, creator.Individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", diversity_of_expected_fraction_detected)  # Our objective function

    # Genetic operators
    toolbox.register("mate", tools.cxOnePoint)  # one-point crossover
    toolbox.register("mutate", mutate_individual, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)    
        
    pop = toolbox.population(100)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("max", max)

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.1, ngen=100, stats=stats, halloffame=hof, verbose=True)

    best_solution_found = hof[0]
    best_fitness = best_solution_found.fitness.values[0]

    return best_solution_found, best_fitness

def output_final_results(best_solution_found, best_fitness):
    print("\nBest solution found:", best_solution_found)
    best_solution_capacities_dict = {}
    for (loc_name, original_cap), ga_cap in zip(sorted_locs, best_solution_found):
        print(f"Location: {loc_name}") 
        print(f"Original real capacity: {original_cap} ")
        
        scaled_original_cap = original_cap * fraction_cargo_containers_storing_drugs        
        print(f"Original real capacity but scaled by fraction_cargo_containers_storing_drugs: {scaled_original_cap}")
        print(f"GA-chosen capacity: {ga_cap}")
        
        percent_diff = ((ga_cap - scaled_original_cap) / scaled_original_cap) * 100
        print(f"Percent difference: {percent_diff:.2f}%")
        print()
        
        best_solution_capacities_dict[loc_name] = ga_cap
        
    print("The dictionary of capacities for the best solution found:")
    pprint(best_solution_capacities_dict)
    
    expected_fraction_detected_each_year_dict_for_best_solution_found = compute_expected_fraction_detected_each_year(best_solution_found)
    print("expected_fraction_detected_each_year for the best individual:")
    for year, fraction in expected_fraction_detected_each_year_dict_for_best_solution_found.items():
        print(f"  {year}: {fraction:,.2f}")
    print(f"Best diversity (sum of pairwise payoff differences): {best_fitness:,.2f}")   
    
    output_data = {
        "best_solution_capacities": best_solution_capacities_dict,
        "detector_accuracies": detector_accuracies,
        "expected_fraction_detected_each_year": expected_fraction_detected_each_year_dict_for_best_solution_found
    }
    
    with open('output_automated_search_for_capacities.json', 'w') as json_file:
        json.dump(output_data, json_file, indent=4)        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load configuration for entropy plot calculations.")
    parser.add_argument('-config', type=str, required=True, help='Path to the JSON configuration file.')
    args = parser.parse_args()
    item_vals, resource_sets, num_resource_sets, hiding_locations, fraction_cargo_containers_storing_drugs, sizes_hiding_locations, detectors, detector_accuracies, NUM_SAMPLES_NEEDED_PER_BIN, NUM_BINS = entropy_plot_input_variables.get_configuration(args.config)
        
    tolerance_percent_deviation_real_capacity = 15  # This is the percent deviation in real capacity either up or down that cna be taken
    sorted_locs = sorted(hiding_locations.items(), key=lambda item: item[1], reverse=True)     
    caps_normalized = [cap * fraction_cargo_containers_storing_drugs for loc, cap in sorted_locs]    # We multiply each original real capacity by the fraction of TEUs that contain drugs. In effect, this gives us the number of cargo containers at each port which will store drugs.
    NUM_LOCATIONS = len(caps_normalized)
    
    best_solution_found, best_fitness = run_genetic_algorithm()
    output_final_results(best_solution_found, best_fitness)
    

    