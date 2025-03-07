"""
This module employs a genetic algorithm to optimize detector/detector capacities across multiple locations,
with the goal of maximizing the diversity in the expected fraction of value detected over different resource sets.
We don't solve a budget MIP, instead we just have a fixed set of detectors which is used no matter what the resource set is. 
This is fine because we are just interested in choosing capacities which lead to a large diversity in the expected fraction of 
value detected over different resource sets. We want this because, for instance, if we can see that for our fixed set of detectors and some 
capacities found by this genetic algorithm, the expected fraction detected in resource_set_i is higher than the expected value 
detected in resource_set_j, then that implies that more of the value in resource_set_j is spread among the further lower capacity nodes as compared to resource_set_i.
That implies it would probably be better to buy more cheaper (but less accurate) detectors than fewer (but more accurate) detectors, so that we
can have our detectors reach those further nodes (when they are deterministically lined up) to give some more detection at those further nodes
where more of the total value is spread to.
In other words, that suggests the optimal set of detectors for resource_set_i would probably be noticeably different than for resource_set_j.
That in turn suggests, since the optimal set of detectors found by solving budget problems for resource_set_i and resource_set_j would be different,
the expected fraction detected for resource sets resource_set_i and resource_set_j (under their respective optimal set of detectors), would be noticeably different 
since we are using different sets of detectors.
Thus, having more information (i.e. less entropy) would be noticeably better.
And this difference in the expected fractions detected for resource sets resource_set_i and resource_set_j (and thus the improvement we can achieve by having less entropy) should 
increase the larger the fitness function value is. That is why we search for a solution which maximizes this fitness function value.

Key Functionalities:
- Compute the expected fraction detected for each resource_set based on a given capacity configuration.
- Measure the diversity as the sum of pairwise differences of the fractions detected for the resource sets. This is our fitness metric. 
- Generate candidate capacity configurations using randomized initialization within a tolerance range of the real capacities.
- Evolve these configurations using genetic operators (crossover, mutation, and tournament selection) provided by the deap library.
- Compare the GA-derived capacities with scaled original capacities and output the results to both the console and a JSON file.

Execution Flow:
1. Load configuration data using setup_data.
2. Initialize a population of candidate capacity configurations.
3. Run the genetic algorithm to maximize the diversity in the expected fraction detected across different resource sets for a fixed set of dummy detectors.
4. Print detailed comparisons between the original scaled capacities and the GA-derived capacities,
   and also print the expected fractions detected for the resource sets.
5. Save the final results to an output JSON file.
"""


import random
from deap import base, creator, tools, algorithms
import argparse
from pprint import pprint as pprint
import json
import setup_data
from create_entropy_plots import calculate_expected_value_under_equilibrium_each_node




def compute_expected_fraction_detected_each_resource_set(individual):
    """
    For each resource set, we calculate the expected fraction detected under the capacities being the variable individual from the genetic algorithm.
    It is cleaner to create a new function to calculate the payoffs here rather than just importing some functionality from create_entropy_plots.py. 
    This is because the function in create_entropy_plots.py to calculate the payoff is structured very differently because we are iterating over probability distributions there.
    """
    expected_fraction_detected_each_resource_set = {}
    for resource_set_name, resource_set in resource_sets.items():    
        expected_value_items_each_node_this_resource_set, _ = calculate_expected_value_under_equilibrium_each_node(resource_set, item_vals, capacities=individual)  
        total_value_items_this_resource_set = sum(expected_value_items_each_node_this_resource_set.values())
        expected_fraction_detected_each_resource_set[resource_set_name] = sum(dummy_detector_accuracies[i] * expected_value_items_each_node_this_resource_set[i] for i in range(len(expected_value_items_each_node_this_resource_set))) / total_value_items_this_resource_set      
        
    return expected_fraction_detected_each_resource_set

def diversity_of_expected_fraction_detected(individual):
    """
    This function is used to evaluate the diversity of the expected fraction detected, which is how we will measure the fitness of an individual.
    This function returns the sum of pairwise differences among the expected_fraction_detected_each_resource_set.
    """
    expected_fraction_detected_each_resource_set_dict = compute_expected_fraction_detected_each_resource_set(individual)
    expected_fraction_detected_each_resource_set = list(expected_fraction_detected_each_resource_set_dict.values())    # Extract all payoff values in a list (order doesn’t matter)
    
    total_diff = 0.0
    for i in range(len(expected_fraction_detected_each_resource_set)):
        for j in range(i + 1, len(expected_fraction_detected_each_resource_set)):
            total_diff += abs(expected_fraction_detected_each_resource_set[i] - expected_fraction_detected_each_resource_set[j])
    
    return (total_diff,)     # The deap module requires the fitness values to be provided as tuples.


def random_capacity_in_range(cap):
    """
    Return an integer capacity within ±tolerance_percent_deviation_real_capacity of a real capacity (cap).
    This is used in the mutate_individual
    """
    min_cap = int(cap * (1 - tolerance_percent_deviation_real_capacity/100))
    max_cap = int(cap * (1 + tolerance_percent_deviation_real_capacity/100))
    
    return random.randint(min_cap, max_cap)

def init_individual(icls):
    """
    Construct one individual with each location's capacity within ±tolerance_percent_deviation_real_capacity of its real value.
    """
    capacities = [random_capacity_in_range(sizes_hiding_locations[i]) for i in range(NUM_HIDING_LOCATIONS)]
    
    return icls(capacities)

def mutate_individual(individual, indpb=0.1):
    """
    With probability indpb, re-draw the capacity from ±tolerance_percent_deviation_real_capacity range of real capacity.
    """
    for i in range(len(individual)):
        if random.random() < indpb:
            individual[i] = random_capacity_in_range(sizes_hiding_locations[i])
    return (individual,)

def run_genetic_algorithm():
    """
    Runs a genetic algorithm to optimize the given fitness function.
    The function initializes the genetic algorithm, registers the necessary
    genetic operators, and executes the algorithm for a specified number of
    generations. It returns the best solution found and its corresponding fitness value.
    Returns:
        best_solution_found (Individual): The best individual found by the genetic algorithm.
        best_fitness (float): The fitness value of the best individual.
    Notes:
        - The population size is set to 100.
        - The algorithm runs for 100 generations.
        - The crossover probability (cxpb) is set to 0.7.
        - The mutation probability (mutpb) is set to 0.1.
        - The tournament size for selection is set to 3.
        - The Hall of Fame (hof) keeps track of the best individual found.
        - The statistics object (stats) records the maximum fitness value in each generation.
    """
    toolbox.register("individual", init_individual, creator.Individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", diversity_of_expected_fraction_detected)  # Our fitness function

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
        
        scaled_original_cap = original_cap * fraction_cargo_containers_storing_drugs_max_resource_set        
        print(f"Original real capacity but scaled by fraction_cargo_containers_storing_drugs_max_resource_set: {scaled_original_cap}")
        print(f"GA-chosen capacity: {ga_cap}")
        
        percent_diff = ((ga_cap - scaled_original_cap) / scaled_original_cap) * 100
        print(f"Percent difference: {percent_diff:.2f}%")
        print()
        
        best_solution_capacities_dict[loc_name] = ga_cap
        
    print("The dictionary of capacities for the best solution found:")
    pprint(best_solution_capacities_dict)
    
    expected_fraction_detected_each_resource_set_dict_for_best_solution_found = compute_expected_fraction_detected_each_resource_set(best_solution_found)
    print("expected_fraction_detected_each_resource_set for the best individual:")
    for resource_set_name, fraction in expected_fraction_detected_each_resource_set_dict_for_best_solution_found.items():
        print(f"  {resource_set_name}: {fraction:,.2f}")
    print(f"Best diversity (sum of pairwise payoff differences): {best_fitness:,.2f}")   
    
    output_data = {
        "best_solution_capacities": best_solution_capacities_dict,
        "dummy_detector_accuracies": dummy_detector_accuracies,
        "expected_fraction_detected_each_resource_set": expected_fraction_detected_each_resource_set_dict_for_best_solution_found
    }
    
    with open('output_automated_search_for_capacities.json', 'w') as json_file:
        json.dump(output_data, json_file, indent=4)        
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load configuration for entropy plot calculations.")
    parser.add_argument('-config_path', type=str, required=True, help='Path to the JSON configuration file.')
    args = parser.parse_args()
    item_vals, resource_sets, NUM_RESOURCE_SETS, hiding_locations, NUM_HIDING_LOCATIONS, sizes_hiding_locations, _, _, = setup_data.get_configuration(args.config_path)

    dummy_detectors= [0.868, 0.82, 0.788, 0.67, 0.8, 0.4, 0.4, 0.5, 0.5]   # See the module-level docstring for why we can just use a set of dummy detectors.
    dummy_detector_accuracies = [0.868, 0.82, 0.788, 0.67, 0.8, 0.4, 0.4, 0.5, 0.5] + [0] * (NUM_HIDING_LOCATIONS - len(dummy_detectors))
    tolerance_percent_deviation_real_capacity = 15  # This is the percent deviation in real capacity either up or down that can be taken. The smaller tolerance_percent_deviation_real_capacity is, the closer the capacities found by the genetic algorithm will be to their original capacities. However, the smaller tolerance_percent_deviation_real_capacity is, the less diversity in the expected fraction detect could occur between resource sets for our given fixed set of detectors.
    sorted_locs = sorted(hiding_locations.items(), key=lambda item: item[1], reverse=True)      # Sort the hiding locations in descending order, from largest capacity, to smallest capacity.
    
    # Chatgpt said we should put this next part of the code in if __name__ == "__main__" because then it makes it globally available. And the deap creator registers custom classes globally. If you put these inside run_genetic_algorithm and call it multiple times, you'll attempt to re-register the same types, which can lead to errors.
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()
    
    best_solution_found, best_fitness = run_genetic_algorithm()
    output_final_results(best_solution_found, best_fitness)
    

