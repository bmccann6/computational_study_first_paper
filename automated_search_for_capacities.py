import random
from deap import base, creator, tools, algorithms
from entropy_plot_input_variables import resource_sets, hiding_locations, fraction_cargo_containers_storing_drugs, total_real_detectors, null_detectors_count, detector_accuracies, max_num_items_across_years
from create_entropy_plot import calculate_backloading_stack_values, calculate_breakpoints, calculate_expected_value_under_equilibrium_each_node
from pprint import pprint as pprint



def compute_expected_fraction_detected(individual):
    """
    For each year, we calculate the expected fraction detected under the capacities being the variable individual from the genetic algorithm
    """
    expected_fraction_detected_each_year = {}
    for year, resource_set_dict in resource_sets.items():    
        backloading_stack_values = calculate_backloading_stack_values(resource_set_dict, capacities=individual)    
        breakpoints = calculate_breakpoints(backloading_stack_values, capacities=individual)
        expected_value_items_each_node_this_year = calculate_expected_value_under_equilibrium_each_node(backloading_stack_values, breakpoints)       
        total_value_items_this_year = sum(expected_value_items_each_node_this_year.values())
        expected_fraction_detected_each_year[year] = sum(detector_accuracies[i] * expected_value_items_each_node_this_year[i] for i in range(len(expected_value_items_each_node_this_year))) / total_value_items_this_year      
        
    return expected_fraction_detected_each_year

def diversity_of_expected_fraction_detected(individual):
    """
    This function returns the sum of pairwise differences among the expected_fraction_detected_each_year.
    """
    expected_fraction_detected_each_year_dict = compute_expected_fraction_detected(individual)
    expected_fraction_detected_each_year = list(expected_fraction_detected_each_year_dict.values())    # Extract all payoff values in a list (order doesn’t matter)
    
    total_diff = 0.0
    for i in range(len(expected_fraction_detected_each_year)):
        for j in range(i + 1, len(expected_fraction_detected_each_year)):
            total_diff += abs(expected_fraction_detected_each_year[i] - expected_fraction_detected_each_year[j])
    
    return (total_diff,)


# \ici Create another module called automated_search_for_detector_prices which does a genetic algorithm.
# And in this module, we will use the capacities we had outputted by automated_search_for_detector_prices.


# \ici I think perhaps an issue with the entropy histogram is that its an average. So like in say low entropy states, we will 
# basically have each of the distributions selected at random for one run. And then we average over all those runs, and we 
# get something that just looks like the average of the expected fractions detected for each year. 
# And also that the defender does the same strategy regardless of the year.


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


# \ici Is the reason we keep seeing larger deviations than 15% is because we do the 15% before applying the scaling factor?
# I thought whether we did 15% before or after applying the scaling factor would not matter, but maybe it does.

def init_individual(icls):
    """
    Construct one individual with each location's capacity in ±tolerance_percent_deviation_real_capacity of real value.
    """
    print(f"This is the total number of items originally: {sum(caps_normalized)}")  
    capacities = [random_capacity_in_range(caps_normalized[i]) for i in range(NUM_LOCATIONS)]
    
    # scaling_factor = sum(capacities) / max_num_items_across_years
    # capacities = [cap / scaling_factor for cap in capacities]
    print(f"This is the sum of the capacities once the deviations are done and the scaling factor applied: {sum(capacities)}")
    
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

def print_final_results(best_solution_found, best_fitness):
    print("\nBest solution found:", best_solution_found)
    for (loc_name, original_cap), ga_cap in zip(sorted_locs, best_solution_found):
        print(f"Location: {loc_name}") 
        # scaling_factor = sum(best_solution_found) / max_num_items_across_years
        # cap = cap / scaling_factor
        print(f"Original real capacity: {original_cap} ")
        scaled_original_cap = original_cap * fraction_cargo_containers_storing_drugs        
        print(f"Original real capacity but scaled by fraction_cargo_containers_storing_drugs: {scaled_original_cap}")
        print(f"GA-chosen capacity: {ga_cap}")
        percent_diff = ((ga_cap - scaled_original_cap) / scaled_original_cap) * 100
        print(f"Percent difference: {percent_diff:.2f}%")
        print()
        
    expected_fraction_detected_each_year_dict_for_best_solution_found = compute_expected_fraction_detected(best_solution_found)
    print("expected_fraction_detected_each_year for the best individual:")
    for year, fraction in expected_fraction_detected_each_year_dict_for_best_solution_found.items():
        print(f"  {year}: {fraction:,.2f}")
    print(f"Best diversity (sum of pairwise payoff differences): {best_fitness:,.2f}")   
        
if __name__ == "__main__":
    
    tolerance_percent_deviation_real_capacity = 15  # This is the percent deviation in real capacity either up or down that cna be taken
    sorted_locs = sorted(hiding_locations.items(), key=lambda item: item[1], reverse=True)
    # sorted_locs = [(loc, cap * fraction_cargo_containers_storing_drugs) for loc, cap in sorted_locs]
    caps_normalized = [cap * fraction_cargo_containers_storing_drugs for loc, cap in sorted_locs]    
    
    NUM_LOCATIONS = len(caps_normalized)
    
    best_solution_found, best_fitness = run_genetic_algorithm()
    print_final_results(best_solution_found, best_fitness)
    

    