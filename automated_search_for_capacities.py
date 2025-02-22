import random
from deap import base, creator, tools, algorithms
from entropy_plot_input_variables import resource_sets, hiding_locations, fraction_cargo_containers_storing_drugs, total_real_detectors, null_detectors_count, detector_accuracies, max_num_items_across_years
from create_entropy_plot import calculate_backloading_stack_values, calculate_breakpoints, calculate_expected_value_under_equilibrium_each_node
from pprint import pprint as pprint



def compute_equilibrium_payoffs(individual):
    """
    For each year, we calculate the equilibrium payoff under the capacities being the variable individual from the genetic algorithm
    """
    payoffs = {}
    for year, resource_set_dict in resource_sets.items():    
        backloading_stack_values = calculate_backloading_stack_values(resource_set_dict, capacities=individual)    
        breakpoints = calculate_breakpoints(backloading_stack_values, capacities=individual)
        expected_value_items_each_node_this_year = calculate_expected_value_under_equilibrium_each_node(backloading_stack_values, breakpoints)       
        payoffs[year] = sum(detector_accuracies[i] * expected_value_items_each_node_this_year[i] for i in range(len(expected_value_items_each_node_this_year)))       
        
    return payoffs

def diversity_of_payoffs(individual):
    """
    This function returns the sum of pairwise differences among the 5 payoffs.
    """
    payoffs_dict = compute_equilibrium_payoffs(individual)
    payoffs = list(payoffs_dict.values())    # Extract all payoff values in a list (order doesn’t matter)
    
    total_diff = 0.0
    for i in range(len(payoffs)):
        for j in range(i + 1, len(payoffs)):
            total_diff += abs(payoffs[i] - payoffs[j])
    
    return (total_diff,)


# ---------------------------------------------
# Create custom "Individual" and "FitnessMax"
# ---------------------------------------------
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

def random_capacity_in_range(real_cap):
    """
    Return an integer capacity within ±tolerance_percent_deviation_real_capacity of the real capacity.
    """
    min_cap = int(real_cap * (1 - tolerance_percent_deviation_real_capacity/100))
    max_cap = int(real_cap * (1 + tolerance_percent_deviation_real_capacity/100))
    return random.randint(min_cap, max_cap)

def init_individual(icls):
    """
    Construct one individual with each location's capacity in ±tolerance_percent_deviation_real_capacity of real value.
    """
    capacities = [random_capacity_in_range(real_caps[i]) for i in range(NUM_LOCATIONS)]
    return icls(capacities)

def mutate_individual(individual, indpb=0.1):
    """
    With probability indpb, re-draw the capacity from ±tolerance_percent_deviation_real_capacity range of real capacity.
    """
    for i in range(len(individual)):
        if random.random() < indpb:
            individual[i] = random_capacity_in_range(real_caps[i])
    return (individual,)

toolbox.register("individual", init_individual, creator.Individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", diversity_of_payoffs)  # Our objective function

# Genetic operators
toolbox.register("mate", tools.cxOnePoint)  # one-point crossover
toolbox.register("mutate", mutate_individual, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)

def run_genetic_algorithm():
    pop = toolbox.population(50)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("max", max)

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.1, ngen=100, stats=stats, halloffame=hof, verbose=True)

    best_solution_found = hof[0]
    best_fitness = best_solution_found.fitness.values[0]

    return best_solution_found, best_fitness

def print_final_results(best_solution_found, best_fitness):
    print("\nBest solution found:", best_solution_found)
    for (loc_name, cap), ga_cap in zip(sorted_locs, best_solution_found):
        print(f"Location: {loc_name}")
        print(f"Real capacity: {cap}")
        print(f"GA-chosen capacity: {ga_cap}")
        percent_diff = ((ga_cap - cap) / cap) * 100
        print(f"Percent difference: {percent_diff:.2f}%")
        print()
        
    payoffs_dict_for_best_solution_found = compute_equilibrium_payoffs(best_solution_found)
    print("Payoffs for the best individual:")
    for year, payoff_value in payoffs_dict_for_best_solution_found.items():
        print(f"  {year}: {payoff_value:,.2f}")
    print(f"Best diversity (sum of pairwise payoff differences): {best_fitness:,.2f}")   
        
if __name__ == "__main__":
    
    
    \ici I think we need something as well which ensures that for a given year, the total capacity is >= the number of items once we do the tolerance_percent_deviation_real_capacity.
    But isnt the total capacity supposed to be the same regardless of the year? Yeah.
    So how do we resolve this? Because we cant have the total capacity be such that for some year, the number of drugs is greater than the total capacity.
    
    tolerance_percent_deviation_real_capacity = 15  # This is the percent deviation in real capacity either up or down that cna be taken
    sorted_locs = sorted(hiding_locations.items(), key=lambda item: item[1], reverse=True)
    sorted_locs = [(loc, cap * fraction_cargo_containers_storing_drugs) for loc, cap in sorted_locs]
    # real_caps = [cap for loc, cap in sorted_locs]
    real_caps = [cap for loc, cap in sorted_locs]
    #\ici The above version of real_caps worked well. Maybe instead of 0.01, just pick a value like 0.05. That way we still greatly limit the search space from likely impractical solutions, but still cut down on state space.
    # max_cap = get_max_total_capacity_across_all_resource_sets(resource_sets)    # It wouldn't make sense to have any location capacities be greater than this, since this is the maximum number of resources in any year.
    
    # print(f"This is original real_caps: {[cap for loc, cap in sorted_locs]}")
    # real_caps = [min(cap, max_cap) for loc, cap in sorted_locs]
    # print(f"This is real_caps modified: {real_caps}")
    # #\ici clean up all the above code.
    
    NUM_LOCATIONS = len(real_caps)
    
    best_solution_found, best_fitness = run_genetic_algorithm()
    print_final_results(best_solution_found, best_fitness)
    

    