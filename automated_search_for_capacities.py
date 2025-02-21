import random
from deap import base, creator, tools, algorithms
from entropy_plot_input_variables import hiding_locations, total_real_detectors, null_detectors_count, detector_accuracies
from create_entropy_plot import calculate_backloading_stack_values, calculate_breakpoints, calculate_expected_value_under_equilibrium_each_node


def compute_equilibrium_payoffs(capacities):
    """
    Placeholder: returns random payoffs for demonstration.
    Replace this with your real solver that returns [p1, p2, p3, p4, p5].
    """
    return [random.uniform(0, 100) for _ in range(5)]

def diversity_of_payoffs(individual):
    """
    Objective: Sum of pairwise differences among the 5 payoffs.
    """
    payoffs = compute_equilibrium_payoffs(individual)
    total_diff = 0
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
    Return an integer capacity within ±15% of the real capacity.
    """
    min_cap = int(real_cap * 0.85)
    max_cap = int(real_cap * 1.15)
    return random.randint(min_cap, max_cap)

def init_individual(icls):
    """
    Construct one individual with each location's capacity in ±15% of real value.
    """
    capacities = [
        random_capacity_in_range(real_caps[i]) for i in range(NUM_LOCATIONS)
    ]
    return icls(capacities)

def mutate_individual(individual, indpb=0.1):
    """
    With probability indpb, re-draw the capacity from ±15% range of real capacity.
    """
    for i in range(len(individual)):
        if random.random() < indpb:
            individual[i] = random_capacity_in_range(real_caps[i])
    return (individual,)

toolbox.register("individual", init_individual, creator.Individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Our objective function
toolbox.register("evaluate", diversity_of_payoffs)

# Genetic operators
toolbox.register("mate", tools.cxOnePoint)  # one-point crossover
toolbox.register("mutate", mutate_individual, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)

def main():
    pop = toolbox.population(n=50)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("max", max)

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.1, ngen=20, stats=stats, halloffame=hof, verbose=True)

    best_individual = hof[0]
    best_fitness = best_individual.fitness.values[0]
    print("\nBest individual found:", best_individual)
    print("Best diversity (sum of pairwise payoff differences):", best_fitness)

    return best_individual, best_fitness

if __name__ == "__main__":
    sorted_locs = sorted(hiding_locations.items(), key=lambda item: item[1], reverse=True)
    locations = [loc for loc, cap in sorted_locs]
    real_caps = [cap for loc, cap in sorted_locs]
    NUM_LOCATIONS = len(locations)
    
    main()