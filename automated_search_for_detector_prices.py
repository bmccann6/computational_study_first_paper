from gurobipy import Model, GRB
import argparse
import setup_data
from create_entropy_plots import calculate_expected_value_under_equilibrium_each_node


# def calculate_expected_value_under_equilibrium_each_node():
#     """
#     For each year, we calculate the expected fraction detected under the capacities being the variable individual from the genetic algorithm
#     """
    # expected_fraction_detected_each_year = {}
    # for year, resource_set_dict in resource_sets.items():    
    #     expected_value_items_each_node_this_year, _ = calculate_expected_value_under_equilibrium_each_node(resource_set_dict, item_vals, capacities=individual)  
    #     total_value_items_this_year = sum(expected_value_items_each_node_this_year.values())
    #     expected_fraction_detected_each_year[year] = sum(detector_accuracies[i] * expected_value_items_each_node_this_year[i] for i in range(len(expected_value_items_each_node_this_year))) / total_value_items_this_year      
        
    # return expected_fraction_detected_each_year

    
# Assume necessary data structures A_w, lambda_t, C_t, and B are defined here
# Example data structure (You should define these based on your problem context)
A_w = {i: random_value_for_each_i}  # Expected values for each i
lambda_t = {t: random_value_for_each_t}  # lambda values for each t
C = {t: random_cost_value_for_each_t}  # Cost values for each t
B = total_budget  # Total available budget

# Create a new model
m = Model("integer_program")

# Define variables
x = m.addVars(T, I, vtype=GRB.BINARY, name="x")  # Binary decision variables

# Set objective
objective = quicksum(A_w[i] * (1 - lambda_t[t]) * x[t, i] for t in T for i in I)
m.setObjective(objective, GRB.MINIMIZE)

# Add constraints
# Budget constraint
m.addConstr(quicksum(C[t] * x[t, i] for t in T for i in I) <= B, "budget_constraint")

# Each item can be selected at most once across all t
for i in I:
    m.addConstr(quicksum(x[t, i] for t in T) <= 1, f"selection_constraint_{i}")

# Optimize the model
m.optimize()

# Print the solution
for v in m.getVars():
    if v.x > 0:
        print(f"{v.varName}, value: {v.x}")
   

def main():
    1. Get compute_expected_fraction_detected_each_year and lambda_t values as well. These are setups for later.
    2. Run the mip (which should be a function we call). Have the optimal value of the mip returned, as well as the price of each detector determined, and the number of sensors of each type purchased.
    3. 
  
What is the objective of the genetic algorithm? What is trying to do? 

\ici Just look at ici comment in overleaf doc first before trying to do this module
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load configuration for entropy plot calculations.")
    parser.add_argument('-config_path', type=str, required=True, help='Path to the JSON configuration file.')
    args = parser.parse_args()
    item_vals, resource_sets, NUM_RESOURCE_SETS, hiding_locations, fraction_cargo_containers_inbound_to_US_storing_drugs, sizes_hiding_locations_each_year, detector_accuracies, NUM_SAMPLES_PER_BIN, NUM_BINS = setup_data.get_configuration(args.config_path, args.prob_distributions_path)
        


	In this module, we will use the capacities we had outputted by automated_search_for_detector_prices.
	See last page of "random scrap" in iPad notes for details on how to go about doing this.
