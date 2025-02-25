from gurobipy import Model, GRB
from create_entropy_plot import calculate_expected_value_under_equilibrium_each_node


# def calculate_expected_value_under_equilibrium_each_node():
#     """
#     For each year, we calculate the expected fraction detected under the capacities being the variable individual from the genetic algorithm
#     """
#     expected_fraction_detected_each_year = {}
#     for year, resource_set_dict in resource_sets.items():    
#         backloading_stack_values = calculate_backloading_stack_values(resource_set_dict, item_vals, capacities=individual)    
#         breakpoints = calculate_breakpoints(backloading_stack_values, capacities=individual)
#         expected_value_items_each_node_this_year = calculate_expected_value_under_equilibrium_each_node(backloading_stack_values, breakpoints)       

#     return expected_value_items_each_node_this_year

    
# Assume necessary data structures A_w, lambda_t, C_t, and B are defined here
# Example data structure (You should define these based on your problem context)
A_w = {i: random_value_for_each_i}  # Expected values for each i
lambda_t = {t: random_value_for_each_t}  # lambda values for each t
C_t = {t: random_cost_value_for_each_t}  # Cost values for each t
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
m.addConstr(quicksum(C_t[t] * x[t, i] for t in T for i in I) <= B, "budget_constraint")

# Each item can be selected at most once across all t
for i in I:
    m.addConstr(quicksum(x[t, i] for t in T) <= 1, f"selection_constraint_{i}")

# Optimize the model
m.optimize()

# Print the solution
for v in m.getVars():
    if v.x > 0:
        print(f"{v.varName}, value: {v.x}")


	In this module, we will use the capacities we had outputted by automated_search_for_detector_prices.
	See last page of "random scrap" in iPad notes for details on how to go about doing this.
