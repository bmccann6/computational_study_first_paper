from gurobipy import Model, GRB, quicksum

def minimize_x_given_entropy(r, b):
    # Create a new model
    m = Model("entropy_minimization")

    # Add variables
    x = m.addVar(lb=1e-10, ub=1, vtype=GRB.CONTINUOUS, name="x")
    p = [m.addVar(lb=1e-10, ub=1, vtype=GRB.CONTINUOUS, name=f"p{i}") for i in range(2, r+1)]

    # Create log variables for x and each p_i
    log_x = m.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="log_x")
    log_p = [m.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name=f"log_p{i}") for i in range(2, r+1)]

    # Set the objective to minimize x
    m.setObjective(x, GRB.MINIMIZE)

    # Add constraint: sum of x and p_i equals 1
    m.addConstr(x + quicksum(p) == 1, "probability_sum")

    # Define entropy expression with the logarithmic variables
    entropy_expr = -(x * log_x + quicksum(pi * log_pi for pi, log_pi in zip(p, log_p)))
    m.addConstr(entropy_expr <= b, "entropy")

    # Link logarithmic constraints
    m.addGenConstrLog(x, log_x)  # Corrected: remove the third argument
    for pi, log_pi in zip(p, log_p):
        m.addGenConstrLog(pi, log_pi)  # Corrected: remove the third argument

    # Optimize the model
    m.optimize()

    if m.status == GRB.OPTIMAL:
        print(f"Optimal value of x: {x.X}")
        print("Optimal values of p_i:")
        for pi in p:
            print(f"{pi.varName}: {pi.X}")
    else:
        print("No optimal solution found.")

# Example usage
minimize_x_given_entropy(10, 0.1)


This code is not useful anymore.