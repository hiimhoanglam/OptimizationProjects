from gurobipy import Model, GRB, quicksum, multidict
import numpy as np
# Define the data
# Number of customers and facilities
num_customers = 100
num_facilities = 100

# Set seed for reproducibility
np.random.seed(0)

# Generate random demands for customers
I, d = multidict({i: np.random.randint(50, 150) for i in range(1, num_customers + 1)})

# Generate random capacities and fixed costs for facilities
capacities = {j: np.random.randint(300, 700) for j in range(1, num_facilities + 1)}
fixed_costs = {j: np.random.randint(800, 1200) for j in range(1, num_facilities + 1)}

# Generate facilities with capacity and fixed cost combined
J = list(capacities.keys())
M = capacities
f = fixed_costs

# Generate random transportation costs
c = {(i, j): np.random.randint(1, 10) for i in I for j in J}

# I, d = multidict({1: 80, 2: 270, 3: 250, 4: 160, 5: 180})
# J, M, f = multidict({1: [500, 1000], 2: [500, 1000], 3: [500, 1000]})
# c = {(1, 1): 4, (1, 2): 6, (1, 3): 9,
#      (2, 1): 5, (2, 2): 4, (2, 3): 7,
#      (3, 1): 6, (3, 2): 3, (3, 3): 4,
#      (4, 1): 8, (4, 2): 5, (4, 3): 3,
#      (5, 1): 10, (5, 2): 8, (5, 3): 4,
#      }




# Define the model
def flp(I, J, d, M, f, c):
    model = Model("flp")
    x, y = {}, {}

    # Create variables
    for j in J:
        y[j] = model.addVar(vtype=GRB.BINARY, name="y(%s)" % j)
        for i in I:
            x[i, j] = model.addVar(vtype=GRB.CONTINUOUS, name="x(%s,%s)" % (i, j))

    # Update model to integrate new variables
    model.update()

    # Add demand constraints
    for i in I:
        model.addConstr(quicksum(x[i, j] for j in J) == d[i], name="Demand(%s)" % i)

    # Add capacity constraints
    for j in J:
        model.addConstr(quicksum(x[i, j] for i in I) <= M[j] * y[j], name="Capacity(%s)" % j)

    # Add strong constraints
    for (i, j) in x:
        model.addConstr(x[i, j] <= d[i] * y[j], name="Strong(%s,%s)" % (i, j))

    # Set objective
    model.setObjective(
        quicksum(f[j] * y[j] for j in J) +
        quicksum(c[i, j] * x[i, j] for i in I for j in J),
        GRB.MINIMIZE
    )

    # Store data for further use
    model._x = x
    model._y = y

    return model


# Create the model
model = flp(I, J, d, M, f, c)

# Optimize the model
model.optimize()

# Print the results
if model.status == GRB.Status.OPTIMAL:
    print("\nOptimal value:", model.objVal)
    print("\nFacility open decisions:")
    for j in J:
        if model._y[j].x > 0.5:
            print("Facility %d is open" % j)
    print("\nTransport plan:")
    for i in I:
        for j in J:
            if model._x[i, j].x > 1e-6:
                print("Send %g units from Facility %d to Customer %d" % (model._x[i, j].x, j, i))
else:
    print("No optimal solution found.")