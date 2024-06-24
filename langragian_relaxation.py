#import time
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import matplotlib.pyplot as plt

d = [10, 6, 20, 32, 15, 28, 3, 19, 8, 13]

# Define the cost matrix c
c = np.array([
    [10, 7, 11, 12, 32, 15, 20, 26, 4, 41],
    [13, 17, 31, 37, 21, 5, 13, 15, 14, 12],
    [4, 13, 14, 22, 8, 31, 26, 11, 12, 23],
    [21, 21, 13, 18, 9, 27, 11, 16, 26, 32],
    [32, 18, 11, 14, 11, 11, 16, 32, 34, 8],
    [15, 9, 13, 12, 14, 15, 32, 8, 12, 9],
    [28, 32, 15, 2, 17, 12, 9, 6, 11, 6]
])

# Determine the number of locations and customers from the c matrix
num_locations = c.shape[0]
num_customers = c.shape[1]

# Define locations and customers based on the dimensions of c
locations = list(range(num_locations))
customers = list(range(num_customers))

# Number of facilities to open
p = 3
# np.random.seed(0)
# def generate_data(num_locations, num_customers):
#     # Generate random demand vector
#     d = np.random.randint(1, 50, size=num_customers)
#
#     # Generate random cost matrix
#     c = np.random.randint(1, 50, size=(num_locations, num_customers))
#
#     return d, c
#
#
# # Customize number of facilities and customers
# num_locations = 100
# num_customers = 100
#
# # Generate data
# d, c = generate_data(num_locations, num_customers)
#
# # Define locations and customers based on the dimensions of c
# locations = list(range(num_locations))
# customers = list(range(num_customers))
#
# # Number of facilities to open
# p = 50

def optimal(p, locations, customers, c, d):
    # Create a new model
    m = gp.Model("uncapacitated_facility_location")
    num_locations = len(locations)
    num_customers = len(customers)

    # Create variables
    x = m.addVars(num_locations, num_customers, vtype=GRB.CONTINUOUS, name="x")
    y = m.addVars(num_locations, vtype=GRB.BINARY, name="y")

    # Set objective
    m.setObjective(gp.quicksum(d[j] * c[i][j] * x[i, j] for i in range(num_locations) for j in range(num_customers)),
                   GRB.MINIMIZE)

    # Add constraints
    # Each customer must be assigned to exactly one location
    for j in range(num_customers):
        m.addConstr(gp.quicksum(x[i, j] for i in range(num_locations)) == 1, name=f"customer_{j}")

    # The number of locations to open must be exactly p
    m.addConstr(gp.quicksum(y[i] for i in range(num_locations)) == p, name="num_locations")

    # Assignment constraint
    for i in range(num_locations):
        for j in range(num_customers):
            m.addConstr(x[i, j] <= y[i], name=f"assignment_{i}_{j}")

    m.optimize()
    # Retrieve the optimal objective value and the optimal solution
    Z_opt = m.objVal
    x_opt = m.getAttr('x', x)
    y_opt = m.getAttr('x', y)

    return Z_opt, x_opt, y_opt

def lower_bound(lambda_vector):
    #Compute vector v
    v = [0] * num_locations
    for i in range(num_locations):
        for j in range(num_customers):
            v[i] = v[i] + min(0, d[j] * c[i, j] - lambda_vector[j])
    #Sort v from the most negative to zero
    idx = np.argsort(v)
    #Determine y
    y = np.zeros(num_locations)
    y[idx[0:p]] = 1
    #Determine x
    x = np.zeros((num_locations, num_customers))
    for i in range(num_locations):
        for j in range(num_customers):
            if (y[i] == 1) and (d[j] * c[i,j] - lambda_vector[j] < 0):
                x[i,j] = 1
    #Compute the Z_D(lambda^k): The lagrangian relaxation objective value
    Z_D = 0
    for j in range(num_customers):
        Z_D = Z_D + lambda_vector[j] # + sigma lambda_j
        for i in range(num_locations):
            Z_D = Z_D + d[j] * c[i,j] * x[i,j] - lambda_vector[j] * x[i,j]
    return Z_D, x, y

def upper_bound(y):
    #By solving the Lagrangian problem, we have the location variable y and the assignment variable x on our hands
    #However, while location variable y will satisfy all the original constraints
    #Variable x will likely violate the original constraint that is relaxed. To find a feasible solution and obtain
    #an upperbound, we fix y at the solution of the Lagrangian problem, then assign each demand node to the nearest facility

    #Compute x, given y
    #Fixing y means keeping the facility selection unchanged based on the Lagrangian relaxation results.
    x = np.zeros((num_locations, num_customers))
    for j in range(num_customers):
        #Iterate through each customer j, find the facility with the smallest transportation cost to that facility
        #However, we only need to consider the facility that is open: (1-y) * np.max(c) means that: if a facility is not open
        #that facility will not be considered by setting the transportation to the maximum value
        index = np.argmin(c[:,j] + (1 - y) * np.max(c))
        #The facility with the smallest transportation to the customer j will be chosen
        x[index, j] = 1

    #A feasible solution of the original problem will always result in an objective value > lagrangian relaxation objective value
    #That is an upper bound for the optimum solution
    Z = 0
    for i in range(num_locations):
        for j in range(num_customers):
            Z = Z + d[j] * c[i,j] * x[i,j]
    return Z,x
def lagrangian_relaxation(p, max_iter = 1000, tol = 1e-6, theta = 1.0):
    #Track the previous upper and lower bounds
    UB = []
    LB = []

    #Track the best known upper and lower bounds
    Z_UB = float('inf') #lower bound init to +infinity
    Z_LB = float('-inf') #upperbound init to -infinity

    #Track the best known feasible solutions
    x_best = np.zeros((num_locations, num_customers))
    y_best = np.zeros(num_locations)

    #Init multiplier
    lambda_vector = np.zeros(len(customers))
    for k in range(max_iter):
        #Obtain lower and upper bounds
        Z_D, x_D, y = lower_bound(lambda_vector)
        Z, x = upper_bound(y)
        #Update the upper bound
        #If the current UB is less than the best-known UB, it updates the best known UB and the corresponding solution
        if Z < Z_UB:
            Z_UB = Z
            x_best = x
            y_best = y
        #Update the lower bound
        #If the current lower bound is larger than the best known LB, updates the best known LB
        if Z_D > Z_LB:
            Z_LB = Z_D
        #Keep track the upper and lower bounds from the current iteration
        UB.append(Z)
        LB.append(Z_D)
        #Determine the step size and update the multiplier
        sum_x_D = np.sum(x_D, axis=0)
        residual = 1 - sum_x_D
        t = theta * (Z_UB - Z_D) / sum(residual ** 2)
        lambda_vector = lambda_vector + t * residual

        #Compute the optimality gap
        opt_gap = (Z_UB - Z_LB) / Z_UB
        if opt_gap < tol:
            break
    return Z_UB, x_best, y_best, UB, LB

# Call the function
Z_opt, x_opt, y_opt = optimal(p, locations, customers, c, d)

# Print the results
print(f"Optimal objective value without relaxation: {Z_opt}")
# print("Facilities opened:")
# for i in range(num_locations):
#     if y_opt[i] > 0.5:  # Check if the facility is opened
#         print(f"  Facility {i + 1} is opened")
#
# print("Transportation details:")
# for i in range(num_locations):
#     for j in range(num_customers):
#         if x_opt[i, j] > 0:  # Check if there is a transportation from facility i to customer j
#             print(f"  Transported {x_opt[i, j] * d[j]} from facility {i + 1} to customer {j + 1}")

Z_UB, x_best, y_best, UB, LB = lagrangian_relaxation(p)
print(f"Optimal objective value with relaxation: {Z_UB}")

iter = np.arange(0, len(LB))  # Example iteration numbers
# Plotting the data
plt.plot(iter, LB, color="red", linewidth=2.0, linestyle="-", marker="o", label="Lower Bound")
plt.plot(iter, UB, color="blue", linewidth=2.0, linestyle="-.", marker="D", label="Upper Bound")

# Labeling axes
plt.xlabel(r"iteration clock $k$", fontsize="xx-large")
plt.ylabel("Bounds", fontsize="xx-large")

# Putting the legend and determining the location
plt.legend(loc="lower right", fontsize="x-large")

# Add grid lines
plt.grid(color="#DDDDDD", linestyle="-", linewidth=1.0)

# Set tick parameters
plt.tick_params(axis="both", which="major", labelsize="x-large")

# Title
plt.title("Lower and Upper Bounds", fontsize="xx-large")
plt.show()









