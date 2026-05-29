import numpy as np
import cvxpy as cv
import time

def MILP(n,m,C):

    # Variables

    X = [cv.Variable((n+2, n+2), boolean=True) for _m in range(m)]
    u = cv.Variable(n+2, integer=True)
    J = cv.Variable(m, integer=True)
    Z = cv.Variable(integer=True)   
    
    ones = np.ones((n+2,1))
    ones2 = np.ones((n+2,1))
    ones2[0] = m
    ones2[n+1] = 0
    ones3 = np.ones((n+2,1))
    ones3[0] = 0
    ones3[n+1] = m
    

    # Objective function

    objective = cv.Minimize(Z)

    # Constraints

    constraints = []
    
    # so the machine doesn't try to do tasks it can't handle
    
    for _m in range(m):
        for i in range(n+1):
            for j in range(n+1):
                if C[_m][j,i] == 0:
                    constraints += [X[_m][j,i] == 0]

    for _m in range(m):
        constraints += [X[_m][0,:] @ ones == 1] # starts in 0
        constraints += [X[_m][:,n+1] @ ones == 1] # finishes in (n+1)
        constraints += [X[_m][1:-1,:] @ ones <= 1]
        constraints += [X[_m][:,1:-1].T @ ones <= 1]
  
    # if a machine goes to a task, it has to go out from that same task
    for _m in range(m):
        for i in range(1,n+1):
            constraints += [cv.sum(X[_m][:,i]) == cv.sum(X[_m][i,:])]

    # can't switch from one task to the same task
    for _m in range(m):
        for i in range(n+2):
            constraints += [X[_m][i,i] == 0]  

    for _m in range(m):
    # prevent soubtours
        for i in range(1, n+1):
            for j in range(1, n+1):
                if i != j:
                    constraints += [u[ i] - u[j] + 1 <= (100000) * (1 - (sum(X[_m] for _m in range(m)))[i, j])]

    constraints += [(sum(X[_m] for _m in range(m))) @ ones == ones2]
    constraints += [(sum(X[_m].T for _m in range(m))) @ ones == ones3]

    # J vector: total time of each machine
    for _m in range(m):
        constraints += [cv.sum(cv.multiply(C[_m], X[_m])) == J[_m]]

    # Z value: highest value of J
    for _m in range(m):
        constraints += [Z >= J[_m]]

    # Optimization problem

    start = time.time()
    prob = cv.Problem(objective, constraints)
    prob.solve(verbose=False, solver='CBC')
    elapsed_time = time.time() - start

    X_sum = sum(X[_m] for _m in range(m))
    sol = []
    for _m in range(m):
        X_sol_m = np.argwhere(np.isclose(X[_m].value, 1))
        sol.append(X_sol_m)

    X_sol = [a.tolist() for a in sol]

    # Create routes
    
    routes = []

    for _m in range(len(X_sol)):
        route = [0]
        actual = X_sol[_m][0][1]
        while actual !=n+1:
            route.append(actual)
            indx = np.where(np.array(X_sol[_m])[:, 0] == actual)[0]
            next = X_sol[_m][indx[0]][1]
            actual = next
        route.append(n+1)
        routes.append(route)
        

    return np.round(Z.value), routes
