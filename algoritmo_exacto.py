import numpy as np
import cvxpy as cv
import time

def algoritmo_exacto(n,m,C):

    # Define variables

    X = [cv.Variable((n+2, n+2), boolean=True) for _m in range(m)] # C-ko matrize bakoitzerako X bat, _m makinak ingoitun tareak eta hauen ordena adierazteute
    u = cv.Variable(n+2, integer=True)  # makina bakoitzeako tareen orden bat
    J = cv.Variable(m, integer=True)
    Z = cv.Variable(integer=True)
    
    
    ones = np.ones((n+2,1))
    ones2 = np.ones((n+2,1))
    ones2[0] = m
    ones2[n+1] = 0
    ones3 = np.ones((n+2,1))
    ones3[0] = 0
    ones3[n+1] = m
    

    # funcion objetivo

    objective = cv.Minimize(Z)

    # restrikzioak

    constraints = []
    
    # makinak ez iteko egin ezin ditun tareak
    
    for _m in range(m):
        for i in range(n+1):
            for j in range(n+1):
                if C[_m][j,i] == 0:
                    constraints += [X[_m][j,i] == 0]

    for _m in range(m):
        constraints += [X[_m][0,:] @ ones == 1] #0-n hasi
        constraints += [X[_m][:,n+1] @ ones == 1] #(n+1)-en bukatu
        constraints += [X[_m][1:-1,:] @ ones <= 1]  # bestetan igual tarea hoi dao igual ez
        constraints += [X[_m][:,1:-1].T @ ones <= 1]
  
    # _m i tareara badoa i tareatik atera behar du
    for _m in range(m):
        for i in range(1,n+1):
            constraints += [cv.sum(X[_m][:,i]) == cv.sum(X[_m][i,:])]

    # ezin du tarea batetik tarea igualera jun
    for _m in range(m):
        for i in range(n+2):
            constraints += [X[_m][i,i] == 0]  


    for _m in range(m):
    # Restricción para evitar subtours en la ruta
        for i in range(1, n+1):
            for j in range(1, n+1):
                if i != j:
                    constraints += [u[ i] - u[j] + 1 <= (100000) * (1 - (sum(X[_m] for _m in range(m)))[i, j])]

    constraints += [(sum(X[_m] for _m in range(m))) @ ones == ones2]
    constraints += [(sum(X[_m].T for _m in range(m))) @ ones == ones3]

    # J bektorea makina bakoitzak tardatzeun denbora totalakin
    for _m in range(m):
        constraints += [cv.sum(cv.multiply(C[_m], X[_m])) == J[_m]]

    # Z = J bektoreko balio altuena
    for _m in range(m):
        constraints += [Z >= J[_m]]

    # optimizazio problema

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

    # rutak sortu
    
    rutak = []

    for _m in range(len(X_sol)):
        ruta = [0]
        actual = X_sol[_m][0][1]
        while actual !=n+1:
            ruta.append(actual)
            indices = np.where(np.array(X_sol[_m])[:, 0] == actual)[0]
            siguiente = X_sol[_m][indices[0]][1]
            actual = siguiente
        ruta.append(n+1)
        rutak.append(ruta)
        

    return np.round(Z.value), rutak