
import numpy as np

# Función para calcular probabilidades de movimiento
def probabilitatea(
        feromonak: np.ndarray,
        _m: int,
        tarea_actual: int,
        egin_gabe: 'list[int]',
        C,
        alpha: float = 1.0,
        beta: float = 5.0) -> np.ndarray:
    """
    This function calculates the probability of moving to the next task based on the pheromones and the visibility.

    Args:
            feromonak (np.ndarray): Pheromone matrix.
            _m (int): Machine index.
            tarea_actual (int): Current task.
            egin_gabe (list[int]): List of tasks not yet performed.
            alpha (float): Pheromone importance.
            beta (float): Visibility importance.

    Returns:
            list[float]: List of probabilities of moving to the next task.
    """

    # Extract pheromones for the current machine and task
    tau = feromonak[_m, tarea_actual, egin_gabe] ** alpha

    # Calculate visibility (inverse of the cost matrix) and handle division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        eta = np.where(C[_m, tarea_actual, egin_gabe] != 0,
                       (1.0 / C[_m, tarea_actual, egin_gabe]) ** beta,
                       0.0)  # If cost is zero, set visibility to zero

    # Combine pheromone and visibility influences
    biderketa = tau * eta

    # Normalize to get probabilities, ensuring we don't divide by zero
    suma = np.sum(biderketa)
    if suma == 0:
        # Equal probability if all values are zero
        prob = np.ones_like(biderketa) / len(biderketa)
    else:
        prob = biderketa / suma

    return prob


# Función para construir rutas para todas las máquinas
def rutak_sortu(feromonak: np.ndarray,
                m: int,
                n: int,
                C: np.ndarray) -> np.ndarray:
    '''
    This function creates a route for each machine.

    Args:
            feromonak (np.ndarray): Pheromone matrix.
            m (int): Number of machines.
            n (int): Number of tasks.

    Returns:
            list[list[int]]: List of routes for each machine.

    '''
    # Initialize routes with -1 placeholders (indicating unassigned slots)
    # Exclude the tasks that cannot be performed by each machine
    rutak = -1 * np.ones((m, n + 2), dtype=int)

    # Start each route with the dummy start task (0)
    rutak[:, 0] = 0

    # List of tasks to be assigned (excluding dummy tasks 0 and n+1)
    egin_gabe = np.arange(1, n + 1)

    # Track the current position for each machine's route
    current_positions = np.ones(m, dtype=int)

    while egin_gabe.size > 0:
        # Randomly select a machine
        _m = np.random.randint(m)

        # Current task for the selected machine (_m)
        tarea_actual = rutak[_m, current_positions[_m] - 1]

        # Find tasks that the machine can perform (vectorized operation)
        # Boolean mask of tasks that can be performed
        # If the cost is zero, the task cannot be performed

        mask = np.any(C[_m, egin_gabe], axis=1)

        egin_gabe_m = egin_gabe[mask]

        # If no tasks can be assigned, continue to the next iteration
        if egin_gabe_m.size == 0:
            continue

        # Calculate probabilities for selecting the next task
        prob = probabilitatea(feromonak, _m, tarea_actual, egin_gabe_m,C)

        # Choose the next task based on probabilities
        hurrengo_tarea = np.random.choice(egin_gabe_m, p=prob / prob.sum())

        # Add the selected task to the route and remove it from unassigned tasks
        rutak[_m, current_positions[_m]] = hurrengo_tarea
        current_positions[_m] += 1
        # Remove the selected task from the list of unassigned tasks
        egin_gabe = egin_gabe[egin_gabe != hurrengo_tarea]

    # Add the dummy end task (n + 1) to each route
    for i in range(m):
        rutak[i, current_positions[i]] = n + 1

    return rutak


# Actualización de feromonas
def feromonak_berritu(feromonak: np.ndarray,
                      soluzioak: np.ndarray,
                      denb_soluzioak: np.ndarray,
                      rho: float = 0.5,
                      Q: float = 100) -> None:
    '''
    This function updates the pheromones based on the solutions found, and does not return anything.

    Args:
            feromonak (np.ndarray): Pheromone matrix.
            soluzioak (np.ndarray): List of routes for each machine.
            denb_soluzioak (np.ndarray): List of total times for each machine.
            rho (float): Evaporation rate.
            Q (float): Constant for pheromone deposition.

    Returns:
            None
    '''
    # Evaporation: reduce pheromone levels
    feromonak *= (1 - rho)

    # Vectorized deposition
    # deposicion
    for rutak, denb_totala in zip(soluzioak, denb_soluzioak):
        for _m, ruta in enumerate(rutak):
            if denb_totala[_m] == 0:
                continue  # Skip if the total time is zero to avoid division by zero

            for i in range(len(ruta)-1):
                task_i = ruta[i]
                task_next = ruta[i+1]
                # Apply pheromone deposit to both directions (i -> i+1 and i+1 -> i)
                feromonak[_m][task_i][task_next] += Q / denb_totala[_m]
                feromonak[_m][task_next][task_i] += Q / denb_totala[_m]


def denbora_kalkulatu(rutak: np.ndarray, C: np.ndarray) -> np.ndarray:
    '''
    Calculates the total time for each route.

    Args:
        rutak (np.ndarray): Array of routes for each machine (shape: m x k, where k varies per route).
        C (np.ndarray): Cost matrix (shape: m x n x n).

    Returns:
        np.ndarray: Array of total times for each machine (shape: m).
    '''
    denbora_totalak = np.zeros(len(rutak))

    for _m, ruta in enumerate(rutak):
        # Calculate the cost for each consecutive pair of tasks
        ruta_pairs = np.array([ruta[:-1], ruta[1:]])
        denbora_totalak[_m] = np.sum(C[_m, ruta_pairs[0], ruta_pairs[1]])

    return denbora_totalak

def ACO(n: int, m: int, C: np.ndarray, hormigas: int = 50, iteraciones: int = 200):
    '''
    This function solves the problem using Ant Colony Optimization (ACO).

    Args:
        n (int): Number of tasks.
        m (int): Number of machines.
        C (np.ndarray): Cost matrix.
        hormigas (int): Number of ants.
        iteraciones (int): Number of iterations.

    Returns:
        float: Best total time found by the algorithm.
    
    '''
    
    # Parameters
    alpha = 1.0           # Importance of pheromone
    beta = 5.0            # Importance of heuristic (visibility)
    rho = 0.5             # Pheromone evaporation rate
    Q = 100               # Constant for pheromone deposition

    # Best solution tracking
    best_total_time = float('inf')

    feromonak = np.ones((m, n+2, n+2))

    for iteration in range(iteraciones):
        #print(f"\nIteration {iteration + 1}/{iteraciones}")

        # Each ant constructs a solution
        routes_ants = []
        times_ants = []

        for _ in range(hormigas):
            # Generate routes and calculate times
            routes = rutak_sortu(feromonak, m, n, C)
            times = denbora_kalkulatu(routes, C)

            routes_ants.append(routes)
            times_ants.append(times)

            # Find the maximum time for this solution (i.e., the makespan)
            max_time = max(times)

            # Update the best solution if a better one is found
            if max_time < best_total_time:
                best_total_time = max_time
                # print("New best solution found with makespan: {best_total_time}")
        # Update pheromones based on all solutions found in this iteration
        feromonak_berritu(feromonak,
                          routes_ants,
                          times_ants, rho, Q)

    return best_total_time
