import numpy as np

def probabilities(
        pheromones: np.ndarray,
        _m: int,
        actual_task: int,
        undone: 'list[int]',
        C,
        alpha: float = 1.0,
        beta: float = 5.0) -> np.ndarray:
    """
    This function calculates the probability of moving to the next task based on the pheromones and the visibility.

    Args:
            pheromones (np.ndarray): Pheromone matrix.
            _m (int): Machine index.
            actual_task (int): Current task.
            undone (list[int]): List of tasks not yet performed.
            alpha (float): Pheromone importance.
            beta (float): Visibility importance.

    Returns:
            list[float]: List of probabilities of moving to the next task.
    """

    # Extract pheromones for the current machine and task
    tau = pheromones[_m, actual_task, undone] ** alpha

    # Calculate visibility (inverse of the cost matrix) and handle division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        eta = np.where(C[_m, actual_task, undone] != 0,
                       (1.0 / C[_m, actual_task, undone]) ** beta,
                       0.0)  # If cost is zero, set visibility to zero

    # Combine pheromone and visibility influences
    mult = tau * eta

    # Normalize to get probabilities, ensuring we don't divide by zero
    sum = np.sum(mult)
    if sum == 0:
        # Equal probability if all values are zero
        prob = np.ones_like(mult) / len(mult)
    else:
        prob = mult / sum

    return prob

def generate_route(pheromones: np.ndarray,
                m: int,
                n: int,
                C: np.ndarray) -> np.ndarray:
    '''
    This function creates a route for each machine.

    Args:
            pheromones (np.ndarray): Pheromone matrix.
            m (int): Number of machines.
            n (int): Number of tasks.

    Returns:
            list[list[int]]: List of routes for each machine.

    '''
    # Initialize routes with -1 placeholders (indicating unassigned slots)
    # Exclude the tasks that cannot be performed by each machine
    routes = -1 * np.ones((m, n + 2), dtype=int)

    # Start each route with the dummy start task (0)
    routes[:, 0] = 0

    # List of tasks to be assigned (excluding dummy tasks 0 and n+1)
    undone = np.arange(1, n + 1)

    # Track the current position for each machine's route
    current_positions = np.ones(m, dtype=int)

    while undone.size > 0:
        # Randomly select a machine
        _m = np.random.randint(m)

        # Current task for the selected machine (_m)
        actual_task = routes[_m, current_positions[_m] - 1]

        # Find tasks that the machine can perform (vectorized operation)
        # Boolean mask of tasks that can be performed
        # If the cost is zero, the task cannot be performed

        mask = np.any(C[_m, undone], axis=1)

        undone_m = undone[mask]

        # If no tasks can be assigned, continue to the next iteration
        if undone_m.size == 0:
            continue

        # Calculate probabilities for selecting the next task
        prob = probabilitatea(pheromones, _m, actual_task, undone_m,C)

        # Choose the next task based on probabilities
        next_task = np.random.choice(undone_m, p=prob / prob.sum())

        # Add the selected task to the route and remove it from unassigned tasks
        routes[_m, current_positions[_m]] = next_task
        current_positions[_m] += 1
        # Remove the selected task from the list of unassigned tasks
        undone = undone[undone != next_task]

    # Add the dummy end task (n + 1) to each route
    for i in range(m):
        routes[i, current_positions[i]] = n + 1

    return routes

def pheromone_update(pheromones: np.ndarray,
                      solutions: np.ndarray,
                      time_solutions: np.ndarray,
                      rho: float = 0.5,
                      Q: float = 100) -> None:
    '''
    This function updates the pheromones based on the solutions found, and does not return anything.

    Args:
            pheromones (np.ndarray): Pheromone matrix.
            solutions (np.ndarray): List of routes for each machine.
            time_solutions (np.ndarray): List of total times for each machine.
            rho (float): Evaporation rate.
            Q (float): Constant for pheromone deposition.

    Returns:
            None
    '''
    # Evaporation: reduce pheromone levels
    pheromones *= (1 - rho)

    # Vectorized deposition
    # deposicion
    for routes, time_total in zip(solutions, time_solutions):
        for _m, route in enumerate(routes):
            if time_total[_m] == 0:
                continue  # Skip if the total time is zero to avoid division by zero

            for i in range(len(route)-1):
                task_i = route[i]
                task_next = route[i+1]
                # Apply pheromone deposit to both directions (i -> i+1 and i+1 -> i)
                pheromones[_m][task_i][task_next] += Q / time_total[_m]
                pheromones[_m][task_next][task_i] += Q / time_total[_m]


def calculate_time(routes: np.ndarray, C: np.ndarray) -> np.ndarray:
    '''
    Calculates the total time for each route.

    Args:
        routes (np.ndarray): Array of routes for each machine (shape: m x k, where k varies per route).
        C (np.ndarray): Cost matrix (shape: m x n x n).

    Returns:
        np.ndarray: Array of total times for each machine (shape: m).
    '''
    total_times = np.zeros(len(routes))

    for _m, route in enumerate(routes):
        # Calculate the cost for each consecutive pair of tasks
        route_pairs = np.array([route[:-1], route[1:]])
        total_times[_m] = np.sum(C[_m, route_pairs[0], route_pairs[1]])

    return total_times

def ACO(n: int, m: int, C: np.ndarray, ants: int = 50, iterations: int = 200):
    '''
    This function solves the problem using Ant Colony Optimization (ACO).

    Args:
        n (int): Number of tasks.
        m (int): Number of machines.
        C (np.ndarray): Cost matrix.
        ants (int): Number of ants.
        iterations (int): Number of iterations.

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

    pheromones = np.ones((m, n+2, n+2))

    for iteration in range(iterations):
        #print(f"\nIteration {iteration + 1}/{iterations}")

        # Each ant constructs a solution
        routes_ants = []
        times_ants = []

        for _ in range(ants):
            # Generate routes and calculate times
            routes = generate_route(pheromones, m, n, C)
            times = calculate_time(routes, C)

            routes_ants.append(routes)
            times_ants.append(times)

            # Find the maximum time for this solution (the makespan)
            max_time = max(times)

            # Update the best solution if a better one is found
            if max_time < best_total_time:
                best_total_time = max_time
                # print("New best solution found with makespan: {best_total_time}")
        # Update pheromones based on all solutions found in this iteration
        pheromones_berritu(pheromones,
                          routes_ants,
                          times_ants, rho, Q)

    return best_total_time
