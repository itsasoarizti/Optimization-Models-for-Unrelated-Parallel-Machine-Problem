import numpy as np
import random
from deap import base, creator, tools, algorithms

# Create individual
def generator(C: np.ndarray, 
              tasks: 'dict[int, list[int]]',
              ) -> 'dict[int, list[int]]':
    '''
    This function generates a random order of tasks for each machine
    It returns a dictionary with the order of tasks for each machine
    Example:
        Machine 0 can perform tasks 1, 3, 4, 5
        Machine 1 can perform tasks 1, 2, 4, 5, 6
        Machine 2 can perform tasks 1, 2, 5, 6

        Tasks : {0: [1, 3, 4, 5], 1: [1, 2, 4, 5, 6], 2: [1, 2, 5, 6]}
    The function will return:
        Machine 0 performs tasks in order : 3
        Machine 1 performs tasks in order : 4
        Machine 2 performs tasks in order : 5, 2, 6, 1

        Output: {0: [0, 1, 0, 0], 1: [0, 0, 1, 0, 0], 2: [4, 2, 1, 3]}

    Args:
        C (np.ndarray): The matrix of costs 
        tasks (dict[int, list[int]]): The tasks that each machine can perform
        
    Returns:
        dict: A dictionary with the order of tasks for each machine

    
    '''
    # Get the number of machines and tasks
    m, n, _ = C.shape
    # Remove the dummy tasks
    n -= 2

    # Create individual
    ind = {}
    for _m in range(m):
        ind[_m] = [0] * len(tasks[_m])
    done = []
    while len(done) < n:
        _m = random.randint(0, m - 1)
        # Get the tasks that can be done in machine _m but are not done yet
        available_tasks = [i for i in tasks[_m] if i not in done]
        random.shuffle(available_tasks)
        # Get saved order, initially all 0 values
        order = ind[_m]
        # Check if any task can be done
        if len(available_tasks)>0:
            # Get last task order, if 0 set to 1
            aux = max(order) + 1
            for task_id in available_tasks:
                # Randomly decide to jump to the next task
                if random.random() < 0.5:
                    continue
                index_of_i = tasks[_m].index(task_id)
                order[index_of_i] = aux
                aux += 1
                done.append(task_id)
        ind[_m] = order
        
    return ind

def evalMTSP(individual : 'dict[int, list[int]]',
             C: np.ndarray,
             tasks:'dict[int, list[int]]',) -> tuple[int]:
    '''
    Evaluate the time taken to complete all tasks for a given individual
    The time taken is the maximum time taken by any machine to complete all tasks
    Args:
        individual (dict[int, list[int]]): The order of tasks for each machine
    Returns:
        tuple[int]: The time taken to complete all tasks
    '''
    m, n, _ = C.shape
    n -= 2
    all_times = []
    for _m in range(m):
        cant = len([x for x in individual[_m] if x != 0])
        m_time = 0
        if cant >= 1:
            for i in range(cant):
                if i != 0:
                    ind = individual[_m].index(i) # Task order
                    aux = tasks[_m][ind] # Task i 
                else: 
                    aux = 0 # Machine start from dummy job
                ind2 = individual[_m].index(i+1) # Next task order
                aux2 = tasks[_m][ind2] # Task j
                m_time += C[_m][aux][aux2] # Pt(j) + St(i,j)
    
        all_times.append(m_time)
    makespan = max(all_times) # longest time
    return makespan,

def machine_i(individual: 'dict[int, list[int]]', 
              i: int,
              C: np.ndarray,
              tasks:  'dict[int, list[int]]') -> list[int]:
        '''
        Get in which machine the task i is performed
        Args:
            individual (dict[int, list[int]]): The order of tasks for each machine
            i (int): The task to count
            C (np.ndarray): The matrix of costs
            tasks (dict[int, list[int]]): The tasks that each machine can perform
        Returns:
            list['int']: The machines that perform the task i
        '''
        m, _, _ = C.shape
        machines = []
        for _m in range(m):
            if i in tasks[_m]:
                index_i = tasks[_m].index(i)
                order_i = individual[_m][index_i]
                if order_i !=0:
                    machines.append(_m)
        return machines

def fix(individual: 'dict[int, list[int]]',
             C: np.ndarray,
             tasks: 'dict[int, list[int]]') -> 'dict[int, list[int]]':
    '''
    This is a repair function that fixes the individual by making sure that each task is performed exactly once

    Args:
        individual (dict[int, list[int]]): The order of tasks for each machine
        C (np.ndarray): The matrix of costs
        tasks (dict[int, list[int]]): The tasks that each machine can perform

    Returns:
        dict[int, list[int]]: The fixed individual
    '''

    m, n , _ = C.shape
    n -= 2
    
    for i in range(1, n + 1):
        # Get the machines that performs task i in the individual
        machines_w_i = machine_i(individual,i, C, tasks)
        if len(machines_w_i) > 1:
            selected = random.choice(machines_w_i)
            for _m in machines_w_i:
                if _m != selected: # Remove the task from the other machines
                    index_i = tasks[_m].index(i) # Get the index of the task in the machine
                    task_order = individual[_m][index_i] # Get the order of the task
                    individual[_m][index_i] = 0 # Remove the task from the machine
                    for i in range(0,len(individual[_m])): 
                        if individual[_m][i] > task_order: # Update the order of the tasks
                            individual[_m][i] -= 1
        machines_w_i = machine_i(individual,i, C, tasks)
        if len(machines_w_i) == 0:
            posible_machines = [_m for _m in range(m) if i in tasks[_m]]
            _m = random.choice(posible_machines)
            index_i = tasks[_m].index(i)
            individual[_m][index_i] =  max(individual[_m]) + 1 # Set in the last position                   
    return individual

def mutate(individual : 'dict[int, list[int]]',
           indpb: float,
           C: np.ndarray) -> 'tuple[dict[int, list[int]]]':
    '''
    Mutate the individual by changing the order of tasks in a machine
    Args:
        individual (dict[int, list[int]]): The order of tasks for each machine
        indpb (float): The probability of mutation
        C (np.ndarray): The matrix of costs
    Returns:
        tuple[dict[int, list[int]]]: The mutated individual
    
    '''
    m, _, _ = C.shape

    for _m in range(m):
        if random.random() < indpb:
            nonzero_indx = [i for i, x in enumerate(individual[_m]) if x != 0]
            if len(nonzero_indx) != 0:
                nonzero_values = [individual[_m][i] for i in nonzero_indx]
                random.shuffle(nonzero_values)
                for idx, valor in zip(nonzero_indx, nonzero_values):
                    individual[_m][idx] = valor
    return individual,

def mate(ind1: 'dict[int, list[int]]',
         ind2: 'dict[int, list[int]]',
         C: np.ndarray,
         tasks:'dict[int, list[int]]',
         toolbox) -> 'tuple[dict[int, list[int]]]':
    '''
    Mate two individuals by exchanging a random task
    Args:
        ind1 (dict[int, list[int]]): The first individual
        ind2 (dict[int, list[int]]): The second individual
        C (np.ndarray): The matrix of costs
        tasks (dict[int, list[int]]): The tasks that each machine can perform
    Returns:
        tuple[dict[int, list[int]], dict[int, list[int]]]: The mated individuals
    '''
    
    m, _, _ = C.shape
    _m = random.choice(range(m))
    child1 = toolbox.clone(ind1)
    child2 = toolbox.clone(ind2)
    child1[_m] = ind2[_m]
    child2[_m] = ind1[_m]
    child1 = fix(child1, C, tasks)
    child2 = fix(child2, C, tasks)
    return child1, child2

def GEN(n: int, m: int, C: np.ndarray, population: int = 500) -> float, np.ndarray:
    '''
    This function solves the problem using a Genetic Algorithm

    Args:
        n (int): Number of tasks
        m (int): Number of machines
        C (np.ndarray): Cost matrix
        population (int): The size of the population

    Returns:
        float: The time taken to complete all tasks
        np.ndarray: tasks sequence for each machine
    '''

    tasks = {_m: [i for i in range(1, n + 1) if sum(C[_m][i]) != 0] for _m in range(m)}


    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # Minimize total time
    creator.create("Individual", dict, fitness=creator.FitnessMin) # Represents an individual as a sequence of tasks and machines
    toolbox = base.Toolbox()


    toolbox.register("generator", generator, C=C, tasks=tasks)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.generator)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    toolbox.register("mate", mate, C=C, tasks=tasks, toolbox=toolbox)  # Mate with gene 0 always at the beginning
    toolbox.register("mutate", mutate, indpb=0.1, C=C)  # Mutate with gene 0 always at the beginning
    toolbox.register("select", tools.selTournament, tournsize=3)  # Selection: tournament featuring 3 individuals
    toolbox.register("evaluate", evalMTSP, C=C, tasks=tasks)  # Evaluation

    
    pop = toolbox.population(population)  # We create a population of 100 individuals
    hof = tools.HallOfFame(3)  # We keep the best candidate we find

    # We evaluate the initial population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit


    algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.4, ngen=100, stats=False, halloffame=hof, verbose=False)

    # Remove the registred functions
    toolbox.unregister("population")
    toolbox.unregister("mate")
    toolbox.unregister("mutate")
    toolbox.unregister("select")
    toolbox.unregister("evaluate")
    toolbox.unregister("individual")
    toolbox.unregister("generator")

    del creator.FitnessMin
    del creator.Individual
    del toolbox     

    best_individual = hof[0]    
    best_seq = generate_seq(best_individual, tasks)
  
    return best_individual.fitness.values[0], best_seq
