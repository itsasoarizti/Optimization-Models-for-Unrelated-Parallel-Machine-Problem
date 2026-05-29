import random
from deap import base, creator, tools, algorithms

def creator(C: np.ndarray, 
            tasks: ’dict[int, list[int]]’,
            ) -> ’dict[int, list[int]] ’:
    '''
    This function generates random task sequences for each machine. 
    It returns a dictionary with the task order for each machine.     
    Example:
        Machine 0 can perform tasks 1, 3, 4 and 5
        Machine 1 can perfomr tasks 1, 2, 4, 5 and 6
        Machine 2 can perform tasks 1, 2, 5, and 6

        Tasks: {0: [1, 3, 4, 5] , 1: [1 , 2 , 4 , 5 , 6] , 2: [1 , 2 , 5 , 6]}

    The function will return the following:
        Task sequence of Machine 0: 3
        Task sequence of Machine 1: 4
        Task sequence of Machine 2: 5 , 2 , 6 , 1

        Output : {0: [0 , 1 , 0 , 0] , 1: [0 , 0 , 1 , 0 , 0] , 2: [4 , 2 , 1 , 3]}

    Parameters:
        C (np. ndarray): Cost matrix
        tasks (dict[int, list[int]]): The tasks each machine can perform

    Returns:
        dict: Dictionary indicating the task order for each machine

    '''
    m, n, _ = C.shape
    n -= 2
def genetic_algorithm(n,m,C):
    
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", dict, typecode='i', fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    
    # tasks each machine can do
    tareak = {_m: [i for i in range(1, n + 1) if sum(C[_m][i]) != 0] for _m in range(m)}

    # individuoak eta populazioa sortu
    
    def indices():
        ind = {}
        for _m in range(m):
            ind[_m] = [0] * len(tareak[_m])
            
        done = []
        while len(done) < n:
            _m = random.randint(0, m - 1)
            # Get the tasks that can be done in machine _m but are not done yet
            available_tasks = [i for i in tareak[_m] if i not in done]
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
                    index_of_i = tareak[_m].index(task_id)
                    order[index_of_i] = aux
                    aux += 1
                    done.append(task_id)
            ind[_m] = order
            
        return ind
    
    toolbox.register("indices", indices)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    #ruta bakoitzaren denbora kalkulatzeko funtzioa
    
    def evalMTSP(individual): 
    
        all_times = []
        for _m in range(m):
            kant = len([x for x in individual[_m] if x != 0])
            m_time =0
            if kant >= 1:
                for i in range(kant):
                    if i != 0:
                        ind = individual[_m].index(i)
                        aux = tareak[_m][ind]
                    else:
                        aux = 0
                    ind2 = individual[_m].index(i+1)
                    aux2 = tareak[_m][ind2]
                    m_time += C[_m][aux][aux2]
                                                
            all_times.append(m_time)
            
        final_time = max(all_times) # denbora luzeena
        return final_time,
    
    #inidiviuo batek ez badu balio, hau konpontzeko funtzioa
    
    def machine_i(individual,i):
        machines = []
        for _m in range(m):
            if i in tareak[_m]:
                index_i = tareak[_m].index(i)
                orden_i = individual[_m][index_i]
                if orden_i !=0:
                    machines.append(_m)
        return machines

    def konpondu(individual):
        for i in range(1, n + 1):
            # Get the machines that performs task i in the individual
            machines_w_i = machine_i(individual,i)
            if len(machines_w_i) > 1:
                selected = random.choice(machines_w_i)
                for _m in machines_w_i:
                    if _m != selected: # Remove the task from the other machines
                        index_i = tareak[_m].index(i) # Get the index of the task in the machine
                        task_order = individual[_m][index_i] # Get the order of the task
                        individual[_m][index_i] = 0 # Remove the task from the machine
                        for i in range(0,len(individual[_m])): 
                            if individual[_m][i] > task_order: # Update the order of the tasks
                                individual[_m][i] -= 1
            machines_w_i = machine_i(individual,i)
            if len(machines_w_i) == 0:
                posible_machines = [_m for _m in range(m) if i in tareak[_m]]
                _m = random.choice(posible_machines)
                index_i = tareak[_m].index(i)
                individual[_m][index_i] =  max(individual[_m]) + 1 # Set in the last position                   
        return individual
        
    # mutazioa
    
    def mutate(individo,indpb):
        if random.random() < indpb:
            for _m in range(m):
                indices_no_ceros = [i for i, x in enumerate(individo[_m]) if x != 0]
                if len(indices_no_ceros) != 0:
                    valores_no_ceros = [individo[_m][i] for i in indices_no_ceros]
                    random.shuffle(valores_no_ceros)
                    for idx, valor in zip(indices_no_ceros, valores_no_ceros):
                        individo[_m][idx] = valor
        return individo,
    
    # kruzea
    
    def mate(ind1,ind2):
        _m = random.choice(range(m))
        child1 = toolbox.clone(ind1)
        child2 = toolbox.clone(ind2)
        child1[_m] = ind2[_m]
        child2[_m] = ind1[_m]
        child1 = konpondu(child1)
        child2 = konpondu(child2)
        return child1, child2
    
    #funtzioak toolbox-en registratu
    
    toolbox.register("mate", mate)  # Cruce con el gen 0 siempre al inicio
    toolbox.register("mutate", mutate, indpb=0.1)  # Mutación con el gen 0 al inicio
    toolbox.register("select", tools.selTournament, tournsize=3)  # Selección: torneo con 3 individuos
    toolbox.register("evaluate", evalMTSP)
    
    #populazioa sortu eta algoritmoa exekutatu
    
    pop = toolbox.population(100)  # 100 individuoko populazioa sortu
    hof = tools.HallOfFame(1)  # aurkitutako individuo hoberena gorde

    # hasierako populazioa ebaluatu
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
        
    # algoritmoa exekutatu
    
    algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.4, ngen=50, stats=None, halloffame=hof, verbose=False)

    # individuo hoberena
    
    best_individual = hof[0]
    
    # individuo hoberenean, makina bakoitzaren ruta sortu
    
    ruta=[]

    for _m in range(m):
        ruta_m = []
        tarea_kant = len([x for x in best_individual[_m] if x != 0])
        for i in range(1,tarea_kant+1):      
            ind = best_individual[_m].index(i)
            tarea = tareak[_m][ind]
            ruta_m.append(tarea)
        ruta_m = [0] + ruta_m + [n+1]
        ruta.append(ruta_m)        
    print(ruta)
    
    # individuo hoberenaren makina bakoitzaren denbora eta denbora maximoa
    
    all_times = []
    for _m in range(m):
        m_time = 0
        for i in range(1, len(ruta[_m])):
            if i != 0:
                m_time += C[_m][ruta[_m][i-1]][ruta[_m][i]]
        all_times.append(m_time)
        
    best_time =max(all_times)
    
    return ruta, all_times, best_time
