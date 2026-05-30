def create_eligibility_matrix(m, n):

    """
    This function creates a eligibility matrix, ensuring each machine is assigned at least one task.
    
    Args:
        m (int): Number of machines.
        n (int): Number of real tasks.
        
    Returns:
        eligibility_matrix (np.ndarray): (m,n) size eligibility matrix.
    """
    
    eligibility_matrix = np.zeros((m, n), dtype=bool)
    eligitibility_perc = np.random.uniform(20, 80, size=n)

    for task in range(n):
        n_candidate_machines = max(1, int((eligitibility_perc[task] / 100) * m))
        candidate_machines = np.random.choice(range(m), size=n_candidate_machines, replace=False)
        eligibility_matrix[candidate_machines, task] = True

    # Ensure that each machine is assigned at least one task
    for machine in range(m):
        if not np.any(eligibility_matrix[machine]):
            random_task = np.random.choice(range(n))
            eligibility_matrix[machine, random_task] = True

    return eligibility_matrix

def create_processing_times(m, n, processing_time_gap, eligibility_matrix):

    """
    This function generates random processing times, including dummy tasks (0 and n+1)
    
    Args:
        m (int): Number of machines.
        n (int): Number of real tasks.
        processing_time_gap (tuple): Value-gap of processing times.
        eligibility_matrix (np.ndarray): Eligibility matrix.
        
    Returns:
        processing_times (np.ndarray): (m,n+2) size processing-times matrix.
    """
    
    processing_times = np.random.randint(processing_time_gap[0], processing_time_gap[1], size=(m,n+2))
    
    # Set the processing times for dummy-tasks to 0
    processing_times[:,0] = 0
    processing_times[:,-1] = 0

    for machine in range(m):
        for task in range(1,n+1):
            if not eligibility_matrix[machine, task-1]:
                processing_times[machine, task] = 0

    return processing_times

def create_configuration_times(m, n, configuration_time_gap, eligibility_matrix):

    """
    This function generates random configuration times, including dummy tasks (0 and n+1)
    
    Args:
        m (int): Number of machines.
        n (int): Number of real tasks.
        configuration_time_gap (tuple): Value-gap for configuration times.
        eligibility_matrix (np.ndarray): Eligibility matrix. 
        
    Returns:
        configuration_times (np.ndarray): (m,n+2,n+2) size configuration time matrix.
    """
    
    configuration_times = np.random.randint(configuration_time_gap[0], configuration_time_gap[1], size=(m, n+2, n+2))

    for machine in range(m):
        # A task cannot be its own successor, so set the diagonals to 0
        np.fill_diagonal(configuration_times[machine], 0)

        # If a machine is not capable of performing a task, set the configuration times for that task to 0
        for task in range(1,n+1):
            if not eligibility_matrix[machine, task-1]:
                configuration_times[machine][task, :] = 0
                configuration_times[machine][:, task] = 0

        # 0 dummy-task is not any machines's next task
        configuration_times[machine][:, 0] = 0
        # n+1 dummy-task is not any machines's before task
        configuration_times[machine][n+1, :] = 0
        # The configuration time to go to the n+1 dummy-task is 0
        configuration_times[machine][:, n+1] = 0

    return configuration_times

def create_cost_matrix(m, n, configuration_times, processing_times, eligibility_matrix):
    
    """
    This function creates a cost matrix by combining processing times and setup times.    
    
    Args:
        m (int): Number of machines.
        n (int): Number of real tasks.
        configuration_times (np.ndarray): Configuration times matrix.
        processing_times (np.ndarray): Processing times matrix.
        eligibility_matrix (np.ndarray): Eligitibility matrix.
        
    Returns:
        C (np.ndarray): (m,n+2,n+2) size cost matrix.
    """
    
    C = np.zeros((m, n+2, n+2))

    for machine in range(m):
        for task1 in range(n+1):
            for task2 in range(n+2):
                if task1 != task2:    
                    C[machine, task1, task2] = processing_times[machine, task2] + configuration_times[machine, task1, task2]

        for task in range(1,n+1):
            if not eligibility_matrix[machine, task-1]:
                C[machine][task,:]=0                

    return C 

def generator(m, n, processing_time_gap, configuration_time_gap):
    
    """
    This function generates the cost matrix and the eligibility matrix for given parameters.

    Args:
        m (int): Number of machines.
        n (int): Number of real tasks.
        processing_time_gap (tuple): Value gap of processing times.
        configuration_time_gap (tuple): Value gap of configuration times.

    Returns:
        C (tuple): Cost matrix.
    """
    
    # Create eligibility matrix
    eligibility_matrix = create_eligibility_matrix(m, n)
        
    # Generate configuration and processing times
    processing_times = create_processing_times(m, n, processing_time_gap, eligibility_matrix)
    configuration_times = create_onfiguration_times(m, n, configuration_time_gap, eligibility_matrix)
    
    # Create cost matrix
    cost_matrix = create_cost_matrix(m, n, configuration_times, processing_times, eligibility_matrix)
    
    return cost_matrix
