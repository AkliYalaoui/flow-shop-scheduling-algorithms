import numpy as np

def calc_makespan(solution, proccessing_time, number_of_jobs, number_of_machines):
    # list for the time passed until the finishing of the job
    cost = [0] * number_of_jobs
    # for each machine, total time passed will be updated
    for machine_no in range(0, number_of_machines):
        for slot in range(number_of_jobs):
            # time passed so far until the task starts to process
            cost_so_far = cost[slot]
            if slot > 0:
                cost_so_far = max(cost[slot - 1], cost[slot])
            cost[slot] = cost_so_far + proccessing_time[solution[slot]][machine_no]
    return cost[number_of_jobs - 1]

# Heuristic 1 

def mutation(solution,prob_mut=.2):
    prob = np.random.uniform()
    if prob > prob_mut : 
        return solution
    
    # copy the solution
    mutated_solution = list(solution)
    solution_length = len(solution)
    # pick 2 positions to swap randomly
    swap_positions = list(np.random.permutation(np.arange(solution_length))[:2])
    first_job = solution[swap_positions[0]]
    second_job = solution[swap_positions[1]]
    mutated_solution[swap_positions[0]] = second_job
    mutated_solution[swap_positions[1]] = first_job
    return mutated_solution


# Heuristic 2

def crossover(parent1,prob_cro=.8):
    prob = np.random.uniform()
    if prob > prob_cro : 
        return parent1
    
    length_of_parent = len(parent1)
    parent2 = np.random.permutation(length_of_parent)
    first_point = int(length_of_parent / 2 - length_of_parent / 4)
    second_point = int(length_of_parent - first_point)
    intersect = parent1[first_point:second_point]

    child = []
    index = 0
    for pos2 in range(len(parent2)):
        if first_point <= index < second_point:
            child.extend(intersect)
            index = second_point
        if parent2[pos2] not in intersect:
                child.append(parent2[pos2])
                index += 1

    return child


#Heuristic 3

def n_mutation(solution,prob_mut=.2,n=10) : 
    for _ in range(n) : 
        solution = mutation(solution,prob_mut)
    return solution

# heuristic 4 
def rz(solution,Cmax) : 
    
    makespan = Cmax(solution)
    n = len(solution)

#     d = {}
#     for swap in combinations(range(n), 2):
#         candidate_neighbord = solution.copy()
#         val = candidate_neighbord[swap[0]]
#         del candidate_neighbord[swap[0]]
#         candidate_neighbord.insert(swap[1], val)
#         candidate_neighbord_val = Cmax(candidate_neighbord)
#         d[swap] = candidate_neighbord_val

#     move, fun = min(d.items(), key=lambda x: x[1])
#     sol = solution.copy()
#     sol[move[0]] = solution[move[1]]
#     sol[move[1]] = solution[move[0]]
    
#     if fun < makespan : 
#         return sol,fun
#     return solution,makespan

    for j in range(n) :
        candidate = solution
        job_j = candidate[j]
        # Remove Job J
        del candidate[j]
        
        # Test job j in all positions except in its original position
        best_candidate_val = float("inf")
        new_j = j
        
        for i in range(n) :
            if i != j :
                candidate_neighbord = candidate.copy()
                candidate_neighbord.insert(i, job_j)
                candidate_neighbord_val = Cmax(candidate_neighbord)

                if candidate_neighbord_val < best_candidate_val :
                    new_j = i
                    best_candidate_val = candidate_neighbord_val
        
        # Insert job j in the position that minimize the most the makespan
        candidate.insert(new_j, job_j)
        candidate_val = Cmax(candidate)
        
        if candidate_val < makespan :
            makespan = candidate_val
            solution = candidate
    
    return solution,makespan

def local_search(heuristic,Cmax,solution) : 
    
    makespan = Cmax(solution)
    solution = heuristic(solution)
    
    return rz(solution,Cmax)

def heuristic_fsp(heuristic_name,C,initial_sol) :
    """
    Desc : 
        Run the heuristic_nmae on the given problem instance C
    
    Args   :
        heuristic_name : string
        C : 2D numpy array
        initial_sol : list
        
    Return : 
        solution : list
        makespan : float
    """
    solution = []
    makespan = 0
    m = C.shape[0]
    j = C.shape[1]
    
    def Cmax(solution) : 
        return calc_makespan(solution, C.T, j, m)
    
    heuristics = ["h1","h2","h3","h4"]
    
    assert heuristic_name in heuristics, "Invalide Heuristic Name"
    
    if heuristic_name == "h1" :
        solution,makespan = local_search(mutation,Cmax,initial_sol)
        
    elif heuristic_name == "h2" :
        solution,makespan = local_search(crossover,Cmax,initial_sol)
        
    elif heuristic_name == "h3" :
        solution,makespan = local_search(n_mutation,Cmax,initial_sol)
        
    elif heuristic_name == "h4" :
        solution,makespan = rz(initial_sol,Cmax)

    return solution,makespan


def QLHH(data,initial_sol,initial_fun,maxiter=100,n_episodes=1000,ep=.5,lr=.1,gamma=0.99,exploration_proba=1,min_exploration_proba=.01,exploration_decreasing_decay=0.001) : 
    initial_sol = list(initial_sol)
    heuristics = ["h1","h2","h3","h4"]
    Q_table = np.zeros((len(heuristics),len(heuristics)))
    
    current_state = np.random.randint(len(heuristics))
    state = current_state
    sol, fun = heuristic_fsp(heuristics[current_state],data,initial_sol)

    total_episode_reward = 0
    
    for e in range(n_episodes) :
        if np.random.uniform(0,1) < exploration_proba:
            action = np.random.randint(len(heuristics))
        else :
            action = np.argmax(Q_table[current_state,:])

        old_fun = fun
        next_state = action

        sol, fun = heuristic_fsp(heuristics[action],data,sol)
        reward = 1 if fun - old_fun > 0 else 0
        total_episode_reward = total_episode_reward + reward
        
        Q_table[current_state, action] = (1-lr) * Q_table[current_state, action] +lr*(reward + gamma*max(Q_table[next_state,:]))
        exploration_proba = max(min_exploration_proba, np.exp(-exploration_decreasing_decay*e))
        current_state = action
        
    print(Q_table)
    # print(total_episode_reward)
    
    sol, fun =  initial_sol,initial_fun
    best_sol,best_fun = sol, fun 
#     obj_values = [fun]
    it = 0
    while it < maxiter :  
        
        if np.random.uniform(0,1) < ep:
            action = np.argmax(Q_table[state,:])
        else :
            action = np.random.randint(len(heuristics))
        action = np.argmax(Q_table[state,:])  
        sol, fun = heuristic_fsp(heuristics[action],data,sol)
#         obj_values.append(fun)
        if fun < best_fun : 
            # print(it,fun)
            best_sol = sol
            best_fun = fun
        
        state = action
        it += 1
    
    return best_sol, best_fun