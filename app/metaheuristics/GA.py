import random
import timeit
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

def initialize_population(population_size, number_of_jobs):
    population = []
    i = 0
    while i < population_size:
        individual = list(np.random.permutation(number_of_jobs))
        if individual not in population:
            population.append(individual)
            i += 1

    return population


def crossover(parents):
    parent1 = parents[0]
    parent2 = parents[1]
    length_of_parent = len(parent1)
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


def mutation(solution):
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

# Selection 
def select_parent(population, processing_time, number_of_jobs, number_of_machines, method, n_parents):
    parent_pairs = []
    parent_pair_count = n_parents
    if(method == 'tournament'):
        for k in range(parent_pair_count):
            parent1 = binary_tournament(number_of_jobs, number_of_machines, population, processing_time)
            parent2 = binary_tournament(number_of_jobs, number_of_machines, population, processing_time)
            if parent1 != parent2 and (parent1, parent2) not in parent_pairs:
                parent_pairs.append((parent1, parent2))
    elif(method == 'elite'):
        selected_parents = elite_selection(population, processing_time, number_of_jobs, number_of_machines)
        for k in range(parent_pair_count - 1):
            parent_pairs.append(selected_parents[k], selected_parents[k+1])  
    elif(method == 'roulette'):
        for k in range(parent_pair_count):
            parent1 = lottery_selection(population, processing_time, number_of_jobs, number_of_machines)
            parent2 = lottery_selection(population, processing_time, number_of_jobs, number_of_machines)
            if((parent1 != parent2) and (parent1, parent2) not in parent_pairs):
                parent_pairs.append((parent1, parent2))                  
    return parent_pairs

def binary_tournament(number_of_jobs, number_of_machines, population, processing_time):
    parent = []
    candidates = random.sample(population, 2)
    makespan1 = calc_makespan(candidates[0], processing_time, number_of_jobs, number_of_machines)
    makespan2 = calc_makespan(candidates[1], processing_time, number_of_jobs, number_of_machines)
    if makespan1 < makespan2:
        parent = candidates[0]
    else:
        parent = candidates[1]
    return parent

def elite_selection(population, processing_time, number_of_jobs, number_of_machines, n_parents):
    population_sorted = sorted(population, key=lambda x: calc_makespan(x, processing_time, number_of_jobs, number_of_machines), reverse=True)
    return population_sorted[:n_parents]

def lottery_selection(population, processing_time, number_of_jobs, number_of_machines):
    population_sorted = sorted(population, key=lambda x: calc_makespan(x, processing_time, number_of_jobs, number_of_machines))
    total_spans = sum([calc_makespan(x, processing_time, number_of_jobs, number_of_machines) for x in population])
    spans = [calc_makespan(x, processing_time, number_of_jobs, number_of_machines) / total_spans for x in population_sorted]
    cumul_sums = np.cumsum(np.array(spans))
    r =  np.random.rand()
    index = np.argwhere(r <= cumul_sums)
    return population_sorted[index[0][0]]

def update_population(population, children,processing_time, no_of_jobs, no_of_machines):
    costed_population = []
    for individual in population:
        ind_makespan = (calc_makespan(individual, processing_time, no_of_jobs, no_of_machines), individual)
        costed_population.append(ind_makespan)
    costed_population.sort(key=lambda x: x[0], reverse=True)

    costed_children = []
    for individual in children:
        ind_makespan = (calc_makespan(individual, processing_time, no_of_jobs, no_of_machines), individual)
        costed_children.append(ind_makespan)
    costed_children.sort(key=lambda x: x[0])
    for child in costed_children:
        if child not in population:
            population.append(individual)
            population.remove(costed_population[0][1])
            break


def run_ga(n_iter, n_pop, p_c, p_m, method, n_parents):
    # Optimal Cmax for the test instances
    filename = "data.txt"
    file = open(filename, 'r')
    line = file.readline().split()

    # number of jobs and machines
    no_of_jobs, no_of_machines = int(line[0]), int(line[1])

    # i-th job's processing time at j-th machine 
    processing_time = []

    for i in range(no_of_jobs):
        temp = []
        line = file.readline().split()
        for j in range(no_of_machines):
            temp.append(int(line[2 * j + 1]))
        processing_time.append(temp)
    #print(processing_time)

    # generate an initial population proportional to no_of_jobs
    number_of_population = n_pop #no_of_jobs**2
    no_of_iterations = n_iter
    p_crossover = p_c
    p_mutation = p_m

    # Initialize population
    population = initialize_population(number_of_population, no_of_jobs)

    # Start time for CPU calculation
    start_time = timeit.default_timer()

    for _ in range(no_of_iterations):
        # Select parents
        parent_list = select_parent(population, processing_time, no_of_jobs, no_of_machines, method, n_parents)
        childs = []

        # Apply crossover to generate children
        for parents in parent_list:
            r = np.random.rand()
            if r < p_crossover:
                childs.append(crossover(parents))
            else:
                if r < 0.5:
                    childs.append(parents[0])
                else:
                    childs.append(parents[1])

        # Apply mutation operation to change the order of the n-jobs
        mutated_childs = []
        for child in childs:
            r = np.random.rand()
            if r < p_mutation:
                mutated_child = mutation(child)
                mutated_childs.append(mutated_child)

        childs.extend(mutated_childs)
        if len(childs) > 0:
            update_population(population, childs, processing_time, no_of_jobs, no_of_machines)

    # End time for CPU calculation        
    end_time = timeit.default_timer()

    costed_population = []
    for individual in population:
        ind_makespan = (calc_makespan(individual, processing_time, no_of_jobs, no_of_machines), individual)
        costed_population.append(ind_makespan)
    costed_population.sort(key=lambda x: x[0])

    bestObjective = costed_population[0][0]
    exec_time = end_time - start_time

    solution = {'Cmax': bestObjective, 'execTime': exec_time, 'sequence': costed_population[0][1]}
    return solution