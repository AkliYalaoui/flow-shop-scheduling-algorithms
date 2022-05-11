import numpy as np
from itertools import combinations
import collections

class PalmerHeuristic :
    
    def __init__(self, dist_mat,nb_m, jobs):
        self.dist_mat = dist_mat
        self.c = None
        self.ordre_opt = []
        self.nb_machines = nb_m
        self.nb_jobs = jobs
        self.weights = None
 
    def init_weights(self):      #middle machine has a weight of 0 and it increases/decreases toward last/first machine at same rate (Example for 3 machines: the weights are w1 =-2 w2 =0 w3 =+2)
        lst = np.array([ (2*i - 1 - self.nb_machines) for i in range(self.nb_machines)])
        return lst - np.mean(lst)
   
    def compute_weighted_sum(self):       #Computing weighted sum for each job (Example for j1: w1*d11 + w2*d12 + w3*d13)
        weighted_sum = []
        for i in range(self.nb_jobs):
            somme = np.dot(self.dist_mat[:,i],self.weights)
            weighted_sum.append(somme)
        return np.array(weighted_sum)
   
    def sum_per_machine(self):      #Find the sum of time of each machine
        return self.dist_mat.sum(axis=1)
                                
    def run(self):
        self.weights = self.init_weights()
        C= np.zeros((self.dist_mat.shape[1],self.dist_mat.shape[0]))       
        self.c = C
        a = self.compute_weighted_sum()
        self.ordre_opt = np.argsort(a)[::-1][:]
        return self.ordre_opt.tolist() 


class Tabu():                     
    # neighborhoods available: 1:swap two consecutive 2:exchange jobs at positions i and j  
    # 3: remove job at position i and insert it at position j
    def __init__(self, dist_mat, nb_m, jobs,initial_solution,neighboorhood=2,tabu_tenure=3,niter=100,Verbose=False):
        self.Verbose = Verbose
        self.dist_mat = dist_mat
        self.niter = niter
        self.tabu_tenure = tabu_tenure
        self.neighoorhood = neighboorhood
        self.Initial_solution = initial_solution
        self.Initial_cmax = None
        self.nb_machines = nb_m
        self.c = None
        self.nb_jobs = jobs
        self.tabu_str, self.Best_solution, self.Best_objvalue = self.run()

    def get_tabuestructure(self):
        dict = {}
        if self.neighoorhood == 1:
            for swap in range(self.nb_jobs-1):
                dict[swap] = {'tabu_time': 0, 'MoveValue': 0}
        else:
            for swap in combinations(range(self.nb_jobs), 2):
                dict[swap] = {'tabu_time': 0, 'MoveValue': 0}
        
        return dict

    
    def get_InitialSolution(self):
        b = PalmerHeuristic(self.dist_mat,self.nb_machines,self.nb_jobs)
        return b.run()


    def Objfun(self,ordre):               # Cmax : makespan
        c = self.c.copy()
        c[0][0] = self.dist_mat[0,ordre[0]]
        for j in range(1,self.dist_mat.shape[0]):
            c[0][j] = self.dist_mat[j,ordre[0]] + c[0][j-1]
        niveau = len(ordre)-1
        for i in range(1,niveau+1):
            for j in range(0,self.dist_mat.shape[0]):
                c[i][j] = max([c[i-1][j],c[i][j-1]]) + self.dist_mat[j,ordre[i]]
        
        return c[len(self.Initial_solution)-1][self.dist_mat.shape[0]-1]
    
    
    def SwapMove(self, solution, swap):
        solution = solution.copy()
        
        if self.neighoorhood == 1:
            i_index = swap
            j_index = swap+1
            solution[i_index], solution[j_index] = solution[j_index], solution[i_index]
        elif self.neighoorhood == 2:
            i_index = solution.index(swap[0])
            j_index = solution.index(swap[1])
            solution[i_index], solution[j_index] = solution[j_index], solution[i_index]
        else:
            i_index = swap[0]
            j_index = swap[1]
            if i_index < j_index:
                solution.insert(j_index,solution[i_index])
                del solution[i_index]
            else:
                del solution[i_index]
                solution.insert(j_index,solution[i_index])
        return solution

    
    def run(self):
        C= np.zeros((self.dist_mat.shape[1],self.dist_mat.shape[0])) 
        self.c = C
        tenure =self.tabu_tenure
#         self.Initial_solution =self.get_InitialSolution()
        tabu_structure = self.get_tabuestructure()  # Initialize the data structures
        best_solution = self.Initial_solution
        best_objvalue = self.Objfun(best_solution)
        current_solution = self.Initial_solution
        current_objvalue = self.Objfun(current_solution)
        self.Initial_cmax = best_objvalue
        
        if self.Verbose:
            print("initial order "+ str(self.Initial_solution))
            print("initial cmax: "+ str(best_objvalue))


        iter = 1
        Terminate = 0
        while Terminate < self.niter:

            # Searching the whole neighborhood of the current solution:
            for move in tabu_structure:
                candidate_solution = self.SwapMove(current_solution, move)
                candidate_objvalue = self.Objfun(candidate_solution)
                tabu_structure[move]['MoveValue'] = candidate_objvalue

            # Admissible move
            while True:
                # select the move with the lowest ObjValue in the neighborhood (minimization)
                best_move = min(tabu_structure, key =lambda x: tabu_structure[x]['MoveValue'])
                MoveValue = tabu_structure[best_move]["MoveValue"]
                tabu_time = tabu_structure[best_move]["tabu_time"]
                # Not Tabu
                if tabu_time < iter:
                    # make the move
                    current_solution = self.SwapMove(current_solution, best_move)
                    current_objvalue = self.Objfun(current_solution)
                    # Best Improving move
                    if MoveValue < best_objvalue:
                        best_solution = current_solution
                        best_objvalue = current_objvalue
                        #print("   best_move: {}, Objvalue: {} => Best Improving => Admissible".format(best_move,
                          #                                                                            current_objvalue))
                        Terminate = 0
                    else:
                        #print("   ##Termination: {}## best_move: {}, Objvalue: {} => Least non-improving => "
                         #     "Admissible".format(Terminate,best_move,
                          #                                                                                 current_objvalue))
                        Terminate += 1
                    # update tabu_time for the move
                    tabu_structure[best_move]['tabu_time'] = iter + tenure
                    iter += 1
                    break
                # If tabu
                else:
                    # Aspiration
                    if MoveValue < best_objvalue:
                        # make the move
                        current_solution = self.SwapMove(current_solution, best_move)
                        current_objvalue = self.Objfun(current_solution)
                        best_solution = current_solution
                        best_objvalue = current_objvalue
                        #print("   best_move: {}, Objvalue: {} => Aspiration => Admissible".format(best_move,
                      #                                                                                current_objvalue))
                        Terminate = 0
                        iter += 1
                        break
                    else:
                        tabu_structure[best_move]["MoveValue"] = float('inf')
                        #print("   best_move: {}, Objvalue: {} => Tabu => Inadmissible".format(best_move,
                       #                                                                       current_objvalue))
                        continue
        #print('#'*50 , "Performed iterations: {}".format(iter), "Best found Solution: {} , Objvalue: {}".format(best_solution,best_objvalue), sep="\n")
        
        if self.Verbose:
            print("best order: " + str(best_solution))
            print("cmax: "+ str(best_objvalue))
            
        return tabu_structure, best_solution, best_objvalue

class VND():
    def __init__(self, dist_mat, nb_m, jobs, initial_solution,neighborhood=2):
        self.dist_mat = dist_mat
        self.nb_m = nb_m
        self.nb_jobs = jobs
        self.neighborhood = neighborhood
        self.current_solution = initial_solution
        self.best_solution = initial_solution
        self.best_solution_cmax = self.Objfun(self.best_solution)
        
    def Objfun(self,ordre):               # Cmax : makespan
        c = np.zeros((self.dist_mat.shape[1],self.dist_mat.shape[0]))
        c[0][0] = self.dist_mat[0,ordre[0]]
        for j in range(1,self.dist_mat.shape[0]):
            c[0][j] = self.dist_mat[j,ordre[0]] + c[0][j-1]
        niveau = len(ordre)-1
        for i in range(1,niveau+1):
            for j in range(0,self.dist_mat.shape[0]):
                c[i][j] = max([c[i-1][j],c[i][j-1]]) + self.dist_mat[j,ordre[i]]
        
        return c[len(self.current_solution)-1][self.dist_mat.shape[0]-1]
    
    def get_neighborhoodstructure(self):
        dict = {}
        if self.neighborhood == 1:
            for swap in range(self.nb_jobs-1):
                dict[swap] = {'MoveValue': 0}
        elif self.neighborhood == 4:
            for swap in range(self.nb_jobs-1):
                dict[(swap,'D')] = {'MoveValue': 0}
            for swap in range(self.nb_jobs-1):
                dict[(swap,'G')] = {'MoveValue': 0}
        else:
            for swap in combinations(range(self.nb_jobs), 2):
                dict[swap] = {'MoveValue': 0}
        
        return dict
    
    def SwapMove(self, solution, swap):
        solution = solution.copy()
        
        if self.neighborhood == 1:
            i_index = swap
            j_index = swap+1
            solution[i_index], solution[j_index] = solution[j_index], solution[i_index]
        elif self.neighborhood == 4:
            a_list = collections.deque(solution)
            if swap[1]=='D':
                a_list.rotate(swap[0])
            else:
                a_list.rotate(-swap[0])
            solution = list(a_list)
        elif self.neighborhood == 2:
            i_index = solution.index(swap[0])
            j_index = solution.index(swap[1])
            solution[i_index], solution[j_index] = solution[j_index], solution[i_index]
        else:
            i_index = swap[0]
            j_index = swap[1]
            if i_index < j_index:
                solution.insert(j_index,solution[i_index])
                del solution[i_index]
            else:
                del solution[i_index]
                solution.insert(j_index,solution[i_index])
        return solution
    
    def run(self):
        neighohood_structure = self.get_neighborhoodstructure()
        
        while True:
            for move in neighohood_structure:
                candidate_solution = self.SwapMove(self.current_solution, move)
                candidate_objvalue = self.Objfun(candidate_solution)
                neighohood_structure[move]['MoveValue'] = candidate_objvalue
            best_move = min(neighohood_structure, key =lambda x: neighohood_structure[x]['MoveValue'])
            MoveValue = neighohood_structure[best_move]["MoveValue"]
            self.current_solution = self.SwapMove(self.current_solution, best_move)
            if MoveValue < self.best_solution_cmax:
                self.best_solution = self.current_solution
                self.best_solution_cmax = MoveValue
            else:
                break

class GVNS():
    
    def __init__(self, dist_mat, nb_m, jobs,tabu_tenure=3,niter=100):
        self.dist_mat = dist_mat
        self.nb_m = nb_m
        self.nb_jobs = jobs
        self.niter = niter
        self.tabu_tenure = tabu_tenure
        self.current_solution = None
        self.best_solution = None
        self.best_solution_cmax = None
        
    def get_InitialSolution(self):
        b = PalmerHeuristic(self.dist_mat,self.nb_m,self.nb_jobs)
        b.run()
        return b.ordre_opt.tolist()
    
    def Objfun(self,ordre):               # Cmax : makespan
        c = np.zeros((self.dist_mat.shape[1],self.dist_mat.shape[0]))
        c[0][0] = self.dist_mat[0,ordre[0]]
        for j in range(1,self.dist_mat.shape[0]):
            c[0][j] = self.dist_mat[j,ordre[0]] + c[0][j-1]
        niveau = len(ordre)-1
        for i in range(1,niveau+1):
            for j in range(0,self.dist_mat.shape[0]):
                c[i][j] = max([c[i-1][j],c[i][j-1]]) + self.dist_mat[j,ordre[i]]
        
        return c[len(self.current_solution)-1][self.dist_mat.shape[0]-1]
    
    def run(self):
        self.current_solution = self.get_InitialSolution()
        self.best_solution = self.current_solution
        self.best_solution_cmax = self.Objfun(self.best_solution)
        for k in range(1,4):
            tabu_search = Tabu(self.dist_mat ,self.nb_m ,self.nb_jobs ,self.best_solution ,neighboorhood=k, tabu_tenure=self.tabu_tenure,niter=self.niter)
            s_prime = tabu_search.Best_solution
            s_prime_cmax = tabu_search.Best_objvalue
            
            for j in range(1,4):
                vnd_search = VND(self.dist_mat ,self.nb_m ,self.nb_jobs ,s_prime ,neighborhood=j)
                s_prime2 = vnd_search.best_solution
                s_prime2_cmax = vnd_search.best_solution_cmax
                if s_prime2_cmax < self.best_solution_cmax:
                    self.best_solution = s_prime2
                    self.best_solution_cmax = s_prime2_cmax
                