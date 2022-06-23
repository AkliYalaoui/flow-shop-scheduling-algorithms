from app.heuristics.Palmer import PalmerHeuristic
from itertools import combinations
import numpy as np

class Tabu():                     
    # Strategy used: Best move
    # Aspiration criteria: if the move is improving cmax
    # Stopping criteria: max number of iterations reached
    
    # Parameters: 
        # Length of Tabu list (relative to the problem)
        # Number of iterations (more can find better solutions but take more time)
        
    # TODO : Ameliorations Intensification and Diversification
    
    def __init__(self, dist_mat, nb_m, jobs,tabu_tenure=3,niter=100,Verbose=True):
        self.Verbose = Verbose
        self.dist_mat = dist_mat
        self.niter = niter
        self.tabu_tenure = tabu_tenure
        self.Initial_solution = None
        self.Initial_cmax = None
        self.nb_machines = nb_m
        self.c = None
        self.nb_jobs = jobs
        self.fitness_function_values = []
        self.tabu_str, self.Best_solution, self.Best_objvalue = self.run()

    def get_tabuestructure(self):
        dict = {}
        for swap in combinations(range(self.nb_jobs), 2):
            dict[swap] = {'tabu_time': 0, 'MoveValue': 0}
        return dict

    
    def get_InitialSolution(self):
        b = PalmerHeuristic(self.dist_mat,self.nb_machines,self.nb_jobs)
        b.run()
        return b.ordre_opt.tolist()


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
    
    
    def SwapMove(self, solution, i ,j):
        solution = solution.copy()
        # job index in the solution:
        i_index = solution.index(i)
        j_index = solution.index(j)
        #Swap
        solution[i_index], solution[j_index] = solution[j_index], solution[i_index]
        return solution

    
    def run(self):
        C= np.zeros((self.dist_mat.shape[1],self.dist_mat.shape[0])) 
        self.c = C
        tenure =self.tabu_tenure
        self.Initial_solution =self.get_InitialSolution()
        tabu_structure = self.get_tabuestructure()  # Initialize the data structures
        best_solution = self.Initial_solution
        best_objvalue = self.Objfun(best_solution)
        self.fitness_function_values.append(best_objvalue)
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
                candidate_solution = self.SwapMove(current_solution, move[0], move[1])
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
                    current_solution = self.SwapMove(current_solution, best_move[0], best_move[1])
                    current_objvalue = self.Objfun(current_solution)
                    # Best Improving move
                    if MoveValue < best_objvalue:
                        best_solution = current_solution
                        best_objvalue = current_objvalue
                        self.fitness_function_values.append(best_objvalue)
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
                        current_solution = self.SwapMove(current_solution, best_move[0], best_move[1])
                        current_objvalue = self.Objfun(current_solution)
                        best_solution = current_solution
                        best_objvalue = current_objvalue
                        self.fitness_function_values.append(best_objvalue)
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