import numpy as np

class PalmerHeuristic :
    
    def __init__(self, dist_mat,nb_m, jobs):
        self.dist_mat = dist_mat
        self.c = None
        self.ordre_opt = []
        self.M = 0
        self.nb_machines = nb_m
        self.nb_jobs = jobs
        self.weights = None
 

    def init_weights(self):      #middle machine has a weight of 0 and it increases/decreases toward last/first machine at same rate (Example for 3 machines: the weights are w1 =-2 w2 =0 w3 =+2)
        lst = np.array([ (2*i - 1 - self.nb_machines) for i in range(self.nb_machines)])
        return lst - np.mean(lst)
   

    def compute_weighted_sum(self,weights):       #Computing weighted sum for each job (Example for j1: w1*d11 + w2*d12 + w3*d13)
        weighted_sum = []
        for i in range(self.nb_jobs):
            somme = np.dot(self.dist_mat[:,i],weights)
            weighted_sum.append(somme)
        return np.array(weighted_sum)
   

    def sum_per_machine(self):      #Find the sum of time of each machine
        return self.dist_mat.sum(axis=1)
                                 
    
    def update_c(self, c, ordre):         # Update the computation time to get the makespan
        c[0][0] = self.dist_mat[0,ordre[0]]
        for j in range(1,self.dist_mat.shape[0]):
            c[0][j] = self.dist_mat[j,ordre[0]] + c[0][j-1]
        niveau = len(ordre)        
        for i in range(1,niveau):
            for j in range(0,self.dist_mat.shape[0]):
                c[i][j] = max([c[i-1][j],c[i][j-1]]) + self.dist_mat[j,ordre[i]]
        return c

    
    def get_cmax(self,ordre_opt):               # Cmax : makespan
        return self.c[len(ordre_opt)-1][self.dist_mat.shape[0]-1]
    
    
    def lower_bound(self):                     #Define the lower bound (upperbound is the cmax found)
        lb = []
        a = self.sum_per_machine()
        for i in range(self.nb_machines):
            if i == 0:
                bound = a[i] + min(self.dist_mat.T[:,1:].sum(axis=1))
            
            elif i == (self.nb_machines -1) :
                bound = a[i] + min(self.dist_mat.T[:,:i].sum(axis=1)) 
            
            else :
                bound = a[i] + min(self.dist_mat.T[:,:i].sum(axis=1)) + min(self.dist_mat.T[:,i+1])
            
            lb.append(bound)
        return max(lb)
    

    def run(self):
        self.weights = self.init_weights()
        C= np.zeros((self.dist_mat.shape[1],self.dist_mat.shape[0]))
        
        a = self.compute_weighted_sum(self.weights)

        self.ordre_opt = np.argsort(a)[::-1][:]

        self.c = C
        self.update_c(self.c,self.ordre_opt)
        self.M = self.get_cmax(self.ordre_opt)
                