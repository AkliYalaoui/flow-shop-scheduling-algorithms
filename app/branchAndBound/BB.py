import numpy as np

class BrandAndBound:
    def __init__(self, dist_mat):
        self.dist_mat = dist_mat
        self.M = float('inf')
        self.ordre_opt = []
        self.c = None
    
    def update_c(self, c, ordre):
        if len(ordre) == 1:
            c[0][0] = self.dist_mat[0,ordre[0]-1]
            for j in range(1,self.dist_mat.shape[0]):
                c[0][j] = self.dist_mat[j,ordre[0]-1] + c[0][j-1]
        else:
            niveau = len(ordre)-1
            c[niveau][0] = c[niveau-1][0] + self.dist_mat[0,ordre[niveau]-1]
            for j in range(1,self.dist_mat.shape[0]):
                c[niveau][j] = max([c[niveau-1][j],c[niveau][j-1]]) + self.dist_mat[j,ordre[niveau]-1]
        return c
    
    def get_c_max(self, c, niveau):
        return c[niveau][self.dist_mat.shape[0]-1]
    
    def evaluate(self, C, ordre, tache_restante):
        lb_max = float('-inf')
        niveau = len(ordre) - 1
        for j in range(self.dist_mat.shape[0]):
            lb = C[niveau][j]
            min_temps_exec_derniere_tache = float('inf')
            for tache in tache_restante:
                lb += self.dist_mat[j][tache-1]
                temps_exec_derniere_tache = 0
                for j_prime in range(j+1,self.dist_mat.shape[0]):
                    temps_exec_derniere_tache += self.dist_mat[j_prime][tache-1]
                if temps_exec_derniere_tache < min_temps_exec_derniere_tache:
                    min_temps_exec_derniere_tache = temps_exec_derniere_tache
            lb += min_temps_exec_derniere_tache
            
            if lb > lb_max:
                lb_max = lb
        return lb_max
    
    def iterate(self, ordre, tache_restante, C):
        for i in tache_restante:
            # init
            new_ordre = ordre.copy()
            new_tache_restante = tache_restante.copy()
            # update
            new_ordre.append(i)
            new_tache_restante.remove(i)
            C_new = self.update_c(C,new_ordre)
            # 
            if len(new_tache_restante) == 0:
                c_max = self.get_c_max(C_new, len(new_ordre)-1)
                if c_max < self.M:
                    self.M = c_max
                    # print("c max: ",c_max)
                    self.ordre_opt = new_ordre
                    self.c = C_new
            else:
                evaluation = self.evaluate(C_new, new_ordre, new_tache_restante)
                # elagage
                if evaluation < self.M:
                    self.iterate(new_ordre, new_tache_restante, C_new)
    
    def run(self):
        ordre = []
        tache_rest = list(range(1,len(self.dist_mat[0])+1))
        C= np.zeros((self.dist_mat.shape[1],self.dist_mat.shape[0]))
        self.c = C
        self.iterate(ordre, tache_rest, C)

