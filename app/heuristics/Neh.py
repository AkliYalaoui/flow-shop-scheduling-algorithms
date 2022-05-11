import numpy as np

class FSP_NEH:
    def __init__(self, dist_mat):
        self.dist_mat = dist_mat
        self.ordre_courant = []
        self.cmax = 0
        self.c = None
        
    def init_c(self,j1,j2):
        self.c[0][0] = self.dist_mat[0,j1-1]
        for j in range(1,self.dist_mat.shape[0]):
            self.c[0][j] = self.dist_mat[j,j1-1] + self.c[0][j-1]
        
        self.c[1][0] = self.c[0][0] + self.dist_mat[0,j2-1]
        for j in range(1,self.dist_mat.shape[0]):
            self.c[1][j] = max([self.c[0][j],self.c[1][j-1]]) + self.dist_mat[j,j2-1]
            
    def get_cmax(self,j,pos):
        c = self.c.copy()
        if (pos==0):
            c[0][0] = self.dist_mat[0,j-1]
            for k in range(1,self.dist_mat.shape[0]):
                c[0][k] = self.dist_mat[k,j-1] + c[0][k-1]
        else:
            niveau = pos
            c[niveau][0] = c[niveau-1][0] + self.dist_mat[0,j-1]
            for k in range(1,self.dist_mat.shape[0]):
                c[niveau][k] = max([c[niveau-1][k],c[niveau][k-1]]) + self.dist_mat[k,j-1]
        
        for j in range(pos+1,len(self.ordre_courant)+1):
            niveau = j
            c[niveau][0] = c[niveau-1][0] + self.dist_mat[0,self.ordre_courant[niveau-1]-1]
            for k in range(1,self.dist_mat.shape[0]):
                c[niveau][k] = max([c[niveau-1][k],c[niveau][k-1]]) + self.dist_mat[k,self.ordre_courant[niveau-1]-1]
        return c[len(self.ordre_courant)][self.dist_mat.shape[0]-1], c
                
        
    def iterate(self, j):
        c_max = float('inf')
        position = None
        c = None
        for pos in range(len(self.ordre_courant)+1):
            c_max_pos, c_pos = self.get_cmax(j,pos)
            if c_max_pos < c_max:
                position = pos
                c_max = c_max_pos
                c = c_pos
        self.ordre_courant.insert(position, j)
        self.c = c
        self.cmax = c_max
    
    def run(self):
        # L'ordre initial
        sommes_j = {}
        self.c = np.zeros((self.dist_mat.shape[1],self.dist_mat.shape[0]))
        for j in range(self.dist_mat.shape[1]):
            sommes_j[j+1] = np.sum(self.dist_mat[:,j])
        ordre_init = dict(sorted(sommes_j.items(),key=lambda x:x[1],reverse = True))
        
        ordre_init = list(ordre_init.keys())
        self.ordre_courant.append(ordre_init[0])
        self.ordre_courant.append(ordre_init[1])
        self.init_c(ordre_init[0],ordre_init[1])
        
        for k in range(2,len(ordre_init)):
            self.iterate(ordre_init[k])