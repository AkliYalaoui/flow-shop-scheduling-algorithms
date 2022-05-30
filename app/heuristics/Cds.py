import numpy as np

class DCS():
    def __init__(self, C):
        self.C = C
        self.nb_jobs = C.shape[0]
        self.nb_machines = C.shape[1]
        self.sequences = {}
        self.memo = {}
        self.solution = None

    def jhonson(self, sub_C):
        if sub_C.shape[1] != 2:
            raise("Jhonson's can only be used on 2 machines")
        l = []
        r = []
        for _ in range(self.nb_jobs):
            # get the min of the matrix
            lin, col = np.unravel_index(sub_C.argmin(), sub_C.shape)
            if col == 0:
                l.append(lin)
            else:
                r.insert(0,lin)
            sub_C[lin, 0] = float('inf')
            sub_C[lin, 1] = float('inf')

        return np.concatenate((l, r))  

    def transform_exec_matrix(self, sequence):
        A = []
        for job in sequence:
            A.append(self.C[int(job), :])

        return np.array(A)   

    def Cij(self, A,task_i,machine_j, memo) :
        if (task_i, machine_j) in memo:
            return memo[(task_i, machine_j)]
        
        if machine_j == 0 :
            s = 0
            for i in range(task_i+1) :
                s += A[i,machine_j]
            memo[(task_i, machine_j)] = s    
            return  s

        if task_i == 0 :
            memo[(task_i, machine_j)] = A[task_i,machine_j] + self.Cij(A,0,machine_j - 1, memo)
            return memo[(task_i, machine_j)]
        
        memo[(task_i, machine_j)] = max(self.Cij(A,task_i - 1,machine_j, memo), self.Cij(A,task_i,machine_j - 1, memo)) + A[task_i,machine_j]
        return memo[(task_i, machine_j)]    


    def get_Cmax(self, sequence):
        new_C = self.transform_exec_matrix(sequence)
        Cmax = self.Cij(new_C, self.nb_jobs - 1, self.nb_machines - 1, self.memo)

        return Cmax  

    def iterate(self, sub_C):
        sequence = self.jhonson(sub_C)
        Cmax = self.get_Cmax(sequence)
        self.sequences[tuple([i+1 for i in sequence])] = Cmax  
        self.memo = {}   

    def get_max_sequence(self):
        c = 0.0
        optimal_seq = None
        items = self.sequences.items()
        for item in items:
            if item[1] >= c:
                optimal_seq = item[0]
                c = item[1]

        return (np.array(optimal_seq).tolist(), int(c))        
    
    def run(self):
        for i in range(self.nb_machines - 1):
            left = np.zeros((self.nb_jobs,))
            right = np.zeros((self.nb_jobs,))
            for s in range(i+1):
                left += self.C[:,s]
                right += self.C[:,-s - 1]
            left = left.T
            right = right.T
            sub_C = np.vstack((left, right))
            sub_C = sub_C.T 
            self.iterate(sub_C)   

        # print("Done")
        self.solution = self.get_max_sequence()
        # print(f"Optimal Sequence : {self.solution[0]} . Cmax = {self.solution[1]}")