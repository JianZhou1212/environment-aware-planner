import cvxpy as cp
import numpy as np

class Initial_Set_Estimation( ):
    def __init__(self, Params):
        self.N_Sam = Params['N_pre_sam']
        self.nu    = Params['nu']

    def Return(self, samples):
        A_initial, b_initial = self.SemiDefiProgramming(samples)
        b_initial = b_initial
        Q_initial = np.linalg.inv(A_initial.T@A_initial)
        q_initial = -np.linalg.inv(A_initial)@b_initial

        return A_initial, b_initial, Q_initial, q_initial
    
    def SemiDefiProgramming(self, samples): 
        nu    = self.nu
        N_Sam = self.N_Sam

        A = cp.Variable((nu, nu), PSD=True)
        b = cp.Variable((nu, 1))
        
        constraints = [A - 1e-8*np.eye(nu) >> 0]  
        for i in range(N_Sam):
            constraints += [cp.norm(A@cp.reshape(samples[i], (2, 1)) + b, 2) <= 1]
        objective = cp.Maximize(cp.log_det(A))
        
        problem = cp.Problem(objective, constraints)
        problem.solve(solver = cp.MOSEK) # verbose=True

        return A.value, b.value