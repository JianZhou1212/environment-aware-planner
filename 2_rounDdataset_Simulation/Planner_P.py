import numpy as np
import cvxpy as cp
import mosek
import casadi
from scipy.linalg import sqrtm, svd
import time
import math

class Planner_P( ):
    def __init__(self, Params):
        
        self.T = Params['T']
        self.N = Params['N']
        self.l_f = Params['l_f']
        self.l_r = Params['l_r']
        self.DEV = Params['DEV']
        self.A_SV = Params['A_SV']
        self.B_SV = Params['B_SV']
        self.nx = Params['nx']
        self.nu = Params['nu']
        self.du = Params['du']
        self.Q1 = Params['Q1']
        self.Q2 = Params['Q2']
        self.Q3 = Params['Q3']
        self.v_low = Params['v_low']
        self.v_up  = Params['v_up']
        self.acc_low    = Params['acc_low']
        self.acc_up     = Params['acc_up']
        self.delta_low  = Params['delta_low']
        self.delta_up   = Params['delta_up']
        self.x_track_sv = Params['x_track_sv']
        self.y_track_sv = Params['y_track_sv']
        self.x_track_ev = Params['x_track_ev']
        self.y_track_ev = Params['y_track_ev']
        self.num_points_sv = Params['num_points_sv']
        self.Q_measure = Params['Q_measure']
        self.Q_safedis = Params['Q_safedis']
        self.L_State = Params['L_State']
        self.L_Occ   = Params['L_Occ']
        self.H       = Params['H']
        self.h       = Params['h']
        self.MPCFormulation = self.MPCFormulation( )
        self.SVMPCFormulation = self.SVMPCFormulation( )
    
    def SVPrediction(self, current_x_SV):
        
        distances = np.sqrt((self.x_track_sv - current_x_SV[0])**2 + (self.y_track_sv - current_x_SV[2])**2)
        min_distance_index = np.argmin(distances)

        index_ref = list(range(min_distance_index + 1, min_distance_index + self.N + 1))
        index_ref = [num if num <= (self.num_points_sv-1) else (self.num_points_sv-1) for num in index_ref]

        x_ref_k = self.x_track_sv[index_ref]
        y_ref_k = self.y_track_sv[index_ref]
        
        SVPredictionTrajectory_k = self.SVMPCFormulation(current_x_SV, x_ref_k, y_ref_k)
        SVPredictionTrajectory_k = SVPredictionTrajectory_k.full( )
        
        return SVPredictionTrajectory_k 

    def Minksum_IA(self, q1, Q1, q2, Q2, L):
        q = q1 + q2 
        
        v = sqrtm(Q2)@L
        w = sqrtm(Q1)@L
        v = v/np.linalg.norm(v)
        w = w/np.linalg.norm(w)
        
        Uv, _, Vv = svd(v)  
        Uw, _, Vw = svd(w)  
        
        S2 = Uw*Vw*Vv@Uv.T
        Q = (sqrtm(Q1) + S2@sqrtm(Q2)).T@(sqrtm(Q1) + S2@sqrtm(Q2))
        
        return q, Q

    def Minksum_EA(self, q1, Q1, q2, Q2, L):
        q       = q1 + q2
        Q_Left  = np.sqrt(L.T@Q1@L) + np.sqrt(L.T@Q2@L)
        Q_Right = Q1/np.sqrt(L.T@Q1@L) + Q2/np.sqrt(L.T@Q2@L)
        Q       = Q_Left*Q_Right
        
        return q, Q

    def ReachableSet(self, current_x_SV, Q_k, q_k):
        N = self.N
        nx = self.nx
        du = self.du
        Q_measure = self.Q_measure
        Q_safedis = self.Q_safedis
        A_SV        = self.A_SV
        B_SV        = self.B_SV
        L_State     = self.L_State
        L_Occ       = self.L_Occ
        
        SVPredictionTrajectory_k = self.SVPrediction(current_x_SV)
        print
        x_pre_sv_k  = SVPredictionTrajectory_k[0, :]
        vx_pre_sv_k = SVPredictionTrajectory_k[1, :]
        y_pre_sv_k  = SVPredictionTrajectory_k[2, :]
        vy_pre_sv_k = SVPredictionTrajectory_k[3, :]
        
        q_Bu = B_SV@q_k
        Q_Bu = B_SV@Q_k@B_SV.T + du*np.eye(nx)

        Q_Inv = np.zeros((2, 2*N)) 
        q = np.zeros((2, N))
        Reachable_Set_Q = list( )  # exact SV reachable set
        Reachable_Set_q = list( )  # exact SV reachable set

        Reachable_Set_q.append(current_x_SV.reshape(4, 1))
        Reachable_Set_Q.append(Q_measure)
        for t in range(1, N + 1):
            Q_t_1    = Reachable_Set_Q[t-1]
            q_t_1    = Reachable_Set_q[t-1]
            q_t, Q_t = self.Minksum_EA(A_SV@q_t_1, A_SV@Q_t_1@A_SV.T, q_Bu, Q_Bu, L_State)
            
            Reachable_Set_Q.append(Q_t)
            Reachable_Set_q.append(q_t)
            Q_t_reach   = Q_t[:, [0, 2]]
            Q_t_reach   = Q_t_reach[[0, 2], :]
            q_t_reach   = np.array([[x_pre_sv_k[t]], [y_pre_sv_k[t]]])
            _, Q_t      = self.Minksum_EA(q_t_reach, Q_t_reach, np.zeros((2, 1)), Q_safedis, L_Occ)
            heading_ref = math.atan2(vy_pre_sv_k[t], vx_pre_sv_k[t])
            Q_t_inv     = np.linalg.inv(Q_t)
            eigenvalues, eigenvectors = np.linalg.eig(Q_t_inv)
            major_axis_vector = eigenvectors[:, np.argmin(eigenvalues)]
            heading_current   = np.arctan2(major_axis_vector[1], major_axis_vector[0])

            heading  = heading_ref - heading_current
            rotation = np.array([[np.cos(heading), -np.sin(heading)], [np.sin(heading), np.cos(heading)]])
            
            Q_t_inverse         = rotation@np.linalg.inv(Q_t)@rotation.T
            Q_Inv[:, 2*t-2:2*t] = Q_t_inverse
            q[:, t-1]           = q_t_reach.reshape(2, )
        return Q_Inv, q, x_pre_sv_k, y_pre_sv_k
    
    def Return(self, current_x_SV, current_x_EV, A_k_1, b_k_1, u_k_1):
        current_x_SV0 = current_x_SV[0]
        current_x_SV1 = current_x_SV[1]
        current_x_SV2 = current_x_SV[2]
        A0_k_1 = A_k_1[0]
        b0_k_1 = b_k_1[0]
        u0_k_1 = u_k_1[0]
        A1_k_1 = A_k_1[1]
        b1_k_1 = b_k_1[1]
        u1_k_1 = u_k_1[1]
        A2_k_1 = A_k_1[2]
        b2_k_1 = b_k_1[2]
        u2_k_1 = u_k_1[2]
        
        current_x_SV0 = np.array([current_x_SV0[0], current_x_SV0[3]*np.cos(current_x_SV0[2]), current_x_SV0[1], current_x_SV0[3]*np.sin(current_x_SV0[2])])
        current_x_SV1 = np.array([current_x_SV1[0], current_x_SV1[3]*np.cos(current_x_SV1[2]), current_x_SV1[1], current_x_SV1[3]*np.sin(current_x_SV1[2])])
        current_x_SV2 = np.array([current_x_SV2[0], current_x_SV2[3]*np.cos(current_x_SV2[2]), current_x_SV2[1], current_x_SV2[3]*np.sin(current_x_SV2[2])])
   
        if np.linalg.norm((A0_k_1@u0_k_1 + b0_k_1), 2) <= 1:
            A0_k  = A0_k_1
            b0_k  = b0_k_1
            Qu0_k = np.linalg.inv(A0_k.T @ A0_k)
            qu0_k = -np.linalg.inv(A0_k)@b0_k 
        else:
            A0_last = A0_k_1.T@A0_k_1
            b0_last = A0_k_1.T@b0_k_1
            c0_last = b0_k_1.T@b0_k_1 - 1
            B0_k, b0_tilde_k = self.SemiDefiProgramming(A0_last, b0_last.reshape(2, 1), c0_last, u0_k_1)
            A0_k  = sqrtm(B0_k)
            b0_k  = np.linalg.inv(A0_k)@b0_tilde_k
            Qu0_k = np.linalg.inv(A0_k.T@A0_k)
            qu0_k = -np.linalg.inv(A0_k)@b0_k  

        if np.linalg.norm((A1_k_1@u1_k_1 + b1_k_1), 2) <= 1:
            A1_k  = A1_k_1
            b1_k  = b1_k_1
            Qu1_k = np.linalg.inv(A1_k.T @ A1_k)
            qu1_k = -np.linalg.inv(A1_k)@b1_k 
        else:
            A1_last = A1_k_1.T@A1_k_1
            b1_last = A1_k_1.T@b1_k_1
            c1_last = b1_k_1.T@b1_k_1 - 1
            B1_k, b1_tilde_k = self.SemiDefiProgramming(A1_last, b1_last.reshape(2, 1), c1_last, u1_k_1)
            A1_k  = sqrtm(B1_k)
            b1_k  = np.linalg.inv(A1_k)@b1_tilde_k
            Qu1_k = np.linalg.inv(A1_k.T@A1_k)
            qu1_k = -np.linalg.inv(A1_k)@b1_k

        if np.linalg.norm((A2_k_1@u2_k_1 + b2_k_1), 2) <= 1:
            A2_k  = A2_k_1
            b2_k  = b2_k_1
            Qu2_k = np.linalg.inv(A2_k.T @ A2_k)
            qu2_k = -np.linalg.inv(A2_k)@b2_k 
        else:
            A2_last = A2_k_1.T@A2_k_1
            b2_last = A2_k_1.T@b2_k_1
            c2_last = b2_k_1.T@b2_k_1 - 1
            B2_k, b2_tilde_k = self.SemiDefiProgramming(A2_last, b2_last.reshape(2, 1), c2_last, u2_k_1)
            A2_k  = sqrtm(B2_k)
            b2_k  = np.linalg.inv(A2_k)@b2_tilde_k
            Qu2_k = np.linalg.inv(A2_k.T@A2_k)
            qu2_k = -np.linalg.inv(A2_k)@b2_k
            
        Qr0_Inv, qr0,  x_pre_sv0_k, y_pre_sv0_k = self.ReachableSet(current_x_SV0, Qu0_k, qu0_k)
        Qr1_Inv, qr1,  x_pre_sv1_k, y_pre_sv1_k = self.ReachableSet(current_x_SV1, Qu1_k, qu1_k)
        Qr2_Inv, qr2,  x_pre_sv2_k, y_pre_sv2_k = self.ReachableSet(current_x_SV2, Qu2_k, qu2_k)

        distances = np.sqrt((self.x_track_ev - current_x_EV[0])**2 + (self.y_track_ev - current_x_EV[1])**2)
        min_distance_index = np.argmin(distances)
        index_ref = list(range(min_distance_index + 1, min_distance_index + self.N + 1))
        index_ref = [num if num <= (self.num_points_sv-1) else (self.num_points_sv-1) for num in index_ref]
        x_ref_k = self.x_track_ev[index_ref]
        y_ref_k = self.y_track_ev[index_ref]
        
        Trajectory_k, Control_k, J_k = self.MPCFormulation(Qr0_Inv, qr0, Qr1_Inv, qr1, Qr2_Inv, qr2, current_x_EV, x_ref_k, y_ref_k)
        Trajectory_k = Trajectory_k.full( )
        Control_k    = Control_k.full( )
        J_k          = J_k.full( )

        A_k    = [A0_k, A1_k, A2_k]
        b_k    = [b0_k, b1_k, b2_k]
        Qu_k   = [Qu0_k, Qu1_k, Qu2_k]
        qu_k   = [qu0_k, qu1_k, qu2_k]
        Qr_Inv  = [Qr0_Inv, Qr1_Inv, Qr2_Inv]
        qr      = [qr0, qr1, qr2]
        
        x_pre = [x_pre_sv0_k, x_pre_sv1_k, x_pre_sv2_k]   
        y_pre = [y_pre_sv0_k, y_pre_sv1_k, y_pre_sv2_k]   
        return Control_k[:, 0], Trajectory_k, J_k, A_k, b_k, Qu_k, qu_k, Qr_Inv, qr, x_pre, y_pre
    
    def SemiDefiProgramming(self, A_last, b_last, c_last, u_last): 
        nu        = self.nu
        B         = cp.Variable((nu, nu), PSD=True) # B = A**2
        tau       = cp.Variable( )
        tau_u     = cp.Variable( )
        b_tilde   = cp.Variable((nu, 1)) # b_tilde = A@b

        A_u = np.eye(nu)
        b_u = -u_last.reshape(2, 1)
        c_u = u_last.T@u_last - self.du

        zero_block = np.zeros((nu, nu))
        
        M  = cp.bmat([[B - tau * A_last, b_tilde - tau * b_last, zero_block],
                     [(b_tilde - tau * b_last).T, cp.reshape(-1 - tau * c_last, (1, 1)), b_tilde.T],
                     [zero_block, b_tilde, -B]])
        Mu = cp.bmat([[B - tau_u * A_u, b_tilde - tau_u * b_u, zero_block],
                     [(b_tilde - tau_u * b_u).T, cp.reshape(-1 - tau_u * c_u, (1, 1)), b_tilde.T],
                     [zero_block, b_tilde, -B]])
        
        constraints = [tau >= 0, tau_u >= 0, -M >> 0, -Mu >> 0, B - 1e-8*np.eye(nu) >> 0]  
        objective   = cp.Maximize(cp.log_det(B))
        problem     = cp.Problem(objective, constraints)
        problem.solve(solver = cp.MOSEK)

        return B.value, b_tilde.value
    
    def MPCFormulation(self):
        N   = self.N
        DEV = self.DEV
        T   = self.T
        Q1  = self.Q1
        Q2  = self.Q2
        Q3  = self.Q3

        v_low     = self.v_low 
        v_up      = self.v_up 
        acc_low   = self.acc_low 
        acc_up    = self.acc_up 
        delta_low = self.delta_low 
        delta_up  = self.delta_up

        opti  = casadi.Opti( )
        X     = opti.variable(DEV, N + 1)
        U     = opti.variable(2, N)
        acc   = U[0, :]
        delta = U[1, :]
        x     = X[0, 1::]
        y     = X[1, 1::]
        v     = X[3, 1::]
        
        Qr0_Inv  = opti.parameter(2, 2*N)
        qr0      = opti.parameter(2, N)
        Qr1_Inv  = opti.parameter(2, 2*N)
        qr1      = opti.parameter(2, N)
        Qr2_Inv  = opti.parameter(2, 2*N)
        qr2      = opti.parameter(2, N)
        Initial = opti.parameter(DEV, 1)
        x_ref   = opti.parameter(1, N)
        y_ref   = opti.parameter(1, N)
    
        opti.subject_to(X[:, 0] == Initial)
        for k in range(N):
            k1 = self.vehicle_model(X[:, k], acc[k], delta[k])
            k2 = self.vehicle_model(X[:, k] + T/2*k1, acc[k], delta[k])
            k3 = self.vehicle_model(X[:, k] + T/2*k2, acc[k], delta[k])
            k4 = self.vehicle_model(X[:, k] + T*k3,   acc[k], delta[k])
            x_next = X[:, k] + T/6 * (k1 + 2*k2 + 2*k3 + k4)
            opti.subject_to(X[:, k + 1] == x_next) 
            
        x_error = x - x_ref
        y_error = y - y_ref 
 
        for k in range(N):
            p_point = X[0:2, k + 1]
            
            Qr0_k_Inv = Qr0_Inv[:, 2*k:2*k + 2]
            qr0_k     = qr0[:, k]
            Qr1_k_Inv = Qr1_Inv[:, 2*k:2*k + 2]
            qr1_k     = qr1[:, k]
            Qr2_k_Inv = Qr2_Inv[:, 2*k:2*k + 2]
            qr2_k     = qr2[:, k]
            opti.subject_to(1 <= (p_point - qr0_k).T@Qr0_k_Inv@(p_point - qr0_k))
            opti.subject_to(1 <= (p_point - qr1_k).T@Qr1_k_Inv@(p_point - qr1_k))
            opti.subject_to(1 <= (p_point - qr2_k).T@Qr2_k_Inv@(p_point - qr2_k))

        opti.subject_to(opti.bounded(v_low, v, v_up))
        opti.subject_to(opti.bounded(acc_low, acc, acc_up))
        opti.subject_to(opti.bounded(delta_low, delta, delta_up))
        
        J = delta@Q1@delta.T + acc@Q2@acc.T +  Q3*(x_error@x_error.T + y_error@y_error.T)
        opti.minimize(J)
        
        opts = {"ipopt.print_level": 0, "ipopt.linear_solver": "ma57", "print_time": False}
        opti.solver('ipopt', opts)

        return opti.to_function('g', [Qr0_Inv, qr0, Qr1_Inv, qr1, Qr2_Inv, qr2, Initial, x_ref, y_ref], [X, U, J])

    def  vehicle_model(self, w, acc, delta):

        l_f = self.l_f
        l_r = self.l_r
        
        beta      = np.arctan(l_r/(l_f + l_r)*np.tan(delta))
        
        x_dot     = w[3]*np.cos(w[2] + beta) 
        y_dot     = w[3]*np.sin(w[2] + beta)
        phi_dot   = w[3]/(l_r)*np.sin(beta)
        v_dot     = acc
        
        return casadi.vertcat(x_dot, y_dot, phi_dot, v_dot)
    
    def  vehicle_model_sim(self, w, acc, delta):

        l_f = self.l_f
        l_r = self.l_r
        
        beta      = np.arctan(l_r/(l_f + l_r)*np.tan(delta))
        
        x_dot     = w[3]*np.cos(w[2] + beta) 
        y_dot     = w[3]*np.sin(w[2] + beta)
        phi_dot   = w[3]/(l_r)*np.sin(beta)
        v_dot     = acc
        
        return np.array([x_dot, y_dot, phi_dot, v_dot])

    def SVMPCFormulation(self):
        N = self.N
        T = self.T

        opti = casadi.Opti( )
        X = opti.variable(4, N + 1)
        U = opti.variable(2, N)
        ax   = U[0, :]
        ay   = U[1, :]
        
        Initial = opti.parameter(4, 1)
        x_ref   = opti.parameter(1, N)
        y_ref   = opti.parameter(1, N)
        
        opti.subject_to(X[:, 0] == Initial)
        for k in range(N):
            k1 = self.SV_vehicle_model(X[:, k], ax[k], ay[k])
            k2 = self.SV_vehicle_model(X[:, k] + T/2*k1, ax[k], ay[k])
            k3 = self.SV_vehicle_model(X[:, k] + T/2*k2, ax[k], ay[k])
            k4 = self.SV_vehicle_model(X[:, k] + T*k3,   ax[k], ay[k])
            x_next = X[:, k] + T/6 * (k1 + 2*k2 + 2*k3 + k4)
            opti.subject_to(X[:, k + 1] == x_next) 
            opti.subject_to(self.H@U[:, k] <= self.h)
            
        x = X[0, 1::]
        y = X[2, 1::]
        
        x_error   = x - x_ref
        y_error   = y - y_ref 
        
        J = x_error@x_error.T + y_error@y_error.T
        opti.minimize(J)
        
        opts = {"ipopt.print_level": 0, "ipopt.linear_solver": "ma57", "print_time": False}
        opti.solver('ipopt', opts)

        return opti.to_function('g', [Initial, x_ref, y_ref], [X])

    def  SV_vehicle_model(self, w, ax, ay):
        
        x_dot  = w[1]
        vx_dot = ax
        y_dot  = w[3]
        vy_dot = ay
        
        return casadi.vertcat(x_dot, vx_dot, y_dot, vy_dot)
