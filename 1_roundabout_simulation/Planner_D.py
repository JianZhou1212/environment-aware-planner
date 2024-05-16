import numpy as np
import math
import casadi
import time

class Planner_D( ):
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
        self.v_up = Params['v_up']
        self.acc_low = Params['acc_low']
        self.acc_up = Params['acc_up']
        self.delta_low = Params['delta_low']
        self.delta_up = Params['delta_up']
        self.x_track_sv = Params['x_track_sv']
        self.y_track_sv = Params['y_track_sv']
        self.num_points_sv = Params['num_points_sv']
        self.x_con_c = Params['x_con_c']
        self.y_con_c = Params['y_con_c']
        self.r_in = Params['r_in']
        self.r_out = Params['r_out']
        self.p = Params['p']
        self.margin = Params['margin']
        self.acc_max = Params['acc_max']
        self.collision_tol = Params['collision_tol']
        self.state_tol = Params['state_tol']
        self.Q_measure = Params['Q_measure']
        self.Q_safedis = Params['Q_safedis']
        self.L_State = Params['L_State']
        self.L_Occ   = Params['L_Occ']
        self.H       = Params['H']
        self.h       = Params['h']
        self.MPCFormulation = self.MPCFormulation( )
        self.MPCFormulationBackup = self.MPCFormulationBackup( )
        self.SVMPCFormulation = self.SVMPCFormulation( )
    
    def check_feasibility(self, x_ev, y_ev, v_ev, acc_ev, del_ev):
        
        x_con_c = self.x_con_c
        y_con_c = self.y_con_c
        r_in = self.r_in 
        r_out = self.r_out
        p = self.p
        
        u_p_inf    = sum(i <= -self.collision_tol for i in 1 - ((x_ev - x_con_c)/r_out)**p - ((y_ev - y_con_c)/r_out)**p)
        l_p_inf    = sum(i <= -self.collision_tol for i in ((x_ev - x_con_c)/r_in)**p + ((y_ev - y_con_c)/r_in)**p - 1 )
        u_v_inf    = sum(i <= -self.state_tol for i in self.v_up - v_ev)
        l_v_inf    = sum(i <= -self.state_tol for i in v_ev - self.v_low)
        u_del_inf  = sum(i <= -self.state_tol for i in self.delta_up - del_ev)
        l_del_inf  = sum(i <= -self.state_tol for i in del_ev - self.delta_low)
        u_acc_inf  = sum(i <= -self.state_tol for i in self.acc_up - acc_ev)
        l_acc_inf  = sum(i <= -self.state_tol for i in acc_ev - self.acc_low)
        
        if (u_p_inf == 0) and (l_p_inf == 0) and (u_v_inf== 0) and (l_v_inf== 0) and (u_del_inf== 0) and (l_del_inf== 0) and (u_acc_inf== 0) and (l_acc_inf== 0) :
        #if (u_p_inf == 0) and (l_p_inf == 0):
            flag = 0 # no collision with road
        else:
            flag = 1 # collision with road
        return flag
    
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

    def Minksum_EA(self, q1, Q1, q2, Q2, L):
        q       = q1 + q2
        Q_Left  = np.sqrt(L.T@Q1@L) + np.sqrt(L.T@Q2@L)
        Q_Right = Q1/np.sqrt(L.T@Q1@L) + Q2/np.sqrt(L.T@Q2@L)
        Q       = Q_Left*Q_Right
        
        return q, Q
    
    def ReachableSet(self, current_x_SV):
        N      = self.N
        L_Occ  = self.L_Occ
        Q_measure = self.Q_measure
        Q_safedis = self.Q_safedis
        
        SVPredictionTrajectory_k = self.SVPrediction(current_x_SV)
        x_pre_sv_k = SVPredictionTrajectory_k[0, :]
        y_pre_sv_k = SVPredictionTrajectory_k[2, :]

        Qr_Inv = np.zeros((2, 2*N)) 
        qr = np.zeros((2, N))
        Q_meas = Q_measure[:, [0, 2]]
        Q_meas = Q_meas[[0, 2], :]
        _, Q_occ = self.Minksum_EA(np.zeros((2, 1)), Q_safedis, np.zeros((2, 1)), Q_meas, L_Occ)

        for t in range(1, N+1):
            Qr_Inv[:, 2*t-2:2*t] = np.linalg.inv(Q_occ)
            qr[:, t-1]           = np.array([x_pre_sv_k[t], y_pre_sv_k[t]])
        return Qr_Inv, qr
    
    def Return(self, current_x_SV, current_x_EV):

        current_x_SV = np.array([current_x_SV[0], current_x_SV[3]*np.cos(current_x_SV[2]), current_x_SV[1], current_x_SV[3]*np.sin(current_x_SV[2])])
            
        Qr_Inv, qr = self.ReachableSet(current_x_SV)

        Trajectory_k, Control_k, J_k = self.MPCFormulation(Qr_Inv, qr, current_x_EV)
        Trajectory_k = Trajectory_k.full( )
        Control_k    = Control_k.full( )
        J_k          = J_k.full( )
        
        flag = self.check_feasibility(Trajectory_k[0, 1::], Trajectory_k[1, 1::], Trajectory_k[3, 1::], Control_k[0], Control_k[1])
        if flag == 1:
            Trajectory_k, Control_k, J_k = self.MPCFormulationBackup(current_x_EV)
            Trajectory_k = Trajectory_k.full( )
            Control_k    = Control_k.full( )
            J_k          = abs(J_k.full( ))

        return Control_k[:, 0], Trajectory_k, J_k, Qr_Inv, qr, flag
    
    def MPCFormulation(self):
        N   = self.N
        DEV = self.DEV
        T   = self.T
        Q1  = self.Q1
        Q2  = self.Q2
        Q3  = self.Q3
        x_con_c = self.x_con_c
        y_con_c = self.y_con_c
        r_in    = self.r_in
        r_out   = self.r_out
        p       = self.p
 
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

        Q_Inv = opti.parameter(2, 2*N)
        q = opti.parameter(2, N)
        Initial = opti.parameter(DEV, 1)
    
        opti.subject_to(X[:, 0] == Initial)
        for k in range(N):
            k1 = self.vehicle_model(X[:, k], acc[k], delta[k])
            k2 = self.vehicle_model(X[:, k] + T/2*k1, acc[k], delta[k])
            k3 = self.vehicle_model(X[:, k] + T/2*k2, acc[k], delta[k])
            k4 = self.vehicle_model(X[:, k] + T*k3,   acc[k], delta[k])
            x_next = X[:, k] + T/6 * (k1 + 2*k2 + 2*k3 + k4)
            opti.subject_to(X[:, k + 1] == x_next) 

        v       = X[3, 1::]
        v_error = v[-1] - self.v_up
 
        for k in range(N):
            p_point = X[0:2, k + 1]
            
            Q_k_Inv = Q_Inv[:, 2*k:2*k + 2]
            q_k = q[:, k]
            opti.subject_to(1 <= (p_point - q_k).T@Q_k_Inv@(p_point - q_k))
            opti.subject_to(1 <= ((p_point[0] - x_con_c)/r_in)**p + ((p_point[1] - y_con_c)/r_in)**p)
            opti.subject_to(((p_point[0] - x_con_c)/r_out)**p + ((p_point[1] - y_con_c)/r_out)**p <= 1)

        opti.subject_to(opti.bounded(v_low, v, v_up))
        opti.subject_to(opti.bounded(acc_low, acc, acc_up))
        opti.subject_to(opti.bounded(delta_low, delta, delta_up))
        
        J = delta@Q1@delta.T + acc@Q2@acc.T +  Q3*v_error@v_error.T 
        opti.minimize(J)
        
        opts = {"ipopt.print_level": 0, "ipopt.linear_solver": "ma57", "print_time": False}
        opti.solver('ipopt', opts)

        return opti.to_function('g', [Q_Inv, q, Initial], [X, U, J])
    
    def MPCFormulationBackup(self):
        N = self.N
        DEV = self.DEV
        T = self.T
        x_con_c = self.x_con_c
        y_con_c = self.y_con_c
        r_in = self.r_in
        r_out = self.r_out
        p = self.p
 
        v_low = self.v_low 
        v_up = self.v_up 
        acc_up = self.acc_up 
        delta_low = self.delta_low 
        delta_up = self.delta_up

        opti = casadi.Opti( )
        X = opti.variable(DEV, N + 1)
        U = opti.variable(2, N)
        acc   = U[0, :]
        delta = U[1, :]

        lam = opti.variable(4, N)
        
        G = opti.parameter(4, 2*N)
        g = opti.parameter(4, N)
        Initial = opti.parameter(DEV, 1)
    

        opti.subject_to(X[:, 0] == Initial)
        for k in range(N):
            k1 = self.vehicle_model(X[:, k], acc[k], delta[k])
            k2 = self.vehicle_model(X[:, k] + T/2*k1, acc[k], delta[k])
            k3 = self.vehicle_model(X[:, k] + T/2*k2, acc[k], delta[k])
            k4 = self.vehicle_model(X[:, k] + T*k3,   acc[k], delta[k])
            x_next = X[:, k] + T/6 * (k1 + 2*k2 + 2*k3 + k4)
            opti.subject_to(X[:, k + 1] == x_next) 


        v     = X[3, 1::]
        for k in range(N):
            p_point = X[0:2, k + 1]
            opti.subject_to(1 <= ((p_point[0] - x_con_c)/r_in)**p + ((p_point[1] - y_con_c)/r_in)**p)
            opti.subject_to(((p_point[0] - x_con_c)/r_out)**p + ((p_point[1] - y_con_c)/r_out)**p <= 1)

        opti.subject_to(acc[0] == -self.acc_max)
        opti.subject_to(acc[1] == -self.acc_max)
        opti.subject_to(opti.bounded(v_low, v, v_up))
        opti.subject_to(opti.bounded(-self.acc_max, acc, self.acc_up))
        opti.subject_to(opti.bounded(delta_low, delta, delta_up))
        
        J = -acc@acc.T
        opti.minimize(J)
        
        opts = {"ipopt.print_level": 0, "ipopt.linear_solver": "ma57", "print_time": False}
        opti.solver('ipopt', opts)

        return opti.to_function('g', [Initial], [X, U, J])

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
        x_ref = opti.parameter(1, N)
        y_ref = opti.parameter(1, N)
        U_A = opti.parameter(4, 2)
        U_b = opti.parameter(4, 1)
        
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
        
        x_dot = w[1]
        vx_dot = ax
        y_dot = w[3]
        vy_dot = ay
        
        return casadi.vertcat(x_dot, vx_dot, y_dot, vy_dot)
