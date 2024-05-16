import numpy as np
import random as random
from shapely.geometry import Polygon, Point

class Check_Safety( ):
    def __init__(self, Params):
        
        self.l_veh = Params['l_veh']
        self.w_veh = Params['w_veh']
        self.x_con_c = Params['x_con_c']
        self.y_con_c = Params['y_con_c']
        self.r_in = Params['r_in']
        self.r_out = Params['r_out']
        self.p = Params['p']
        
    def check_road_collision(self, state_ev):
        
        l_veh = self.l_veh
        w_veh = self.w_veh
        x_con_c = self.x_con_c
        y_con_c = self.y_con_c
        r_in = self.r_in
        r_out = self.r_out
        p = self.p
        
        x = state_ev[0]
        y = state_ev[1]
        heading = state_ev[2]
        
        x = state_ev[0]
        y = state_ev[1]
        heading = state_ev[2]
        box = [(x - l_veh/2, y + w_veh/2), (x + l_veh/2, y + w_veh/2), (x + l_veh/2, y - w_veh/2), (x - l_veh/2, y - w_veh/2)]
        X, Y = self.rota_rect(box, heading, x, y)
        
        X = np.array(X)
        Y = np.array(Y)
        
        in_con = ((X - x_con_c)/r_in)**p + ((Y - y_con_c)/r_in)**p - 1  
        out_con = 1 - ((X - x_con_c)/r_out)**p - ((Y - y_con_c)/r_out)**p
        in_con_infeasible  = sum(i <= -0.001 for i in in_con)
        out_con_infeasible = sum(i <= -0.001 for i in out_con)
        
        if (in_con_infeasible == 0) and (out_con_infeasible == 0):
            flag = 0 # no collision with road
        else:
            flag = 1 # collision with road
        
        
        return flag
    
    def check_vehicle_collision(self, state_ev, state_sv):
        
        l_veh = self.l_veh
        w_veh = self.w_veh
        
        x = state_sv[0]
        y = state_sv[1]
        heading = state_sv[2]
        box = [(x - l_veh/2, y + w_veh/2), (x + l_veh/2, y + w_veh/2), (x + l_veh/2, y - w_veh/2), (x - l_veh/2, y - w_veh/2)]
        x_SV, y_SV = self.rota_rect(box, heading, x, y)

        x = state_ev[0]
        y = state_ev[1]
        heading = state_ev[2]
        box = [(x - l_veh/2, y + w_veh/2), (x + l_veh/2, y + w_veh/2), (x + l_veh/2, y - w_veh/2), (x - l_veh/2, y - w_veh/2)]
        x_EV, y_EV = self.rota_rect(box, heading, x, y)
        
        pSV = Polygon([(x_SV[0],y_SV[0]), (x_SV[1],y_SV[1]), (x_SV[2],y_SV[2]), (x_SV[3],y_SV[3])])
        pEV = Polygon([(x_EV[0], y_EV[0]), (x_EV[1], y_EV[1]), (x_EV[2], y_EV[2]), (x_EV[3], y_EV[3])])
        
        return pEV.distance(pSV)
    
    def rota_rect(self, box, theta, x, y):
        
        box_matrix = np.array(box) - np.repeat(np.array([[x, y]]), len(box), 0)
        theta = -theta
        rota_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                [np.sin(theta), np.cos(theta)]])

        new = box_matrix.dot(rota_matrix) + np.repeat(np.array([[x, y]]), len(box), 0)

        x = [new[0][0], new[1][0], new[2][0], new[3][0], new[0][0]]
        y = [new[0][1], new[1][1], new[2][1], new[3][1], new[0][1]]
        
        return x, y