"""
Synthetic Occupant Data Generation Schemes 1D and 2D
"""


import numpy as np
from scipy.stats import norm , bernoulli
import json

from . import objfunc


def probit_link_func(x):
    """
    Probit link function
    Output : CDF of x 
    """
    return norm.cdf(x)

def gen_output(u1, u2):
    """
    Inputs:
    u1 : first column
    u2 : second column
    Outputs:
    y : binary variable (1 or 0) ; 1 if x2 is preferred ; 0 if x1 is preferred
    p: actual preference probability
    """
    # utility function value generation
    diff = u2 - u1
    #y = 1*(diff > 0.)
    p = probit_link_func(3*diff)
    y = bernoulli.rvs(p)
    return y


class ThermalPrefDataGen(object):
    def __init__(self, config_file):
        """
        Synthetic occupant pairwise comparison data generation
        We assume that there is a synthetic occupant who has a specific utility.
        This class is used to generate 1D and 2D feature state based pairwise
        comparisons. We assume that the visual state is defined by two features,
        Operating temp and relative humidity. If you want to add more feature
        to define the state of the room, you need to work on adding method to this
        class which would take care of 3D feature pairwise comparison data.
        """
        # Read configuration file
        with open(config_file, 'r') as fd:
            config = json.loads(fd.read())
        
        # Grid parameters
        self.Gridmin1 = config['Grid']['min1']
        self.Gridmax1 = config['Grid']['max1']
        self.numgrid1 = config['Grid']['numgrid1']
        self.Gridmin2 = config['Grid']['min2']
        self.Gridmax2 = config['Grid']['max2']
        self.numgrid2_1 = config['Grid']['numgrid2_1']
        self.numgrid2_2 = config['Grid']['numgrid2_2']
        
        # Thermal utility function parameters 2D
        self.m1 = config['Temp_RH_Gaussian']['t_mean']
        self.v1 = config['Temp_RH_Gaussian']['t_var']
        self.m2 = config['Temp_RH_Gaussian']['rh_mean']
        self.v2 = config['Temp_RH_Gaussian']['rh_var']
        self.off =  config['Temp_RH_Gaussian']['off_diag']
        self.n = config['Temp_RH_Gaussian']['u_norm']
        
        # Thermal utility function parameters 1D
        self.a = config['Beta_Objective_func']['a']
        self.b = config['Beta_Objective_func']['b']
        self.l = config['Beta_Objective_func']['l']
    
    def normalize1Dpairwise(self, X):
        """
        Normalize vertical illuminance feature duels
        Inputs: 
            X (N x 2) : pairwise comparison data [X_prev, X_current]
        Outputs:
            X_norm : normalized pairwise comparison data
        """
        X_norm = 2./(self.Gridmax1 - self.Gridmin1)*(X - self.Gridmin1) - 1.
        
        return X_norm
    
    def normalize1D(self, X):
        """
        Normalize 1 D feature vector 
        Input:
            X (N x 1) : 1D feature vector
        Output: 
            X_norm : normalized 1D feature vector
        """
        
        return self.normalize1Dpairwise(X)
    
    def normalize2D(self, X):
        """
        Normalize 2D feature matrix
        Inputs:
            X (N x 2) : 2D feature matrix 
        Outputs:
            X_norm (N X 2) : Noramlized 2D feature matrix
        """
        raise ValueError('2D norm is broken.... needs revisions')
        X_norm = np.zeros(shape = X.shape)
        X_norm[:,0] = (X[:,0] - self.Gridmin1)/self.Gridmax1
        X_norm[:,1] = (X[:,1] - self.Gridmin2)/self.Gridmax2
        
        return X_norm
    
    def normalize2Dpairwise(self, X):
        """
        Normalize 2D pariwise comparison matrix
        Inputs:
            X (N x 4) : pairwise comparion data [X_prev, X_current]
        Outputs:
            X_norm : normalized pairwise comparison data
        """
        X_norm = np.zeros(shape = X.shape)
        X_norm[:,0] = (X[:,0] - self.Gridmin1)/self.Gridmax1
        X_norm[:,1] = (X[:,1] - self.Gridmin2)/self.Gridmax2
        X_norm[:,2] = (X[:,2] - self.Gridmin1)/self.Gridmax1
        X_norm[:,3] = (X[:,3] - self.Gridmin2)/self.Gridmax2
    
        return X_norm
    
    def feat1_grid_duels(self, num_points):
        """
        Grid for duels : only vertical illuminance as a feature
        This grid is required for calculating preference probabilities
        Outputs:
            Xgrid, Xgridnorm, num_points (number of grid points)
        """
        # Normalization of features
        Grid = np.linspace(self.Gridmin1, self.Gridmax1, num_points)
        # Grid configuration 1D
        Xtt1, Xtt2 = np.meshgrid(Grid, Grid)
        Xtt1_ravel = np.ravel(Xtt1)
        Xtt2_ravel = np.ravel(Xtt2)
        # Grid points defined
        Xgrid = np.zeros(shape = (Xtt1_ravel.shape[0], 2))
        Xgrid[:,0] = Xtt1_ravel
        Xgrid[:,1] = Xtt2_ravel
        # Normalization of grid points
        Xgridnorm = self.normalize1Dpairwise(Xgrid)
        
        return Xgrid, Xgridnorm
    
    def feat1_nxt_point_grid(self, num_points):
        """
        Grid for selection of next datapoint:
        if feat = 1 then we will have a 1D grid so as to 
        calculate the posterior utility at those points
        """
        # Grid Points
        vecforut = np.linspace(self.Gridmin1, self.Gridmax1, num_points)
        vec_norm = self.normalize1D(vecforut)
        
        return vecforut[:,None], vec_norm[:,None]
    
    def feat2_grid_duels(self, num_points1, num_points2):
        """
        Grid for duels : if feat = 2 then we will have a 4D grid so as to 
        calculate the preference probabilities
        """
        # Grid Points
        Grid1 = np.linspace(self.Gridmin1, self.Gridmax1, num_points1)
        Grid2 = np.linspace(self.Gridmin2, self.Gridmax2, num_points2)
        Xtt11, Xtt21, Xtt12, Xtt22 =  np.meshgrid(Grid1, Grid2, Grid1, Grid2)
        Xtt11_ravel = np.ravel(Xtt11)
        Xtt21_ravel = np.ravel(Xtt21)
        Xtt12_ravel = np.ravel(Xtt12)
        Xtt22_ravel = np.ravel(Xtt22)
        
        Xgrid = np.zeros(shape = (Xtt11_ravel.shape[0], 4))
        Xgrid[:,0] = Xtt11_ravel
        Xgrid[:,1] = Xtt12_ravel
        Xgrid[:,2] = Xtt21_ravel
        Xgrid[:,3] = Xtt22_ravel
        
        Xgridnorm = self.normalize2Dpairwise(Xgrid)
        
        return Xgrid, Xgridnorm
    
    def feat2_nxt_point_grid(self, numpoints1, numpoints2):
        """
        Grid for selection of next state ( num of features  = 2)
        """
        # Grid Points
        Grid1 = np.linspace(self.Gridmin1, self.Gridmax1, numpoints1)
        Grid2 = np.linspace(self.Gridmin2, self.Gridmax2, numpoints2)
        # Matrix for testing states
        Xtt1, Xtt2 = np.meshgrid(Grid1, Grid2)
        
        Xtt1_ravel = np.ravel(Xtt1)
        Xtt2_ravel = np.ravel(Xtt2)
        
        # Grid points defined
        Xgrid = np.zeros(shape = (Xtt1_ravel.shape[0], 2))
        Xgrid[:,0] = Xtt1_ravel
        Xgrid[:,1] = Xtt2_ravel
        
        Xgridnorm = self.normalize2D(Xgrid)
        
        return Xgrid, Xgridnorm
    
    def response_gen2D(self, X):
        """
        Given duels matrix 2D features, generate response of synthetic occ
        Inputs:
            X (N x 4) : pairwise comparion data [X_prev, X_current]
        Outputs:
            Y (N x 1) : yi vector, yi = 1 if xcurrent is preferred, 0 otherwise
        """
        x1 = X[:,:2]
        x2 = X[:,2:]
        u1 = objfunc.u2D(x1, self.m1, self.v1,
                         self.m2, self.v2, self.off)
        u2 = objfunc.u2D(x2, self.m1, self.v1,
                         self.m2, self.v2, self.off)
        y_pr = gen_output(u1, u2)
        
        return y_pr
    
    def response_gen1D(self, X):
        """
        Given duels matrix 1D features, generate response of synthetic occ
        Inputs:
            X (N x 2) : pairwise comparion data [X_prev, X_current]
        Outputs:
            Y (N x 1) : yi vector, yi = 1 if xcurrent is preferred, 0 otherwise
        """
        x1 = X[:,0]
        x2 = X[:,1]
        u1 = objfunc.beta_utility_gen(x1, self.l,
                                             self.a, self.b)
        u2 = objfunc.beta_utility_gen(x2, self.l,
                                             self.a, self.b)
        
        y_pr = gen_output(u1, u2)
        return y_pr
    
    def duels_gen(self, num_feat, num_datapoints):
        """
        Generation of training duels and associated utilities
        """
        n = 50 # number of datapoints
        x1 = np.linspace(self.Gridmin1, self.Gridmax1, n)[:, None] # Operating temp.
        x2 = np.linspace(self.Gridmin2, self.Gridmax2, n)[:, None] # Relative humidity
        
         # sampling from indexes
        indexes = np.arange(n)
        
        ind_samp1 = np.random.choice(indexes, size = 1)
        x_samp1_01 = x1[ind_samp1]
        
        ind_samp2_all1 = np.random.choice(indexes, size = num_datapoints)
        x_samp21 = x1[ind_samp2_all1]
        x_samp1_n1 = x_samp21[:-1]
        x_samp11 = np.append(x_samp1_01, x_samp1_n1)[:,None]
        
        ind_samp2 = np.random.choice(indexes, size = 1)
        x_samp1_02 = x2[ind_samp2]
        ind_samp2_all2 = np.random.choice(indexes, size = num_datapoints)
        x_samp22 = x2[ind_samp2_all2]
        x_samp1_n2 = x_samp22[:-1]
        x_samp12 = np.append(x_samp1_02, x_samp1_n2)[:,None]
        
        if num_feat == 1:
            u1 = objfunc.beta_utility_gen(x_samp11, self.l,
                                             self.a, self.b)
            u2 = objfunc.beta_utility_gen(x_samp21, self.l,
                                             self.a, self.b) 
        elif num_feat == 2:
            x_samp1 = np.hstack([x_samp11, x_samp12])
            x_samp2 = np.hstack([x_samp21, x_samp22])
            u1 = objfunc.u2D(x_samp1, self.m1, self.v1,
                         self.m2, self.v2, self.off)
            u2 = objfunc.u2D(x_samp2, self.m1, self.v1,
                             self.m2, self.v2, self.off)
        else:
         raise ValueError('Need to write a function for feat > 2')   

        return (x_samp11, x_samp12, x_samp21, x_samp22, u1, u2)
    
    def pairwise2D(self, num_datapoints, save_file_name, save_file = False):
        """
        Generate 2D pairwise visual preferences dataset
        Two features values
        """
        num_feat = 2
        (x_samp11, x_samp12,
         x_samp21, x_samp22, u1, u2) = self.duels_gen(num_feat, num_datapoints)
        y_pr = gen_output(u1, u2)
        X = np.hstack([x_samp11, x_samp12,
                       x_samp21, x_samp22])
        if save_file: 
            np.savez(save_file_name, X = X, Y = y_pr[:,None])
        return X, y_pr[:,None]
    
    def pairwise1D(self, num_datapoints, save_file_name, save_file = False):
        """
        Generate 1D pairwise preferences dataset
        One feature values
        """
        num_feat = 1
        (x_samp11, x_samp12,
         x_samp21, x_samp22, u1, u2) = self.duels_gen(num_feat, num_datapoints)
        y_pr = gen_output(u1, u2)
        X = np.hstack([x_samp11, x_samp21])
        if save_file: 
            np.savez(save_file_name, X = X, Y = y_pr)
        return X, y_pr
    
    
class ReachableStates(object):
    def __init__(self, Xprev):
        """
        Class which contains methods that output states which can be reached
        at any given moment. These states can be constrained/ unconstrained.
        Inputs:
        Xprev : Previous state value
        Outputs:
            Xgrid : states which can be reached from current state
            Xgridnorm : Normalized Xgrid
        if shape(Xprev) = (1,) ; shape(Xreachable) = 1 X R (R = num of reachable states)
        that means only 1D feature is used to define state
        if shape(Xprev) = (2,) ; shape(Xreachable) = 2 X R (R = num of reachable states)
        """
        self.Xprev = Xprev
        
    def rs1D(self, config_file):
        """
        For this method, the reachable state is not dependent on the previous state
        1D feature is used to define the state of the room
        """
        VD = ThermalPrefDataGen(config_file)
        Xr, Xrnorm = VD.feat1_nxt_point_grid(VD.numgrid1)
        print self.Xprev
        
        return Xr, Xrnorm
    
    def rs2D(self, config_file):
        """
        For this method, the reachable state is not dependent on the previous state
        2D features are used to define the state of the room
        """
        VD = ThermalPrefDataGen(config_file)
        Xr, Xrnorm = VD.feat2_nxt_point_grid(VD.numgrid2_1, VD.numgrid2_1)
        
        return Xr, Xrnorm
    
    def reachable(self, config_file):
        """
        Reachable states
        """
        VD = ThermalPrefDataGen(config_file)
        # Read configuration file
        with open(config_file, 'r') as fd:
            config = json.loads(fd.read())
        reachable_temp_diff = config['reachable_states']['temp_diff']
        reachable_num_points = config['reachable_states']['num_points']
        grid_min = config['Grid']['min1']
        grid_max = config['Grid']['max1']
        reach_min = self.Xprev - reachable_temp_diff
        reach_max = self.Xprev + reachable_temp_diff
        if reach_min < grid_min:
            reach_min = grid_min
        if reach_max > grid_max:
            reach_max = grid_max
        Xr = np.linspace(reach_min, reach_max, reachable_num_points)[:,None]
        Xrnorm = VD.normalize1D(Xr)
        
        return Xr, Xrnorm
        
        
    def user_defined(self):
        """
        User defined reachable state function
        """
        print 'Define your reachable state function ...'
        
        return
    
    
if __name__ == '__main__':
    config_file = '../config_files/thermal_config.json'
    save_file_name1 = '../data/initial_duels/train1D.npz'
    save_file_name2 = '../data/initial_duels/train2D.npz'
    ThermalP = ThermalPrefDataGen(config_file)
    
    X1, y_pr1 = ThermalP.pairwise1D(2, save_file_name1, save_file = False)
    X2, y_pr2 = ThermalP.pairwise2D(40, save_file_name2, save_file = False)
    
       
    