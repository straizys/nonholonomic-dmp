import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import pytransform3d
from pytransform3d.plot_utils import plot_mesh
from pytransform3d.transformations import plot_transform
from scipy import optimize
import copy

import utils

class NonholonomicDMP():
    
    def __init__(self,N_bf=20,alphaz=4.0,betaz=1.0,dt=0.001):
        
        self.alphax = 1.0
        self.alphaz = alphaz
        self.betaz = betaz
        self.N_bf = N_bf # number of basis functions
        self.dt = dt
        self.T = 1.0

        self.phase = 1.0 # initialize phase variable

    def imitate(self, pose_demo):

        self.N = pose_demo.shape[0]
        
        self.x_des = copy.deepcopy(pose_demo[:,:3]) # position 
        self.q_des = copy.deepcopy(pose_demo[:,3:]) # orientation, quaternion represenation (scalar last format)
                
        # Centers of basis functions 
        self.c = np.ones(self.N_bf) 
        c_ = np.linspace(0,self.T,self.N_bf)
        for i in range(self.N_bf):
            self.c[i] = np.exp(-self.alphax *c_[i])

        # Widths of basis functions 
        # (as in https://github.com/studywolf/pydmps/blob/80b0a4518edf756773582cc5c40fdeee7e332169/pydmps/dmp_discrete.py#L37)
        self.h = np.ones(self.N_bf) * self.N_bf**1.5 / self.c / self.alphax
        
        # Derivatives for position and orientation components
        self.dx_des = np.gradient(self.x_des,axis=0)/self.dt # linear velocity
        # self.ddx_des = np.gradient(self.dx_des,axis=0)/self.dt # linear acceleration
        self.w_des = utils.quaternion_diff(self.q_des,self.dt) # angular velocity
        self.dw_des = np.gradient(self.w_des,axis=0)/self.dt # angular acceleration

        # Initial conditions
        self.x0 = self.x_des[0,:]
        # self.dx0 = self.dx_des[0,:] 
        # self.ddx0 = self.ddx_des[0,:]
        self.q0 = self.q_des[0,:]
        self.w0 = self.w_des[0,:] 
        self.dw0 = self.dw_des[0,:]

        # Initial condition must not violate the constraint
        R_ = R.from_quat(self.q_des[0,:]).as_matrix()
        n_ = R_[:,1]
        v_rand = np.random.rand(3)
        p1 = np.cross(n_,v_rand)
        p2 = np.cross(n_,p1)
        M = np.vstack((p1,p2)).T # Plane matrix
        proj_dxyz = np.matmul(M,np.matmul(np.linalg.inv(np.matmul(M.T,M)),np.dot(M.T,self.dx_des[0,:])))
        self.dx_des[0,:] = proj_dxyz
        self.ddx_des = np.gradient(self.dx_des,axis=0)/self.dt
        self.dx0 = self.dx_des[0,:] 
        self.ddx0 = self.ddx_des[0,:]
        
        # Final configuration
        self.xT = self.x_des[-1,:]
        self.qT = self.q_des[-1,:]
        
        # Initialize the DMP
        self.x = copy.deepcopy(self.x0)
        self.dx = copy.deepcopy(self.dx0)
        self.ddx = copy.deepcopy(self.ddx0)
        self.q = copy.deepcopy(self.q0)
        self.w = copy.deepcopy(self.w0)
        self.dw = copy.deepcopy(self.dw0)

        # Evaluate the forcing terms for position ...
        forcing_target_pos = self.ddx_des - \
                             self.alphaz*(self.betaz*(self.xT-self.x_des) - self.dx_des)
        
        # ... and orientation
        forcing_target_ori = np.zeros([self.N,3]) 
        for n in range(self.N):
            forcing_target_ori[n,:] = self.dw_des[n,:] - \
                                      self.alphaz*(self.betaz * utils.logarithmic_map(
                                      utils.quaternion_error(self.qT,self.q_des[n,:])) - \
                                                 self.w_des[n,:])
        
        self.fit_dmp(forcing_target_pos, forcing_target_ori)
        
        return self.x_des, self.dx_des, self.ddx_des, self.q_des, self.w_des, self.dw_des
    
    def RBF(self, phase):

        if type(phase) is np.ndarray:
            return np.exp(-self.h*(phase[:,np.newaxis]-self.c)**2)
        else:
            return np.exp(-self.h*(phase-self.c)**2)

    def forcing_function_approx(self,weights,phase):

        BF = self.RBF(phase)
        if type(phase) is np.ndarray:
            return np.dot(BF,weights)*phase/np.sum(BF,axis=1)
        else:
            return np.dot(BF,weights)*phase/np.sum(BF)
    
    def fit_dmp(self,forcing_target_pos, forcing_target_ori):

        phase = np.exp(-self.alphax*np.linspace(0.0,self.T,self.N))
        BF = self.RBF(phase)
        X = BF*phase[:,np.newaxis]/np.sum(BF,axis=1)[:,np.newaxis]
        
        regcoef = 0.01 # regularization coefficient
        
        self.weights_pos = np.zeros([self.N_bf,3])
        self.weights_ori = np.zeros([self.N_bf,3])
        
        for d in range(3):        
            self.weights_pos[:,d] = np.dot(np.dot(np.linalg.pinv(np.dot(X.T,(X)) + \
                                    regcoef * np.eye(X.shape[1])),X.T),forcing_target_pos[:,d].T)     
            
            self.weights_ori[:,d] = np.dot(np.dot(np.linalg.pinv(np.dot(X.T,(X)) + \
                                regcoef * np.eye(X.shape[1])),X.T),forcing_target_ori[:,d].T)

    def reset(self):
        
        self.phase = 1.0
        
        self.x = copy.deepcopy(self.x0)
        self.dx = copy.deepcopy(self.dx0)
        self.ddx = copy.deepcopy(self.ddx0)
        
        self.q = copy.deepcopy(self.q0)
        self.angular_vel = copy.deepcopy(self.angular_vel0)
        self.angular_acc = copy.deepcopy(self.angular_acc0)

    def rollout(self,con=True,optim=False):
        
        x_rollout = np.zeros([self.N,3])
        dx_rollout = np.zeros([self.N,3])
        ddx_rollout = np.zeros([self.N,3])
        x_rollout[0,:] = self.x0
        dx_rollout[0,:] = self.dx0
        ddx_rollout[0,:] = self.ddx0
        
        q_rollout = np.zeros([self.N,4])
        w_rollout = np.zeros([self.N,3])
        dw_rollout = np.zeros([self.N,3])
        q_rollout[0,:] = self.q0
        w_rollout[0,:] = self.w0
        dw_rollout[0,:] = self.dw0
        
        phase = np.exp(-self.alphax*np.linspace(0.0,self.T,self.N))

        constraint = []
        constraint_forces = []
        
        for n in range(1,self.N):

            # Position forcing term
            forcing_term_pos = self.forcing_function_approx(self.weights_pos,phase[n-1])

            # Orientation forcing term
            forcing_term_ori = self.forcing_function_approx(self.weights_ori,phase[n-1])

            # Calculate an unconstrained acceleration for position ...
            f_unc_pos = self.alphaz*(self.betaz*(self.xT-x_rollout[n-1,:]) - dx_rollout[n-1,:]) + forcing_term_pos
            
            # ... and orientation
            f_unc_ori = self.alphaz*(self.betaz * utils.logarithmic_map(
                utils.quaternion_error(self.qT,q_rollout[n-1,:])) - w_rollout[n-1,:]) + forcing_term_ori

            f_con = np.zeros(3)
            
            # Constrained force coupling term
            if con:

                if optim:

                  # Find the optimal angular acceleration
                  sol = optimize.minimize(utils.objective_function, x0=f_unc_ori, 
                                          args=(x_rollout[n-1,:], dx_rollout[n-1,:], f_unc_pos,
                                                q_rollout[n-1,:], w_rollout[n-1,:],f_unc_ori,self.dt),
                                          method='BFGS')#,options={'gtol': 1e-6, 'disp': True})
                  f_unc_ori = sol.x 

                # First, evaluate unconstrained system
                dx_unc = dx_rollout[n-1,:] + f_unc_pos*self.dt
                x_unc = x_rollout[n-1,:] + dx_unc*self.dt

                w_unc = w_rollout[n-1,:] + f_unc_ori*self.dt
                q_unc = utils.quaternion_product(
                    utils.exponential_map(w_unc*self.dt),q_rollout[n-1,:])

                # Then, evaluate the U-K constraint forces ... 
                f_con = utils.nonholonomic_constraint_force(q_unc,dx_unc,w_unc,f_unc_pos)
                constraint_forces.append(f_con)

            ddx_rollout[n,:] = f_unc_pos + f_con
            dw_rollout[n,:] = f_unc_ori
            
            # Integrate position DMP ...
            dx_rollout[n,:] = dx_rollout[n-1,:] + ddx_rollout[n,:]*self.dt
            x_rollout[n,:] = x_rollout[n-1,:] + dx_rollout[n,:]*self.dt
            
            # ... and orientation DMP
            w_rollout[n,:] = w_rollout[n-1,:] + dw_rollout[n,:]*self.dt
            q_rollout[n,:] = utils.quaternion_product(
                utils.exponential_map(w_rollout[n,:]*self.dt),q_rollout[n-1,:])
            
            constraint.append(utils.nonholonomic_constraint_equation(dx_rollout[n,:],q_rollout[n,:]))
        
        return x_rollout, dx_rollout, ddx_rollout, q_rollout, w_rollout, dw_rollout, constraint, constraint_forces