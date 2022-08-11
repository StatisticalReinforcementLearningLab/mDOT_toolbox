# TS_Toolbox_Data_Generation_v1.py
import numpy as np
from statistics import NormalDist
from scipy.linalg import block_diag

class TS_Toolbox_Data_Generation():
    
    def __init__(self,seed):
        self.rng = np.random.default_rng(seed)
        
        
    def action_center_initialization(self, _action_center_ind):
        # Scientists should define the action-center function f(S)
        # There should be an interface to translate their model to the code here
        try:
            if(sum(_action_center_ind==1)+sum(_action_center_ind==0) != len(_action_center_ind)):
                raise ValueError("Input should be a vector of 0's and 1's")
        except ValueError as err:
            raise err
        else:
            self._state_dim=len(_action_center_ind)
            self._action_center_ind=_action_center_ind

    
        
    def baseline(self, state):
        # I would need to include the bias term
        # return g(S)
        tmp_vct=np.copy(state)
        tmp_vct=np.concatenate((tmp_vct, np.ones((1,1))),axis=0)
        return tmp_vct
        
    def action_center(self, state):
        # I would need to include the bias term
        # return f(S)
        tmp_vct=np.copy(state)
        idx=(self._action_center_ind==1)
        tmp_vct=tmp_vct[idx.flatten(),:]
        tmp_vct=np.concatenate((tmp_vct, np.ones((1,1))),axis=0)
        return tmp_vct
        
                
        
    def reward_model(self, state, action):
        # We do NOT assume an action-centered model
        Phi=np.concatenate((self.baseline(state),action*self.action_center(state)),axis=0)
        return Phi
    
    def reward_model_sampling(self, state, action, alpha0, beta, noise):
        tmp_vct=np.zeros((len(alpha0)+len(beta),1))
        tmp_vct[:len(alpha0)]=alpha0
        tmp_vct[len(alpha0):]=beta
        mu=np.matmul(np.transpose(self.reward_model(state,action)),tmp_vct)
        reward=self.rng.normal(mu[0],noise,1)
        return reward
    
    
    def state_sampling(self, state_lower, state_upper):
        return self.rng.uniform(state_lower,state_upper)
