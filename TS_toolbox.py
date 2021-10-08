# TS_Toolbox.py
import numpy as np
from statistics import NormalDist
from scipy.linalg import block_diag

class TS_Toolbox():
    def __init__(self):
        self.count=0
        
    def baseline_initialization(self, _baseline_ind):
        # Scientists should define the baseline function g(S)
        # There should be an interface to translate their model to the code here
        try:
            if(sum(_baseline_ind==1)+sum(_baseline_ind==0) != len(_baseline_ind)):
                raise ValueError("Input should be a vector of 0's and 1's")
        except ValueError as err:
            raise err
        else:
            self._state_dim=len(_baseline_ind)
            self._baseline_ind=_baseline_ind
        
    def action_center_initialization(self, _action_center_ind):
        # Scientists should define the action-center function f(S)
        # There should be an interface to translate their model to the code here
        try:
            idx=self._baseline_ind==0
            if(len(_action_center_ind) != self._state_dim):
                raise ValueError("The length of the input should be the same as the dimension of the state")
            elif(sum(_action_center_ind==1)+sum(_action_center_ind==0) != len(_action_center_ind)):
                raise ValueError("Input should be a vector of 0's and 1's")
            elif(sum(_action_center_ind[idx.flatten(),:]==1)>0):
                raise ValueError("f(S) should be a subset of g(S)")
        except ValueError as err:
            raise err
        else:
            self._action_center_ind=_action_center_ind

    def parameter_initialization(self, alpha0_mu, alpha0_Sigma, beta_mu, beta_Sigma, _noise, _upper_clip=1, _lower_clip=0, _update=1):
        # Scientists should input the parameters with the priors
        # There should be an interface to translate their initialization to the code here
        # This assumes that parameters are Gaussian distributed with some mean and covariance matrix, where we assume that $\alpha0\sim$ Normal(alpha0_mu, alpha0_Sigma) and $\beta\sim$ Normal(beta_mu, beta_Sigma)
        # We may need to define the noise parameter ourselves (how)
        # Let's assume that alpha0_Sigma and beta_Sigma are diagonal
        # The clip parameter clips the posterior sampling distribution to ensure that both arms are selected with probability p\in [_lower_clip,_upper_clip]
        # The update parameter determines how often the posterior estimation should be updated
        
        try:
            if(sum(self._baseline_ind)+1 != len(alpha0_mu)):
                raise ValueError("The length of alpha0_mu does not match with g(S)")
            if(sum(self._baseline_ind)+1 != len(alpha0_Sigma)):
                raise ValueError("The length of alpha0_Sigma does not match with g(S)")
            if(sum(alpha0_Sigma>0) != len(alpha0_Sigma)):
                raise ValueError("alpha0_Sigma should all be positive")
            if(sum(self._action_center_ind)+1 != len(beta_mu)):
                raise ValueError("The length of beta_mu does not match with f(S)")
            if(sum(self._action_center_ind)+1 != len(beta_Sigma)):
                raise ValueError("The length of beta_Sigma does not match with f(S)")
            if(sum(beta_Sigma>0) != len(beta_Sigma)):
                raise ValueError("beta_Sigma should all be positive")
            if(_noise <= 0):
                raise ValueError("_noise should be positive")
            if(_upper_clip < 0 or _upper_clip >1):
                raise ValueError("_upper_clip should be between 0 and 1")
            if(_lower_clip < 0 or _lower_clip >1):
                raise ValueError("_lower_clip should be between 0 and 1")
            if(_upper_clip <= _lower_clip):
                raise ValueError("_lower_clip should be smaller than _upper_clip")
            if (float(_update).is_integer()==False or _update <= 0):
                raise ValueError("_update should be a positive integer")
        except ValueError as err:
            raise err
        else:
            self._alpha_len=len(alpha0_mu)+len(beta_mu)
            theta_mu=np.concatenate((alpha0_mu,beta_mu),axis=0)
            self.theta_mu=np.concatenate((theta_mu,beta_mu),axis=0)
            alpha_new_Sigma=np.diag(alpha0_Sigma)
            beta_new_Sigma=np.diag(beta_Sigma)
            theta_Sigma=block_diag(alpha_new_Sigma,beta_new_Sigma)
            self.theta_Sigma=block_diag(theta_Sigma,beta_new_Sigma)
            self._noise=_noise
            self._upper_clip=_upper_clip
            self._lower_clip=_lower_clip
            self._update=int(_update)
        
        
    def baseline(self, state):
        # The controller algorithm calls this function
        # I would need to include the bias term
        # return g(S)
        tmp_vct=np.copy(state)
        idx=(self._baseline_ind==1)
        tmp_vct=tmp_vct[idx.flatten(),:]
        tmp_vct=np.concatenate((tmp_vct, np.ones((1,1))),axis=0)
        return tmp_vct
        
    def action_center(self, state):
        # The controller algorithm calls this function
        # I would need to include the bias term
        # return f(S)
        tmp_vct=np.copy(state)
        idx=(self._action_center_ind==1)
        tmp_vct=tmp_vct[idx.flatten(),:]
        tmp_vct=np.concatenate((tmp_vct, np.ones((1,1))),axis=0)
        return tmp_vct
        
        
        
    def choose_action(self,state,avail):
        # The controller algorithm calls this function each time they want to take an action
        # This is Eq.(2) and it will define self.pi
        if(self.count%self._update==0 and avail==1):
            beta_mu=self.theta_mu[self._alpha_len:]
            beta_Sigma=self.theta_Sigma[self._alpha_len:,:][:,self._alpha_len:]
            mu_t=np.matmul(np.transpose(self.action_center(state)),beta_mu)
            Sigma_t=np.matmul(np.transpose(self.action_center(state)),beta_Sigma)
            Sigma_t=np.matmul(Sigma_t,self.action_center(state))
            pi=1-NormalDist(mu=mu_t, sigma=Sigma_t).cdf(0)
            pi=max(self._lower_clip,pi)
            pi=min(self._upper_clip,pi)
            self.pi=pi
        if(avail==1):
            action=np.random.binomial(1,self.pi)
        else:
            action=0
        self.count=self.count+1
        return action
    
    # I'm not sure if we should let the controller decide when this function would be call. In that case, we do not need to check the time here. 
    def Thompson_sampling(self, state_set, action_set, reward_set):
        # The controller algorithm calls this each time they want to update the posterior estimates
        if(self.count%self._update==0):
            for i in range(len(state_set)):
                state=state_set[i]
                action=action_set[i]
                reward=reward_set[i]
                K_l_numer=np.matmul(self.theta_Sigma,self.reward_model(state,action))
                K_l_denom=self._noise+np.matmul(np.transpose(self.reward_model(state,action)),K_l_numer)
                K_l=(1/K_l_denom)*K_l_numer
                self.theta_Sigma=self.theta_Sigma-np.matmul(K_l,np.transpose(K_l_numer))
                self.theta_mu=self.theta_mu+np.matmul(K_l,\
                                                      reward-np.matmul(np.transpose(self.reward_model(state,action)),self.theta_mu))
                
        
    def reward_model(self, state, action):
        # This is called by Thompson_sampling
        # This assumes Eq.(3) in Peng's paper
        Phi=np.concatenate((self.baseline(state),self.pi*self.action_center(state)) ,axis=0)
        Phi=np.concatenate((Phi,(action-self.pi)*self.action_center(state)),axis=0)
        return Phi
    
    def reward_model_sampling(self, state, action, alpha0, beta):
        tmp_vct=np.zeros((len(alpha0)+len(beta)*2,1))
        tmp_vct[:len(alpha0)]=alpha0
        tmp_vct[len(alpha0):len(alpha0)+len(beta)]=beta
        tmp_vct[len(alpha0)+len(beta):]=beta
        mu=np.matmul(np.transpose(self.reward_model(state,action)),tmp_vct)
        reward=np.random.normal(mu[0],self._noise,1)
        return mu[0]
    
    def return_pi(self):
        return self.pi
    def return_mu(self):
        return self.theta_mu
    def return_Sigma(self):
        return np.diag(self.theta_Sigma)