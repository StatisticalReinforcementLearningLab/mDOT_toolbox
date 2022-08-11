# TS_Toolbox_inverse_gamma_v1.py
# The code follows the derivations here: https://www.statlect.com/fundamentals-of-statistics/Bayesian-regression
# This is the first version in an attempt to match with the mDOT toolbox
# All the "underscore" indicates parameters that will NOT be changed
import numpy as np
from statistics import NormalDist
from scipy.linalg import block_diag
from scipy.stats import t

class TS_Toolbox_inverse_gamma():
        
    def action_center_initialization(self, _action_center_ind):
        # Scientists should define the action-center function f(S)
        # In particular, the 1's indicate the tailoring variables
        try:
            if(sum(_action_center_ind==1)+sum(_action_center_ind==0) != len(_action_center_ind)):
                raise ValueError("Input should be a vector of 0's and 1's")
        except ValueError as err:
            raise err
        else:
            self._state_dim=len(_action_center_ind)
            self._action_center_ind=_action_center_ind
            
    
    def parameter_initialization(self, alpha0_mu, alpha0_std_Sigma, beta_mu, beta_std_Sigma, L, var_noise, _upper_clip=1, _lower_clip=0, _fixed_rand_prob=0.5, _fixed_rand_period=0):
        # Initialize the parameters with the configurations provided by the web interface
        # The last elements in alpha0 and beta are the bias terms
        # (VERY IMPORTANT FOR HSINYU) The input std_Sigma's are not yet transformed as a scale of var_noise. The transformation happens in this function.
        # The clip parameter clips the posterior sampling distribution to ensure that both arms are selected with probability p\in [_lower_clip,_upper_clip]
        # The unit of _fixed_rand_period is "decision point." For example, if the personalization starts on day 3, and each day there is 5 decision points, then _fixed_rand_period=3*5=15.
        
        try:
            # The following errors should not occur because of the current design of the interface
            if(self._state_dim+1 != len(alpha0_mu)):
                raise ValueError("The length of alpha0_mu does not match with the number of covariates")
            if(self._state_dim+1 != len(alpha0_std_Sigma)):
                raise ValueError("The length of alpha0_std_Sigma does not match with the number of covariates")
            if(sum(self._action_center_ind)+1 != len(beta_mu)):
                raise ValueError("The length of beta_mu does not match with f(S)")
            if(sum(self._action_center_ind)+1 != len(beta_std_Sigma)):
                raise ValueError("The length of beta_Sigma does not match with f(S)")
                
                
            # The following errors depend on the input check of the interface
            if(sum(alpha0_std_Sigma>0) != len(alpha0_std_Sigma)):
                raise ValueError("alpha0_Sigma should all be positive")
            
            if(sum(beta_std_Sigma>0) != len(beta_std_Sigma)):
                raise ValueError("beta_Sigma should all be positive")
            if(L <= 0):
                raise ValueError("degree of confidence L should be positive")
            if(var_noise <= 0):
                raise ValueError("var_noise should be positive")
            if(_upper_clip < 0 or _upper_clip >1):
                raise ValueError("_upper_clip should be between 0 and 1")
            if(_lower_clip < 0 or _lower_clip >1):
                raise ValueError("_lower_clip should be between 0 and 1")
            if(_upper_clip <= _lower_clip):
                raise ValueError("_lower_clip should be smaller than _upper_clip")
            if(_fixed_rand_prob < 0 or _fixed_rand_prob >1):
                raise ValueError("_fixed_rand_prob should be between 0 and 1")
            if (float(_fixed_rand_period).is_integer()==False or _fixed_rand_period < 0):
                raise ValueError("_fixed_rand_period should be a non-negative integer")
        except ValueError as err:
            raise err
        else:
            self._alpha_len=len(alpha0_mu)+len(beta_mu)
            theta_mu=np.concatenate((alpha0_mu,beta_mu),axis=0)
            # the dimension of _theta_mu_ini is ( len(alpha0_mu)+2*len(beta_mu) )x1
            self._theta_mu_ini=np.concatenate((theta_mu,beta_mu),axis=0)
            
            # This is where we consider the Sigma's as a scale of the noice variance
            alpha_new_Sigma=np.diag((alpha0_std_Sigma**2)/var_noise)
            beta_new_Sigma=np.diag((beta_std_Sigma**2)/var_noise)
            theta_Sigma=block_diag(alpha_new_Sigma,beta_new_Sigma)
            # the dimension of _theta_Sigma_ini is ( len(alpha0_mu)+2*len(beta_mu) ) x ( len(alpha0_mu)+2*len(beta_mu) )
            self._theta_Sigma_ini=block_diag(theta_Sigma,beta_new_Sigma)
            
            self._L_ini = L
            self._noise_ini = var_noise
            self._upper_clip = _upper_clip
            self._lower_clip = _lower_clip
            self._fixed_rand_prob = _fixed_rand_prob
            self._fixed_rand_period = int(_fixed_rand_period)
        
        
    def baseline(self, state):
        # The algorithm calls this function
        # This function adds the bias term
        # return g(S)
        tmp_vct=np.copy(state)
        tmp_vct=np.concatenate((state, np.ones((1,1))),axis=0)
        return tmp_vct
        
    def action_center(self, state):
        # The algorithm calls this function
        # This function chooses the tailoring variables and adds the bias term
        # return f(S)
        tmp_vct=np.copy(state)
        idx=(self._action_center_ind==1)
        tmp_vct=tmp_vct[idx.flatten(),:]
        tmp_vct=np.concatenate((tmp_vct, np.ones((1,1))),axis=0)
        return tmp_vct
    
    
    def decision(self, state, theta_mu, theta_Sigma, L, noise, fixed_rand_count):
        # This function is called at decision time
        # theta_mu, theta_Sigma, L, noise are the parameters
        # Currently we do not handle availability. The mobile app handles this.
        # I call several "underscore parameters e.g., _alpha_len" here. You may want to make them part of the inputs.
        
        fixed_rand_count = fixed_rand_count+1
        
        # This is the personalization period
        if(fixed_rand_count>self._fixed_rand_period):
            # We choose the beta's from theta
            beta_mu=theta_mu[self._alpha_len:]
            beta_Sigma=theta_Sigma[self._alpha_len:,:][:,self._alpha_len:]
        
            # mu_t and Sigma_t are associated with the f(S)*beta
            mu_t=np.matmul(np.transpose(self.action_center(state)),beta_mu)
            Sigma_t=np.matmul(np.transpose(self.action_center(state)),beta_Sigma)
        
            # Notice that the posterior variance of f(S)*beta is scaled by noise
            Sigma_t= noise*np.matmul(Sigma_t,self.action_center(state))
        
            # f(S)*beta is a multivariate t distribution with mean mu_t, variance Sigma_t, and degree of freedom L
            pi=1-t.cdf(0, L, loc=mu_t, scale=Sigma_t)
            pi=max(self._lower_clip,pi)
            pi=min(self._upper_clip,pi)
            
        # This is the fixed randomization period
        else:
            pi=self._fixed_rand_prob
        
        action=np.random.binomial(1,pi)
        
        # You need to update fixed_rand_count and pi. These two are individualized parameters as well!
        return action, pi, fixed_rand_count
                
    
    def update(self, state_set, action_set, reward_set, pi):
        # This function is called at update time
        # All the sets are "list." As discussed, the set includes all the previous individual data. We are NOT doing incremental update.
        # The for loop is terrible I admit. It most likely can be optimized.
        # Notice that there are a lot of "underscore parameters." Feel free to make them inputs to this function.
        
        state=state_set[0]
        action=action_set[0]
        Phi=self.reward_model(state,action, pi)
        reward=reward_set[0]
        reward=np.expand_dims(reward,axis=0)
        reward_all=np.copy(reward)
        for i in range(1,len(state_set)):
            state=state_set[i]
            action=action_set[i]
            Phi=np.concatenate((Phi,self.reward_model(state,action, pi)),axis=1)
            reward=reward_set[i]
            reward=np.expand_dims(reward,axis=0)
            reward_all=np.concatenate((reward_all,reward),axis=0)
        
        # Phi is a numpy array with dimension len(self._theta_mu_ini) x len(state_set) where len(state_set) is the number of data points.
        # reward_all is a numpy array with dimension len(state_set) x 1.
        
        # Here we update theta
        theta_Sigma=np.linalg.inv(np.linalg.inv(self._theta_Sigma_ini)+np.matmul(Phi,np.transpose(Phi)))
        theta_mu=np.matmul(np.linalg.inv(self._theta_Sigma_ini),self._theta_mu_ini)+np.matmul(Phi,reward_all)
        theta_mu=np.matmul(theta_Sigma,theta_mu)
        
        # Here we update noise
        L=self._L_ini+len(state_set)
        tmp0=reward_all-np.matmul(np.transpose(Phi),self._theta_mu_ini)
        tmp=np.linalg.solve(np.matmul(np.matmul(np.transpose(Phi),self._theta_Sigma_ini),Phi)+np.identity(len(state_set)), tmp0)
        noise=1/L*(len(state_set)*self._noise_ini+np.matmul(np.transpose(tmp0),tmp))
        
        
        return theta_mu, theta_Sigma, L, noise
                
        
    def reward_model(self, state, action, pi):
        # The algorithm calls this function
        # This assumes Eq.(3) in Peng's paper
        Phi=np.concatenate((self.baseline(state),pi*self.action_center(state)) ,axis=0)
        Phi=np.concatenate((Phi,(action-pi)*self.action_center(state)),axis=0)
        return Phi

    
    def return_ini_params(self):
        return self._theta_mu_ini, self._theta_Sigma_ini, self._L_ini, self._noise_ini