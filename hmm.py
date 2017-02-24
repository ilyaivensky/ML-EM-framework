#!/usr/bin/env python

import numpy as np

class HMM(object):
    """
    HMM - generic implementation
    """
    
    def __init__(self, underlying_model, pi, A):
        """ Initialize the parameters of HMM

        :param underlying_model : observations model (e.g. GMM)
        :param pi : initial state probabilities
        :param A : transition probabilities
        """
        # Number of states
        self.N = len(model.pi)
        
        # Sequence model
        self.pi = pi
        self.A = A
        
        # Observation mondel
        self.obs_model = underlying_model
        self.mu = self.obs_model.mu
        self.name = 'HMM-' + self.obs_model.name
        
    def numStates(self):
        return self.N
     
    def dump(self):
        print 'pi', self.pi
        print 'mu', self.mu
        print 'sigma'
        print self.obs_model.sigma
        print 'transitions'
        print self.A
        
    def classify(self, X):
    
        (gamma, xi), logliks = self.expect(X)
        Z = np.argmax(gamma, axis=1)
        return Z, gamma, logliks
    
    def decode(self, X):
        
        """
        Viterbi decoding
        """
        
        T = X.shape[0]
        B = np.matrix(np.zeros((T, self.N)))
        for t in range(T):
            #initialize conditional probabilities of observations for each state
            B[t,:] = np.log(self.obs_model.predict_one(X[t])[1])
        
        V = np.matrix(np.zeros((T,self.N), dtype=[('v',np.float32),('b',np.int32)]))
        print self.pi
        V[0,:]['v'] = np.log(self.pi) + B[0,:]
        print 'V[0,:][\'v\']',  V[0,:]['v']
        for t in range(1, T):
            for j in range(self.N):
                V[t,j]['v'], V[t,j]['b'] = max((V[t-1,i]['v']+np.log(self.A[i,j])+B[t,j], i) for i in range(self.N))
         
        # Restore the path by tracing back
        path = np.zeros(T, dtype=np.int32)               
        path[T-1] = np.argmax([V[T-1,j]['v'] for j in range(self.N)])
        
        for t in range(T-1,0,-1):
            path[t-1] = V[t,path[t]]['b']
        
        # None is for compatibility with classify(X)    
        return path, None, V[-1,path[-1]]['v']
        
    def expect(self, X):   
        
        """
        Calculate tau=(gamma, xi) for each example and for each class (i.e. p(z_{k}|x_{i})) 
        and log-likelihood (i.e. \sum_{i}^{N} \ln \sum_{k}^{K} \pi_{k}p(x_{i}|z_{k}))
        """
        
        # probabilities p(x_{t}|z_{k})
        B = np.matrix(np.zeros((X.shape[0], self.N)))
        
        T = X.shape[0]
        
        for t in range(T):
            #initialize conditional probabilities of observations for each state
            B[t] = self.obs_model.predict_one(X[t])[1]
                    
        # Using scaled version of alfas and betas
        # For details see Rabiner Tutorial on HMM
        alphas = np.matrix(np.zeros((T, self.N)))
        betas = np.matrix(np.zeros((T, self.N)))
        scales = np.zeros(T)
        
        def alpha_chain():
            """
            Calculates joint probability of sequence x_{0}, ..., x_{t} and being at state k at time t
            (i.e. p(x_{0}, ...,x_{t},z_{t}=k) 
            """
        
            for t in range(T):
                local_alpha = np.zeros(self.N)
                for j in range(self.N):
                    p_cond = B[t,j]
                    if t == 0:
                        local_alpha[j] = self.pi[j] * p_cond
                    else:
                        # For each state j: consider all possible previous states
                        local_alpha[j] = sum((alphas[t-1,i] * self.A[i,j] * p_cond) for i in range(self.N))
                      
                alpha_sum = sum(local_alpha)
                c = 1.0 / alpha_sum  
                      
                alphas[t,:] = c * local_alpha 
                scales[t] = c  
                
        def beta_chain():
        
            for t in reversed(range(T)):
                if t+1 == T:
                    betas[t,:] = [scales[t] * 1.0 for k in range(self.N)]
                else:
                    # For each state j: consider all possible previous states
                    for i in range(self.N):
                        beta = sum((betas[t+1,j] * B[t+1,j] * self.A[i,j]) for j in range(self.N))
                        betas[t,i] = scales[t] * beta    
                        
        def gamma(t):
            """
            Smoothing. 
            Calculates probability of being in state i on step t given all data points
            (i.e. returns p(z_{t}|x) for each state i)
            """
            s = sum([alphas[t,i] * betas[t,i] for i in range(self.N)])
            return [(alphas[t,i] * betas[t,i] / s) for i in range(self.N)]
    
        def xi(t):
            """
            Calculates posterior probability of transition from state i to state j at step t
            """
            xit = np.zeros((self.N, self.N))
            s = sum([alphas[t,k] * sum([self.A[k,s] * B[t+1,s] * betas[t+1,s] for s in range(self.N)]) for k in range(self.N)]) 
            
            for i in range(xit.shape[0]):
                for j in range(xit.shape[0]):
                    xit[i,j] = alphas[t,i] * self.A[i,j] * B[t+1,j] * betas[t+1,j] / s
            return xit
    
        alpha_chain()
        beta_chain()
        
        xis = [xi(t) for t in range(T-1)]

        gammas = np.zeros((T, self.N))
        for i in range(T):
            gammas[i,:] = gamma(i)

        return (gammas, xis), -sum([np.log(c) for c in scales]) / T
    
    def maximize(self, X, tau):
        
        """
        Updates model
        """
        
        # unzip tau
        gamma = tau[0]
        xi = tau[1]
        
        T = X.shape[0] 
        
        A = np.zeros((self.N,self.N))
        pi = np.zeros(self.N)
        sum_gamma1 = np.zeros(self.N)
        
        for i in range(self.N):
            pi[i] = gamma[0,i]
            sum_gamma1[i] = sum(g for g in gamma[:T-1,i])
        
        # maximize A   
        for i in range(self.N):
            for j in range(self.N):
                sum_a_ij = sum(xit[i,j] for xit in xi)
                A[i,j] = sum_a_ij / sum_gamma1[i]   
        
        # update observation model        
        self.obs_model.maximize(X, gamma)
        
        # update sequence model with new estimates
        self.A = A
        self.pi = pi
        self.mu = self.obs_model.mu
    
        
                   