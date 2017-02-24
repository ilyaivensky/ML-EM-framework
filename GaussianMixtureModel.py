#!/usr/bin/env python

import math
import numpy as np

def mahalanobis_distance(x, mu, sigma_inv):
    """
    Calculate (x-mu)*sigma_inv*(x-mu)
    """
    delta = np.subtract(x, mu)
    return np.dot(np.dot(delta, sigma_inv), delta.T)
  
class GMM(object):
    """
    Gaussiam Mixture Model
    Can be trained by EM algorithm as a stand-alone model, 
    and as an underlying model in HMM
    """
    
    def __init__(self, pi, mu, sigma):
        self.k = len(pi)
        self.pi = pi
        self.mu = mu
        self.sigma = sigma
        self.name = 'GMM'
        
        # some useful cache
        self.twoPI = math.pow(2 * np.pi, mu.shape[1] / 2)
        self.sigma_inv = [np.linalg.inv(s) for s in self.sigma]
        self.sigma_det = [np.linalg.det(s) for s in self.sigma]
    
    def numStates(self):
        return self.k
    
    def dump(self):
        print 'pi', self.pi
        print 'mu', self.mu
        print 'sigma', self.sigma
    
    def predict_one_given_class(self, x, j, debug=False):
        """
        Calculates p(x|z=j) ~ N(mu[j], sigma[j])
        """
        md = mahalanobis_distance(x, self.mu[j], self.sigma_inv[j]) 
        if debug: print 'mu-sigma-mu GMM',  np.dot(np.dot(self.mu[j],self.sigma_inv[j]),self.mu[j].transpose())
        p = np.exp(-0.5 * md) / np.sqrt(self.twoPI * self.sigma_det[j])
        return np.asscalar(p)
        
    def predict_one(self, x):
        """
        Calculates p(x,z) = p(x|z)*p(z) for each z
        """
        p_joint = np.zeros(self.k)
        p_cond = np.zeros(self.k)
        for j in range(self.k):
            p_cond[j] = self.predict_one_given_class(x,j)
            p_joint[j] = self.pi[j] * p_cond[j]
        return p_joint, p_cond


    def classify(self, X):
        """
        predicts most likely class
        """
        tau, logliks = self.expect(X)
        Z = np.argmax(tau, axis=1)
        return Z, tau, logliks
    
    def expect(self, X):
        """
        Calculate tau for each example and for each class (i.e. p(z_{k}|x_{i})) 
        and log-likelihood (i.e. \sum_{i}^{N} \ln \sum_{k}^{K} \pi_{k}p(x_{i}|z_{k}))
        """
        tau = np.zeros((X.shape[0], self.k))
        logliks = np.zeros((X.shape[0], 1))
                
        for i in range(X.shape[0]):
            p_joint,p_cond = self.predict_one(X[i]) #p(x,z) and p(x|z)
            # normalize across classes
            p_x = sum(p_joint) #p(x_{i}) = \sum_{k}^{K} p(z_{k}) p(x_{i}|z_{k})
            tau[i] = p_joint / p_x  #p(z|x)
            logliks[i] = np.log(p_x)
        
        return tau, np.asscalar(sum(logliks) / X.shape[0])
        
    def maximize(self, X, tau):
        
        pi = np.zeros(self.k)    
        mu = np.zeros((self.k,X.shape[1]))
        sigma = []
        
        for j in range(self.k):
            tau_sum_j = sum(t for t in tau[:,j])
            pi[j] = tau_sum_j / X.shape[0]
            weighted_emp_mean = sum(x*t for x,t in zip(X, tau[:,j]))
            mu[j] = weighted_emp_mean / tau_sum_j
            deltas = [x - self.mu[j] for x in X]
            deltas_sq = [np.dot(d.T, d) for d in deltas]
            
            tau_delta_sum = sum(np.dot(t,d) for t,d in zip(tau[:,j], deltas_sq))
            sigma.append(tau_delta_sum / tau_sum_j)
            
        self.pi = pi
        self.mu = mu 
        self.sigma = sigma
        self.twoPI = math.pow(2 * np.pi, mu.shape[1] / 2)
        self.sigma_inv = [np.linalg.inv(s) for s in self.sigma]
        self.sigma_det = [np.linalg.det(s) for s in self.sigma]