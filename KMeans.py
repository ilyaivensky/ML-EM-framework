#!/usr/bin/env python

import numpy as np

def scale(x, min_x, max_x):
    """
    Scale x to the range [min_x, max_x), assuming x in the range [0,1)
    """
    delta = np.subtract(max_x, min_x)
    return np.add(np.multiply(x, delta), min_x) 
    
class KMeans(object):
    
    def __init__(self, k, mu):
        self.k = k
        self.mu = mu
        self.name = 'K-means'
        
    def numStates(self):
        return self.k   
        
    def predict_one(self, x):   
      
        d = np.zeros(self.mu.shape[0])
        for i in range(self.mu.shape[0]):
            # calculate Eudlidean distance
            d[i] = np.linalg.norm(np.subtract(x,self.mu[i]))
            
        # return index of minimal value
        return np.argmin(d), d
    
    def classify(self, X):
        
        predictions = [self.predict_one(x) for x in X]
        return np.array([p[0] for p in predictions]), None, np.array([p[1] for p in predictions])
    
    def expect(self, X):
        """
        Update classes
        """
        Y = np.zeros((X.shape[0], 1), dtype=int)
        sum_dist = 0
        for i, x in enumerate(X):
            Y[i], d = self.predict_one(x)
            sum_dist += sum(d)
            
        return Y, X.shape[0] / sum_dist
        
    def maximize(self, X, Y):
        """
        Update centers
        """
        mu = np.matrix(np.zeros((self.k,X.shape[1])))
        for i in range(self.k):
            Xk = np.matrix(np.array([x for x,y in zip(np.asarray(X),Y) if y==i]))
            mu[i] = np.mean(Xk, axis=0)
        
        self.mu = mu
                


