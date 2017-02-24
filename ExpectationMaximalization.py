#!/usr/bin/env python

import numpy as np
import math
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('error')

class EM(object):
    """
    Expectation-Maximalization trainer
    """
    
    def __init__(self, model):  
        self.model = model 
        self.log = []
            
    def train(self, X, X_test, epsilon):
          
        # Training objective - it can be log-likelihood or any other "profit" objective
        # (i.e. the goal is to maximize the objective) 
        objective_prev = objective = -float("inf")
          
        t = 0          
        while True:
            
            tau, objective = self.model.expect(X)
            
            tau_valid, objective_valid = self.model.expect(X_valid)
            print 'Epoch', t, 'log-lkh', objective, 'valid log-lkh', objective_valid
            self.log.append((objective, objective_valid))
                     
            """
            Stop condition
            """
            if objective <= objective_prev + epsilon:
                break
            
            objective_prev = objective  
            self.model.maximize(X, tau)     
            
            t += 1    