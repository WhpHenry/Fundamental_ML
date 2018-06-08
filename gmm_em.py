# coding: utf-8

import math
import copy
import numpy as np
import matplotlib.pyplot as plt

  
# gaussian_mxiture with EM algorithm set: 
#   samples X are created by k gaussian_distribution 
#   with same sigma and dissimilar mean (different center location)
#   each xi in X is vector (xi1, zi1, ..zik)
#   Z=(z_1, z_2, ..., z_k) is unknown variable 
class Simple_Gaussian_Mxiture:
    # init basic parameters, k, mu, sigma, X, Z
    def __init__(self, k=2, xlength=300, iter_times=100, epsilon=0.0001, sigma_times=5, mu_times=100):
        self.k = k
        self.xlength = xlength
        self.iter_times = iter_times
        self.epsilon = epsilon
        self.sigma = np.random.rand()*sigma_times
        self.mu = np.random.random_sample(k)*mu_times
        self.known = np.zeros(xlength)
        self.unknown = np.zeros((xlength, k)) 
    
    def generate_known(self):
        for i in range(len(self.known)):
            self.known[i] = np.random.normal()*self.sigma + self.mu[np.random.randint(self.k)]
    
    # Expection: set value of all z1-zk
    #   Qi(zi) = p(zi|xi,theta_i) = p(zi,xi,theta)/p(xi,theta)   
    #   H(theta_i) = sum[p(zi|xi,theta_i)log[sum(p/Q)]]
    def expection(self):
        for i in range(len(self.unknown)):
            p_xi_theta = 0
            for c in range(self.k):
                p_xi_theta += math.exp((-1/(2*(float(self.sigma**2))))*(float(self.known[i]-self.mu[c]))**2)
            for j in range(self.k):
                p_z_xi_theta = math.exp((-1/(2*(float(self.sigma**2))))*(float(self.known[i]-self.mu[j]))**2)
                self.unknown[i, j] = p_z_xi_theta/p_xi_theta

    # Maximization:  update theta based on zi and xi
    #   theta_i+1 = argmax(sum[p(zi|xi,theta_i+1)log[sum(p/Q)]])
    #   H(theta_i+1) >= sum[p(zi|xi,theta_i+1)log[sum(p/Q)]]
    #   goto last step
    def maximization(self):
        for i in range(self.k):
            numerator  = sum([self.unknown[j][i]*self.known[j] for j in range(len(self.known))])
            denominator = sum([self.unknown[j][i] for j in range(len(self.known))])
            self.mu[i] = numerator /denominator

    # iterate times setting
    def iterate(self):
        print "init mu are: " , self.mu
        for i in range(self.iter_times):
            old_mu = copy.deepcopy(self.mu)
            self.expection()
            self.maximization()
            print i, self.mu
            if sum(abs(self.mu-old_mu)) < self.epsilon:
                break

    def show_known(self, piece=50):
        plt.hist(self.known, piece)
        plt.show() 

if __name__ == '__main__':
    obj = Simple_Gaussian_Mxiture(k=3)
    obj.generate_known()
    obj.iterate()
    obj.show_known()
