# coding: utf-8
import numpy as np
import scipy.stats as stats

class Distance:

    def __init__(self, arg):
        self.arg = arg
        
    # Minkowski distance 
    # sum{((x-y)^p)^(1/p)}
    # p = 2 ; Euclidean 
    # p = 1 ; Manhattan
    # p = infinite ; Chebyshev   
    def minkowski(self, x, y, p=2):
        try:
            v1 = np.array(x)
            v2 = np.array(y)
            return sum((np.array(x)+np.array(y)) ** p)        
        except Exception as e:
            print("Error: ",e)

    # Mahalanobis
    # d(x,y) = sqrt((x-y)'Matrix_cov(x,y)(x-y))
    '''
    def mahalanobis(self, x, y):
        try:
            v1 = np.array(x)
            v2 = np.array(y)
            return np.sqrt((v1-v2)*np.cov(v1, v2)*(v1-v2))
        except Exception as e:
            print("Error: ",e)
    '''
    
    # Inner of vectors
    def inner(self, x, y):
        try:
            v1 = np.array(x)
            v2 = np.array(y)
            # np.inner(np.array(x), np.array(y))
            return sum(v1*v2)/(np.sqrt(v1*v1)*np.sqrt(v2*v2)) 
        except Exception as e:
            print("Error: ",e)

    # Pearson correlation
    def pearson(self, x, y):
        v1 = np.array(x)-np.mean(x)
        v2 = np.array(v)-np.mean(y)
        try:
            # stats.pearsonr(x, y)[0]
            return sum(v1*v2)/(np.sqrt(v1*v1)*np.sqrt(v2*v2))
        except Exception as e:
            print("Error: ",e)

    # Edit(Levenshtein) distance
    def edit(self, seq1, seq2):
        if (len(seq1) == 0) or (len(seq2) == 0):
            return len(seq1)+len(seq2)

        tmp = lambda x,y : 0 if x==y else 1
        mi = np.zeros((len(seq1)+1,len(seq2)+1))

        mi[0][:] = range(len(seq2)+1)
        for i in range(len(seq1)+1):
            mi[i][0] = i

        for i in range(1,len(seq1)+1):
            for j in range(1,len(seq2)+1):
                mi[i][j] = min((mi[i-1][j-1]+tmp(seq1[i],seq2[j])), (mi[i-1][j]+1), (mi[i][j-1]+1))
        print(mi)
        return mi[-1][-1]


    # Dynamic Time Warp
    def dtw(self, seq1, seq2):
        if (len(seq1) == 0) or (len(seq2) == 0):
            return sum(seq1)+sum(seq2)

        dis = lambda x,y : 0 if x == y else np.abs(x-y)
        mi = np.zeros((len(seq1)+1,len(seq2)+1))
        
        for i in range(len(seq1)+1):
            mi[i][0] = np.infinite
            mi[i][1:] = [dis(seq1[i], seq2[j]) for j in range(len(seq2))]
        mi[0][0] = 0

        for i in range(1,len(seq1)+1):
            for j in range(1,len(seq2)+1):
                mi[i][j] += min(mi[i-1][j-1], mi[i-1][j], mi[i][j-1])
        print(mi)
        return mi[-1][-1]

    # KL-Divergence (relative entropy)
    # calculate "distance" of two distributions
    def kl(self, distr1, distr2):
        pass

    # Chi-Square
    def chi(self):
        pass