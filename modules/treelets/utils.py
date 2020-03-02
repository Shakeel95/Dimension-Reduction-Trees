import numpy as np
import matplotlib.pyplot as plt

def cov2cor(C): 
    """
    Converts covraiance matrix (numpy) to correlation matrix. 
    Args: 
        - C: variance-covaraince matrix
    """
    
    d = np.power(np.diag(C), -0.5)
    D = np.diag(d)
    CC = np.matmul(np.matmul(D,C),D)
    
    return CC


def plot_loadings(loadings, joins = True): 
    """
    Plots loadings.
    """
    
    p,l = loadings.shape
    loadings = np.abs(loadings)
    for l in range(l): 
        plt.scatter(np.arange(p),loadings[:,l], label = "Basis " + str(l+1))
        if joins: 
            plt.plot(np.arange(p),loadings[:,l])

    plt.xlabel("component")
    plt.ylabel("loading")
    plt.legend()
