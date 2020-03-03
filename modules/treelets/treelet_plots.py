import numpy as np
import matplotlib.pyplot as plt

def loadings(loadings, joins = True): 
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
