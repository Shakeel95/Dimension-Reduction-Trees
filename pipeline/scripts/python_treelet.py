import os 
import numpy as np
import pandas as pd 


def jacobi_rotation(C,a,b):
    """
    Finds a 2x2 Jacobi rotation matrix that decocorrelates two variables. 
    Args: 
        - C: variance-covariance matrix
        - a: index of first variable
        - b: index of second variable
    """
    
    p = len(C)
    C_aa = C[a,a]
    C_bb = C[b,b]
    C_ab = C[a,b]
    
    theta = 0.5*np.arctan(2*C_ab/(C_aa - C_bb))
    cos_theta = np.cos(theta)
    sine_theta = np.sin(theta)
    
    rotation = np.array([[cos_theta, -sine_theta],
                  [sine_theta, cos_theta]])
    
    return rotation


def rotate_covariance_matrix(C, a, b, J):
    """
    De-correlates two variables given their index and a Jacobi rotation matrix. 
    Args: 
        - J: 2x2 Jacobi rotation matrix
        - C: variance-covaraince matrix 
        - a: index of first variable
        - b: index of second variable
    """
    
    
    p = C.shape[0]
    R = np.identity(p)
    R[a,a], R[a,b], R[b,a], R[b,b] = J.flatten()
    C = np.matmul(np.matmul(R.transpose(),C),R)
    
    return C


def treelet_decomposition(X, L): 
    """
    Performs treelet decomposition to a specified depth. 
    Args:
        - X: nxp array of observations 
        - L: treelet depth ∈ {1,...,p} 
        
    Returns a nested dictionary for the treelet decomposition at each level. 
        - C: estimated correlation matrix at level
        - J: rotation performed at level
        - pair: variables were merged
        - order: {0 = sum variable, 1 = difference varaible}
    """
    
    C = np.cov(X)
    n,p = X.shape
    mask = []
    treelet = {0:{"C": C, 
                  "J": None,
                  "pair": (None, None), 
                  "order": (None, None)}
              }
    
    for l in range(1,L):
        
        # find most correlated vars 
        which_max = np.abs(np.triu(C, +1))
        if (l > 1):
            which_max[mask,:] = -1
            which_max[:,mask] = -1    
        a, b = np.unravel_index(np.argmax(which_max, axis = None), which_max.shape)
        print((a,b))
        
        # calc. Jacobi rotation 
        J = jacobi_rotation(C, a, b)
        
        # update correlation matrix
        C = rotate_covariance_matrix(C,a,b,J)
        
        # update treelet
        treelet[l] = {"C": C,
                      "J": J,
                      "pair": (a,b), 
                      "order": (1,0) if C[a,a] > C[b,b] else (0,1)
                     }
        
        mask += [b if C[a,a] > C[b,b] else a]
        print(mask)
        
    return treelet