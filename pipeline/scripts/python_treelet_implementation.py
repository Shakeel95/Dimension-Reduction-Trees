import os 
import numpy as np
import pandas as pd
import utils 


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


def treelet_decomposition(X, L, abs_ = False): 
    """
    Performs treelet decomposition to a specified depth. 
    Args:
        - X: nxp array of observations 
        - L: treelet depth ∈ {1,...,p}
        - abs_: bool, determines whether 
        
    Returns a nested dictionary for the treelet decomposition at each level. 
        - C: estimated correlation matrix at level
        - J: rotation performed at level
        - pair: variables were merged
        - order: {0 = sum variable, 1 = difference varaible}
    """
    
    if (X.shape[0] == X.shape[1]): 
        C = X 
    else : 
        C = np.cov(X)
        
    mask = []
    
    treelet = {0:{"C": C, 
                  "J": None,
                  "pair": (None, None), 
                  "order": (None, None)}
              }
    
    for l in range(1,L):
        
        which_max = np.triu(C, +1)
        if abs_:
            which_max = np.abs(which_max)
        k = (which_max == 0)
        which_max[k] = -1 
            
        if (l > 1):
            which_max[mask,:] = -1
            which_max[:,mask] = -1 
            
        a, b = np.unravel_index(np.argmax(which_max, axis = None), which_max.shape)
        J = jacobi_rotation(C, a, b)
        C = rotate_covariance_matrix(C,a,b,J)
        
        treelet[l] = {"C": C,
                      "J": J,
                      "pair": (a,b), 
                      "order": (1,0) if C[a,a] > C[b,b] else (0,1)
                     }
        
        mask += [b if C[a,a] > C[b,b] else a]
        
    return treelet