import numpy as np
import utils


#
def linear_mixture_model(V,C,sigma,n): 
    """
    Returns n draws form the model specified in example 1. 
    Args: 
        - V: pxk matrix of loading vectors
        - C: correlation structure of factors
        - sigma: variance of error term
        - n: sample size to draw
    """
    
    p,k = V.shape
    L = np.linalg.cholesky(C)
    W = np.random.normal(0,1,n*k).reshape(n,k)
        
    noise_component = sigma*np.random.normal(0,1,n*p).reshape(n,p)
    U = np.array([np.matmul(L,W[i,:]) for i in range(n)])
    factor_component = [factor_mult(V, U[i,:]) for i in range(n)]
    
    X = factor_component + noise_component
    
    return X


#
def factor_mult(V, U): 
    """
    Multiples factors and loadings. 
    Args: 
        - V: pxk matrix of loading vectors
        - U: kx1 vector of factors
    """
    
    k = len(U)
    X = [U[i]*V[:,i] for i in range(k)]
    
    return sum(X)