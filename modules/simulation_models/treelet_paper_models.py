import numpy as np


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


#
def three_correlated_factors(V,sigma,sigma1,sigma2,c1,c2,n): 
    """
    Returns n draws from the model specified in example 2. 
    Args: 
        - V: pxk matrix of loading vectors
        - sigma: standard deviation of noise component
        - sigma1: standard deviation of first factor
        - sigma2: standard deviation of second factor
        - c1: first factor weighting in third factor  
        - c2: second factor weighiting in third factor
        - n: sample size to draw
    """
    
    p,k = V.shape
        
    factor1 = np.random.normal(0,sigma1,n)
    factor2 = np.random.normal(0,sigma2,n)
    factor3 = c1*factor1 + c2*factor2
    U = np.array([factor1,factor2,factor3]).transpose()
    
    factor_component = [factor_mult(V,U[i,:]) for i in range(n)]
    noise_component = np.random.normal(0,sigma,n*p).reshape(n,p)

    X = factor_component + noise_component
    
    return X 


#
def Bair_et_al(V,sigma,n): 
    """
    Returns n draws from the model specified in example 3, with structure taken from Bair et al. (2006)
    Args: 
        - V: pxk matrix of loading vectors
        - sigma: standard deviation of noise component
        - n: sample size os sample to draw
    """
    
    p,k = V.shape
    
    factor1 = np.array([.5*(-1)**np.random.binomial(1,.5) for i in range(n)])
    factor2 = np.array([0 if (np.random.uniform(0,1)<.4) else 1 for i in range(n)])
    factor3 = np.array([0 if (np.random.uniform(0,1)<.3) else 1 for i in range(n)])
    U = np.array([factor1, factor2, factor3]).transpose()
    
    factor_component = [factor_mult(V,U[i,:]) for i in range(n)]
    noise_component = np.random.normal(0,sigma,n*p).reshape(n,p)

    X = factor_component + noise_component
    
    return X 