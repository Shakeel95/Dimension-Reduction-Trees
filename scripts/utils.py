import numpy as np

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


#
def geometric_series(a,r,n): 
    """
    Returns the first n terms in a geometric series. 
    Args: 
        - a: first term 
        - r: common ratio
        - n: number to terms to generate
    """
    series = []
    for i in range(n): 
        series += [a*(r**i)]
    return series