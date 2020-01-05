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

def extract_merge_order(tree): 
    """
    Extracts order in which variables were merged. 
    Args: 
        - tree: nested dictionary returned by treelet function
    """
    
    merge_order = []
    depth = 