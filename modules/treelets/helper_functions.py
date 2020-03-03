import numpy as np 
import scipy.stats as ss


#
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
    
    if (C_aa - C_bb) == 0: 
        theta = np.pi/4
    else: 
        theta = 0.5*np.arctan(2*C_ab/(C_aa - C_bb))
    cos_theta = np.cos(theta)
    sine_theta = np.sin(theta)
    
    rotation = np.array([[cos_theta, -sine_theta],
                  [sine_theta, cos_theta]])
    
    return rotation


#
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


#
def update_basis(B,J,a,b):
    """
    Updates basis function.
    Args: 
        - J: 2x2 Jacobi rotation matrix
        - B: pxp matrix representing basis function 
    """
    
    p = B.shape[0]
    R = np.identity(p)
    R[a,a], R[a,b], R[b,a], R[b,b] = J.flatten()
    basis = np.matmul(B,R)
    
    return basis


#
def merger_pairs(similarity, max_pairs): 
    """
    Returns pairs of variables to merge according to tail greedy procedure. 
    Args: 
        - similarity: upper triangular matrix for similarity ranking
        - max_pairs: maxium sum variables to extract
    """
    
    pairs = []
    
    while (max_pairs > 0): 
            a,b = np.unravel_index(np.argmax(similarity, axis = None), similarity.shape)
            pairs += [a,b]
            similarity[a,b] = -1
            max_pairs -= 1
            
    unique_pairs = pairs[:2]
    for i in np.arange(2,len(pairs),2): 
        pair = pairs[i:i+2]
        if not ((pair[0] in unique_pairs) or (pair[1] in unique_pairs)): 
            unique_pairs += pair
        
    return unique_pairs


#
def multiple_jacobi_rotation(C, pairs): 
    """
    Finds multiple Jacobi roatation from list of unique pairs.
    Args: 
        - C: variance-covariance matrix
        - pairs: pairs of variables to merge
    """
    
    p = len(C)
    J = np.identity(p)
    
    for i in np.arange(0,len(pairs),2): 
        a,b = pairs[i:i+2]
        j = jacobi_rotation(C,a,b)
        J[a,a], J[a,b], J[b,a], J[b,b] = j.flatten()
        
    return J


#
def get_difference_vars(C,pairs):
    """
    Extracts difference varaibles from list of pairs and assigns orders. 
    Args: 
        - C: variance-covariance matrix
        - pairs: pairs of varaibles chosen by tail greedy procedure
    """
    
    diff = []
    orders = []
    
    for i in np.arange(0, len(pairs),2): 
        a,b = pairs[i:i+2]
        if (C[a,a] > C[b,b]): 
            diff += [b]
            orders += [("sum","difference")]
        else: 
            diff += [a]
            orders += [("difference","sum")]
    
    return diff, orders


# 
def UH_rotation(C, a, b, components):
    """
    Finds a 2*2 rotation matrix based on UH transform. 
    Performs one row swap and one negation if correlation is negative. 
    Args:
        - C: variance-covariance matrix
        - a: index of first variable
        - b: index of second variable
        - sum_components: number for merges used to poduce sum variables
    """
    
    aa = (components[a] / (components[a]+components[b]))**0.5 
    bb = (components[b] / (components[a]+components[b]))**0.5

    if (C[a,b] > 0):
        rotation = np.array([[aa,-bb],
                              [bb,aa]])
        return rotation
        
    else:
        rotation = np.array([[aa,bb],
                             [-bb,aa]]) 
        return rotation