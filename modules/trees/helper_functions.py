import numpy as np

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