import numpy as np
import scipy.stats as ss
import utils 


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
        - B: pxp matrix representing Dirac basis
        - pair: variables were merged
        - order: {0 = sum variable, 1 = difference varaible}
    """
    
    if (X.shape[0] == X.shape[1]): 
        C = X 
    else : 
        C = np.cov(X, rowvar = False)
    p = len(C); 
    mask = []
    
    treelet = {0:{"C": C, 
                  "J": None,
                  "B": np.identity(p),
                  "pair": (None, None), 
                  "order": (None, None)}
              }
    B = np.identity(p)
    for l in range(1,L):
        
        cc = utils.cov2cor(C)
        which_max = np.triu(cc, +1)
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
        B = update_basis(B,J,a,b)
        
        treelet[l] = {"C": C,
                      "J": J,
                      "B": B, 
                      "pair": (a,b), 
                      "order": (1,0) if C[a,a] > C[b,b] else (0,1)
                     }
        
        mask += [b if C[a,a] > C[b,b] else a]        
        
    return treelet


#
def energy_score(w,X): 
    """
    Returns normalised energy score for the column of a Dirac basis. 
    Args: 
        - w: px1 vector
        - X: nxp matrix
    """
    n,p = X.shape
    
    mean_cross_prod = sum([np.inner(w,X[i,:])**2 for i in range(n)])
    mean_square_prod = sum(np.apply_along_axis(np.linalg.norm,1,X)**2)
    energy = mean_cross_prod / mean_square_prod
    
    return energy


# 
def best_basis(tree, K, X): 
    """
    Finds best K-basis for a treelet decomposition by maximising the en
    Args: 
        - tree: nested dictionary returned by treelet_decomposition function
        - K: dimesnion of treelet representation
        - X: nxp matrix of observations
        
    Returns: 
        - best_basis: d
        - basis logger: the same information for every K-basis
    """
    
    L = len(tree); p = len(tree[0]["B"])
    best_basis = dict()
    basis_logger = dict()
    
    basis = tree[0]["B"]
    energies = [energy_score(basis[:,i],X) for i in range(p)]
    energy_rankings = ss.rankdata(energies)
    K_best = [i for i in range(p) if energy_rankings[i] > p-K]
    energy = sum([energies[i] for i in K_best])

    basis_logger[0] = {"level": 0, 
                       "energy": energy,
                       "index": K_best,
                       "basis": basis[:,K_best]
                      }
    best_basis = basis_logger[0]
    
    
    for l in range(1,L): 
        
            basis = tree[l]["B"]
            energies = [energy_score(basis[:,i],X) for i in range(p)]
            energy_rankings = ss.rankdata(energies)
            K_best = [i for i in range(p) if energy_rankings[i] > p-K]
            energy = sum([energies[i] for i in K_best])
            
            basis_logger[l] = {"level": l, 
                               "energy": energy, 
                               "index": K_best, 
                               "basis": basis[:,K_best]
                              }
            if (energy > best_basis["energy"]): 
                best_basis = basis_logger[l]
        
    return best_basis, basis_logger