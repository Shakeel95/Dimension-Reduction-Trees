import numpy as np
import scipy.stats as ss
import utils
import python_treelet_implementation as pytree


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
        - order: of pair is sum / difference variable
    """
    
    if (X.shape[0] == X.shape[1]): 
        C = X 
    else : 
        C = np.cov(X, rowvar = False)
    p = len(C); 
    difference_vars = []
    
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
            which_max[difference_vars,:] = -1
            which_max[:,difference_vars] = -1 
            
        a, b = np.unravel_index(np.argmax(which_max, axis = None), which_max.shape)
        J = jacobi_rotation(C, a, b)
        C = rotate_covariance_matrix(C,a,b,J)
        B = update_basis(B,J,a,b)
        
        treelet[l] = {"C": C,
                      "J": J,
                      "B": B, 
                      "pair": (a,b), 
                      "order": ("sum","difference") if C[a,a] > C[b,b] else ("difference","sum")
                     }
        
        difference_vars += [b if C[a,a] > C[b,b] else a]        
        
    return treelet