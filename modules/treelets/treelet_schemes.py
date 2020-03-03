import numpy as np
import scipy.stats as ss
import sys
import treelets.helper_functions as hf 
import treelets.basis_selection as bs 


#
def treelet(X, L, abs_ = False): 
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
        
        cc = hf.cov2cor(C)
        which_max = np.triu(cc, +1)
        if abs_:
            which_max = np.abs(which_max)
        k = (which_max == 0)
        which_max[k] = -1 
            
        if (l > 1):
            which_max[difference_vars,:] = -1
            which_max[:,difference_vars] = -1 
            
        a, b = np.unravel_index(np.argmax(which_max, axis = None), which_max.shape)
        J = hf.jacobi_rotation(C, a, b)
        C = hf.rotate_covariance_matrix(C,a,b,J)
        B = hf.update_basis(B,J,a,b)
        
        treelet[l] = {"C": C,
                      "J": J,
                      "B": B, 
                      "pair": (a,b), 
                      "order": ("sum","difference") if C[a,a] > C[b,b] else ("difference","sum")
                     }
        
        difference_vars += [b if C[a,a] > C[b,b] else a]        
        
    return treelet


#
def tail_greedy_treelet(X, rho, abs_ = False): 
    """
    Performs tail greedy treelet decomposition to a specified depth. 
    N.B. tree is always grown to maximum depth
    Args:
        - X: nxp array of observations 
        - rho: determines number of pairwise mergest to peform
        - abs_: bool, determines whether absolute correlations are used
        
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
    p = depth = len(C)
    difference_vars = []
    l = 0
    
    treelet = {0:{"C": C, 
                  "J": None,
                  "B": np.identity(p),
                  "pairs": (None, None), 
                  "order": (None, None)}
              }
    B = np.identity(p)
    
    while (depth > 0):
        
        l += 1 
        cc = hf.cov2cor(C)
        which_max = np.triu(cc, +1)
        if abs_:
            which_max = np.abs(which_max)
        k = (which_max == 0)
        which_max[k] = -1 
            
        if (depth<p):
            which_max[difference_vars,:] = -1
            which_max[:,difference_vars] = -1 
        
        max_pairs = np.ceil(rho*depth)
        pairs = hf.merger_pairs(which_max,max_pairs)
        J = hf.multiple_jacobi_rotation(C,pairs)
        
        holder = np.matmul(J.transpose(),np.matmul(C,J))
        C = holder 
        holder = np.matmul(B,J)
        B = holder
        
        diff, orders = hf.get_difference_vars(C,pairs)
        difference_vars += diff
        depth -= len(diff)
        
        treelet[l] = {"C": C,
                      "J": J,
                      "B": B, 
                      "pairs": pairs, 
                      "orders": orders
                     }
        
    return treelet


# 
def unbalanced_haar_treelet(X, L, abs_ = True): 
    """
    Performs treelet decomposition 
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
    p = len(C)
    difference_vars = []
    components = [1]*p
    
    treelet = {0:{"C": C, 
              "J": None,
              "B": np.identity(p),
              "pair": (None, None), 
              "order": (None, None)}
          }
    B = np.identity(p)
    for l in range(1,L): 
        
        cc = hf.cov2cor(C)
        which_max = np.triu(cc, +1)
        if abs_:
            which_max = np.abs(which_max)
        k = (which_max == 0)
        which_max[k] = -1
        
        if (l > 1): 
            which_max[difference_vars,:] = -1
            which_max[:,difference_vars] = -1
            
        a, b = np.unravel_index(np.argmax(which_max, axis = None), which_max.shape)
        J = hf.UH_rotation(C, a, b, components)
        C = hf.rotate_covariance_matrix(C,a,b,J)
        B = hf.update_basis(B,J,a,b)
        
        components[a] += 1
        components[b] -= 1
        
        treelet[l] = {"C": C,
              "J": J,
              "B": B, 
              "pair": (a,b), 
              "order": ("sum","difference")
             }
        
        difference_vars += [b]
        
    return treelet