import numpy as np 
import scipy.stats as ss 
import treelets.python_treelet_implementation as pytree


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
        

# 
def treelet_decomposition(X, L, abs_ = True): 
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
        J = UH_rotation(C, a, b, components)
        C = pytree.rotate_covariance_matrix(C,a,b,J)
        B = pytree.update_basis(B,J,a,b)
        
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