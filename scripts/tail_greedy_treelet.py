import numpy as np
import scipy.stats as ss
import utils
import python_treelet_implementation as pytree


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
        j = pytree.jacobi_rotation(C,a,b)
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
def treelet_decomposition(X, rho, abs_ = False): 
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
        cc = utils.cov2cor(C)
        which_max = np.triu(cc, +1)
        if abs_:
            which_max = np.abs(which_max)
        k = (which_max == 0)
        which_max[k] = -1 
            
        if (depth<p):
            which_max[difference_vars,:] = -1
            which_max[:,difference_vars] = -1 
        
        max_pairs = np.ceil(rho*depth)
        pairs = merger_pairs(which_max,max_pairs)
        J = multiple_jacobi_rotation(C,pairs)
        
        holder = np.matmul(J.transpose(),np.matmul(C,J))
        C = holder 
        holder = np.matmul(B,J)
        B = holder
        
        diff, orders = get_difference_vars(C,pairs)
        difference_vars += diff
        depth -= len(diff)
        
        treelet[l] = {"C": C,
                      "J": J,
                      "B": B, 
                      "pairs": pairs, 
                      "orders": orders
                     }
        
    return treelet