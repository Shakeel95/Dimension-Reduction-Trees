import numpy as np
import scipy.stats as ss

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