import numpy as np 
from itertools import combinations
import scipy.stats as ss
import math

def ordered_eigen_decomposition(array):
    """
    Returns the ordered eigen decomposition of a square matrix.
    Args: 
        - array: (n x n) array
    """
    
    eig_values, eig_vectors = np.linalg.eig(array)
    idx = eig_values.argsort()[::-1]
    
    return eig_vectors[:,idx]


#
def max_weight_clique(cov,k, mask):
    """
    Finds the k most correlated variables from a group of n. 
    Args:
        - k: int, size of clique
        - cov: (n x n) array
        - maks: set, vertices to ignore
    """
    
    d = np.power(np.diag(cov),-0.5)
    D = np.diag(d)
    cor = np.matmul(np.matmul(D,cov),D)
    
    N = len(cov)
    vertices = {i for i in range(N)} - mask
    cliques = list(combinations(vertices,k))
    
    k_clique = best_k_clique = cliques[0]
    best_edge_sum = sum([abs(cov[i]) for i in combinations(k_clique,2)])
    
    for clique in cliques[1:]:
        edge_sum = sum([abs(cov[i]) for i in combinations(clique,2)])
        if (edge_sum > best_edge_sum): 
            best_edge_sum = edge_sum
            best_k_clique = clique
    
    return set(best_k_clique)


#
def  update_basis_rotate_cov(J,basis,cov):
    """
    What is does...
    Args: 
        - J: (q x q) orthonormal array
        - basis: (n x n) orthonormal array
        - cov: (n x n) s
    """
    
    basis = np.matmul(basis,J)
    cov = np.matmul(J.T,cov)
    cov = np.matmul(cov,J)
    
    return basis, cov


#
def build_block_cov(cov,block): 
    """
    Builds covariance matrix for set of variables 
    Args: 
        - cov: (n x n) array
        - blocks: set
    """
    
    B, block = len(block), list(block)
    block_cov = np.zeros((B,B))
    
    for i in range(B):
        for j in range(B): 
            block_cov[i,j] = cov[block[i],block[j]]
            
    return block_cov


# 
def build_indexed_Jacobi(block, eig, n): 
    """
    Constructs Jacobi marix from indexed blocks. 
    Args: 
        - block: set, 
        - eig: array
        - n: int
    """
    
    B, block = len(block), list(block)
    J = np.eye(n)
    for i in range(B): 
        for j in range(B): 
            J[block[i],block[j]] = eig[i,j]
    return J

# 
def energy_score(panel, basis): 
    """
    Returns normalised energy scores for the columns of a Dirac basis matrix. 
    Args: 
        - panel: (n x T) array
        - basis: (n x n) array
    """
    
    energy_scores = []
    n, T = panel.shape
    mean_square_prod = sum(np.apply_along_axis(np.linalg.norm,0,panel)**2)
    
    for i in range(n): 
        
        w = basis[:,i]
        mean_cross_prod = sum([np.inner(w,panel[:,i])**2 for i in range(T)])
        energy_scores += [mean_cross_prod / mean_square_prod]
    
    return energy_scores


def variance_score(panel, basis):
    """
    Returns normalised variance scores for linear projections onto Dirac basis matrix. 
    Args: 
        - panel: (n x T) array
        - basis: (n x n) array
    """
    
    variance_scores = []
    n, T = panel.shape
    mean_square_prod = sum(np.apply_along_axis(np.linalg.norm,0,panel)**2)
    
    for i in range(n):
        
        a = basis[:,i]
        z = np.matmul(a.T,panel)
        variance_scores += [np.var(z) / mean_square_prod] 
        
    return variance_scores
        
    
#
def matrix_error(A,B): 
    """
    ...
    """
    n, _ = A.shape
    E = np.abs(A) - np.abs(B)
    error = sum([np.linalg.norm(E[i:])/n for i in range(n)])
    
    return error