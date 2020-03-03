import math
import sys
import numpy as np 
from itertools import combinations

sys.path.append("../")
import trees.treelets as tl
from .helper_functions import *

# 
def standard_PCA(panel,n_factors):
    """
    Estimates loadings, factors, and common component in a static factor model using PCA. 
    Args: 
        - panel: (n x T) array
        - n_factors: int
    """
    
    n, _ = panel.shape
    cov = np.cov(panel)
    
    eig_vectors = ordered_eigen_decomposition(cov)
    P = eig_vectors[:,0:n_factors]
    w = np.matmul(P,P.transpose())
    
    est_loadings = np.sqrt(n)*P 
    est_factors = np.matmul(P.transpose(),panel)
    est_common_component = np.matmul(w,panel)
    
    return est_loadings, est_factors, est_common_component


#
def naive_disjoint_blocks(panel, block_size, n_factors):
    """
    Estimates laodings, factors, and common components via blockwise PCA using disjoint blocks. 
    Args: 
        - panel: (n x T) array
        - block_size: int
        - n_factors: int
    """
    
    n, _ = panel.shape 
    basis = np.eye(n)
    cov = np.cov(panel)
    n_blocks = math.ceil(n/block_size)
    
    for i in range(n_blocks):
        a,b = i*block_size, min(n,(i+1)*block_size)
        block_cov = cov[a:b,a:b]
        basis[a:b,a:b] = ordered_eigen_decomposition(block_cov)
    
    energies = variance_score(panel, basis)
    energy_rankings = ss.rankdata(energies)
    best_basis = [i for i in range(n) if energy_rankings[i] > (n-n_factors)]
    
    P = basis[:,best_basis]
    w = np.matmul(P,P.transpose())
    est_loadings = np.sqrt(n)*P 
    est_factors = np.matmul(P.transpose(),panel)
    est_common_component = np.matmul(w,panel)
    
    return est_loadings, est_factors, est_common_component
    
        
#
def naive_sliding_blocks(panel, block_size, n_factors): 
    """
    Estimates loadings, factors, and common components via blockwise PCA with sliding blocks. 
    Args: 
    """
    
    n, _ = panel.shape
    basis = np.eye(n)
    cov = np.cov(panel)
    n_blocks = math.ceil(n/block_size)
    
    for i in reversed(range(n+1-block_size)):
        a,b = i, i+block_size
        block_cov = cov[a:b,a:b]
        
        J = np.eye(n)
        J[a:b,a:b] = ordered_eigen_decomposition(block_cov)
        basis, cov = update_basis_rotate_cov(J, basis, cov)
        
    energies = variance_score(panel, basis)
    energy_rankings = ss.rankdata(energies)
    best_basis = [i for i in range(n) if energy_rankings[i] > (n-n_factors)]

    P = basis[:,best_basis]
    w = np.matmul(P,P.transpose())
    est_loadings = np.sqrt(n)*P 
    est_factors = np.matmul(P.transpose(),panel)
    est_common_component = np.matmul(w,panel)
    
    return est_loadings, est_factors, est_common_component


# 
def ranked_disjoint_blocks(panel, block_size, n_factors): 
    """
    Estimates loadings, factors, and common component via blockwise PCA. 
    Blocks are taken to be variables which are jointly maximally correlated.
    Args: 
        - panel: (n x T) array of observations
        - block_size: int
        - n_factors: int
    """
    
    n, _ = panel.shape
    n_blocks = math.ceil(n/block_size) 
    mask = set()
    basis = np.eye(n)
    cov = np.cov(panel)

    for i in range(n_blocks):
        
        k = min(block_size, n-i*block_size)
        block = max_weight_clique(cov,k,mask)
        block_cov = build_block_cov(cov,block)
        
        eig = ordered_eigen_decomposition(block_cov)
        J = build_indexed_Jacobi(block, eig, n)
        basis, cov = update_basis_rotate_cov(J, basis, cov)
        
        mask |= block
        
    energies = variance_score(panel, basis)
    energy_rankings = ss.rankdata(energies)
    best_basis = [i for i in range(n) if energy_rankings[i] > (n-n_factors)]
    
    P = basis[:,best_basis]
    w = np.matmul(P,P.transpose())
    est_loadings = np.sqrt(n)*P 
    est_factors = np.matmul(P.transpose(),panel)
    est_common_component = np.matmul(w,panel)
    
    return est_loadings, est_factors, est_common_component


def ranked_sliding_blocks(panel, block_size, n_factors):
    """
    Estimates loadings, factors, and common components via blockwise. 
    In each pass the blocks are chosen to be maximally correlated. 
    Args: 
        - panel: (n x T) array
        - block_size: int
        - n_factors: int.
    """
    
    n, _ = panel.shape
    mask = set()
    basis = np.eye(n)
    cov = np.cov(panel)
    
    for i in range(n+1-block_size): 
        block = max_weight_clique(cov,block_size,set())
        block_cov = build_block_cov(cov,block)
        
        eig = ordered_eigen_decomposition(block_cov)
        J = build_indexed_Jacobi(block, eig, n)
        basis, cov = update_basis_rotate_cov(J, basis, cov)
        
    energies = variance_score(panel, basis)
    energy_rankings = ss.rankdata(energies)
    best_basis = [i for i in range(n) if energy_rankings[i] > (n-n_factors)]
    
    P = basis[:,best_basis]
    w = np.matmul(P,P.transpose())
    est_loadings = np.sqrt(n)*P 
    est_factors = np.matmul(P.transpose(),panel)
    est_common_component = np.matmul(w,panel)
    
    return est_loadings, est_factors, est_common_component


# 
def from_treelets(panel, n_factors): 
    """
    Estimates loadings, factors, and common components by imspecting maximum energy 
    basis in treelet decomposition. 
    Args: 
        - panel: (n x T) array
        - block_size: int
        - n_factors: int
    """
    
    n, _ = panel.shape
    tree = tl.treelet_decomposition(panel.T,n,True)
    best_basis = tl.best_basis(tree, n_factors, panel.T) 
    
    P = best_basis[0]["basis"]
    w = np.matmul(P,P.transpose())
    est_loadings = np.sqrt(n)*P 
    est_factors = np.matmul(P.transpose(),panel)
    est_common_component = np.matmul(w,panel)
    
    return est_loadings, est_factors, est_common_component