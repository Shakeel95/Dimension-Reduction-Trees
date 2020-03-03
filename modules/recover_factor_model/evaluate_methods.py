import numpy as np
import matplotlib.pyplot as plt
import recover_factor_model.blocking_schemes as fit
from recover_factor_model.helper_functions import * 

#
class evaluate_blocking_schemes:
    """
    Wraper class to help compare different blocking schemes. 
    """
    
    def __init__(self,loadings,factors,common_component,ideosyncratic_component,n_factors,block_size):
        
        self.loadings = loadings
        self.factors = factors 
        self.common_component = common_component 
        self.ideosyncratic_component = ideosyncratic_component
        self.panel = self.common_component + self.ideosyncratic_component
        
        self.standard_PCA = self.standard_PCA(self.panel,n_factors)
        self.naive_disjoint_blocks = self.naive_disjoint_blocks(self.panel,block_size, n_factors)
        self.naive_sliding_blocks = self.naive_sliding_blocks(self.panel,block_size, n_factors)
        self.ranked_disjoint_blocks = self.ranked_disjoint_blocks(self.panel,block_size, n_factors)
        self.ranked_sliding_blocks = self.ranked_sliding_blocks(self.panel,block_size, n_factors)
        self.from_treelets = self.from_treelets(self.panel,n_factors)
        
        self.blocking_schemes = ["standard PCA","naive disjoint blocks", "naive sliding blocks",
                                 "ranked disjoint blocks", "ranked sliding blocks", "treelets"]        
        
    class standard_PCA:
        """
        Estimates loadings, factors, and common component in a static factor model using PCA.
        """
        def __init__(self,panel,n_factors):
            (self.est_loadings, 
             self.est_factors, 
             self.est_common_component) = fit.standard_PCA(panel,n_factors)
            self.panel = panel
        def change_params_refit(self,n_factors):
            (self.est_loadings, 
             self.est_factors, 
             self.est_common_component) = fit.standard_PCA(self.panel,n_factors)
            
    class naive_disjoint_blocks:
        """
        Estimates laodings, factors, and common components via blockwise PCA using disjoint blocks. 
        """
        def __init__(self,panel,block_size, n_factors):
            (self.est_loadings, 
             self.est_factors, 
             self.est_common_component) = fit.naive_disjoint_blocks(panel,block_size,n_factors)
            self.panel = panel
        def change_params_refit(self,block_size, n_factors):
            (self.est_loadings, 
             self.est_factors, 
             self.est_common_component) = fit.naive_disjoint_blocks(self.panel,block_size, n_factors)
                
    class naive_sliding_blocks:
        """
        Estimates loadings, factors, and common components via blockwise PCA with sliding blocks.
        """
        def __init__(self,panel,block_size, n_factors):
            (self.est_loadings, 
             self.est_factors, 
             self.est_common_component) = fit.naive_sliding_blocks(panel,block_size, n_factors)
            self.panel = panel
        def change_params_refit(self,block_size, n_factors):
            (self.est_loadings, 
             self.est_factors, 
             self.est_common_component) = fit.naive_sliding_blocks(self.panel,block_size, n_factors)
        
    class ranked_disjoint_blocks:
        """
        Estimates loadings, factors, and common component via blockwise PCA. 
        Blocks are taken to be variables which are jointly maximally correlated.
        """
        def __init__(self,panel,block_size, n_factors):
            (self.est_loadings, 
             self.est_factors, 
             self.est_common_component) = fit.ranked_disjoint_blocks(panel,block_size,n_factors)
            self.panel = panel
        def change_params_refit(self,block_size, n_factors):
            (self.est_loadings, 
             self.est_factors, 
             self.est_common_component) = fit.ranked_disjoint_blocks(self.panel,block_size, n_factors)
            
    class ranked_sliding_blocks:
        """
        Estimates loadings, factors, and common components via blockwise. 
        In each pass the blocks are chosen to be maximally correlated.
        """
        def __init__(self,panel,block_size, n_factors):
            (self.est_loadings, 
             self.est_factors, 
             self.est_common_component) = fit.ranked_sliding_blocks(panel,block_size,n_factors)
            self.panel = panel
        def change_params_refit(self,block_size, n_factors):
            (self.est_loadings, 
             self.est_factors, 
             self.est_common_component) = fit.ranked_sliding_blocks(self.panel,block_size, n_factors)
        
    class from_treelets:
        """
        Estimates loadings, factors, and common components by imspecting maximum energy 
        basis in treelet decomposition. 
        """
        def __init__(self,panel,n_factors):
            (self.est_loadings, 
             self.est_factors, 
             self.est_common_component) = fit.from_treelets(panel,n_factors)
            self.panel = panel
        def change_params_refit(self,n_factors):
            (self.est_loadings, 
             self.est_factors, 
             self.est_common_component) = fit.from_treelets(self.panel,n_factors)    
            
    def change_params_refit(self,block_size, n_factors):
        """"
        Change block size and number of factors; refit all blocking schemes. 
        """
        
        self.standard_PCA.change_params_refit(n_factors)
        self.naive_disjoint_blocks.change_params_refit(block_size, n_factors)
        self.naive_sliding_blocks.change_params_refit(block_size, n_factors)
        self.ranked_disjoint_blocks.change_params_refit(block_size, n_factors)
        self.ranked_sliding_blocks.change_params_refit(block_size, n_factors)
        self.from_treelets.change_params_refit(n_factors)
        
        return True

    def compare_loadings_to_PCA(self,norm = 2):
        
        estimates = (self.naive_disjoint_blocks.est_loadings,
                     self.naive_sliding_blocks.est_loadings,
                     self.ranked_disjoint_blocks.est_loadings,
                     self.ranked_sliding_blocks.est_loadings,
                     self.from_treelets.est_loadings)
        
        error = [matrix_error(estimate,self.standard_PCA.est_loadings)
                 for estimate in estimates]
        
        return error
        
        
    def compare_loadings_to_population(self,norm = 2): 
        
        estimates = (self.standard_PCA.est_loadings,
             self.naive_disjoint_blocks.est_loadings,
             self.naive_sliding_blocks.est_loadings,
             self.ranked_disjoint_blocks.est_loadings,
             self.ranked_sliding_blocks.est_loadings,
             self.from_treelets.est_loadings)
        
        error = [matrix_error(estimate,self.loadings)
                 for estimate in estimates]
        
        return error
    
    def compare_factors_to_PCA(self,norm = 2):
        
        estimates = (self.naive_disjoint_blocks.est_factors,
                     self.naive_sliding_blocks.est_factors,
                     self.ranked_disjoint_blocks.est_factors,
                     self.ranked_sliding_blocks.est_factors,
                     self.from_treelets.est_factors)
        
        error = [np.linalg.norm(np.abs(estimate)-np.abs(self.standard_PCA.est_factors),ord=norm)
                 for estimate in estimates]
        
        return error
        
        
    def compare_factors_to_population(self,norm = 2): 
        
        estimates = (self.standard_PCA.est_factors,
             self.naive_disjoint_blocks.est_factors,
             self.naive_sliding_blocks.est_factors,
             self.ranked_disjoint_blocks.est_factors,
             self.ranked_sliding_blocks.est_factors,
             self.from_treelets.est_factors)
        
        error = [np.linalg.norm(np.abs(estimate)-np.abs(self.factors),ord=norm)
                 for estimate in estimates]