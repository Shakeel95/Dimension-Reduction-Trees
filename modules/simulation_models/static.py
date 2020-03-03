import numpy as np
from statsmodels.tsa import arima_process as arima

#
# def stock_and_watson(loadings, sigma, factor_order): 
#     """
#     Simulates a (N x T) factor model according to Stock and Watson (2002).
#     Returns common component, ideosyncratic component, and panel as arrays.
#     Args: 
#         - loadings: (n x q) array of loadings
#         - 
#         - sigma: std of common component
#         - factor_order: ARIMA order for factors
#     """
    
#     ar = [1]+factor_order[0]
#     ma = [1]+factor_order[1]
#     L = np.linalg.cholesky()
#     shocks = np.array([
#             arima.arma_generate_sample(ar,ma,T,burnin = burn) for i in range(q)
#             ])
    
#     factors = np.matmul(L,shocks)
#     common_component = np.matmul(loadings, factors) 
    
#     return panel, common_component, ideosyncratic_component


#
def no_serial_correlation(loadings, var_cov, sigma, T):
    """
    Simulates observations from a factor model with no serial correlation.
    Args:
        - loadings: (n x q) orthogonal array
        - var_cov: (q x q) p.s.d. array
        - sigma: variance of ideosyncratic component
        - T: sample size 
    """
    
    n,q = loadings.shape
    L = np.linalg.cholesky(var_cov)
    
    shocks = np.random.normal(0,1,(q,T))
    factors = np.matmul(L,shocks)
    ideosyncratic_component = np.random.normal(0,sigma,(n,T))
    
    common_component = np.matmul(loadings,factors)
    panel = common_component + ideosyncratic_component
    
    return panel, factors, common_component, ideosyncratic_component