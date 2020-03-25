library(docstring) # document nicely 
library(mvtnorm) # sample from  multivariate normal
library(Matrix) # fast matrix computations

tv_VAR <- function(Phi, Sigma, t, burnin){
  #' Simulate time varying VAR
  #' 
  #' Simulate draws from vector auto-regression with time varying parameters
  #'
  #'@param Phi function of t, returns a list of (n x n) VAR loading matrices
  #'@param Sigma function of t, returns (n x n) covariance matrix for innovations 
  #'@param t number of time points to simulate
  #'@param burnin number of initial points to discard to achieve statiuonarity
  
  # set params
  n <- ncol(Sigma(1))
  tot_t <- t + burnin
  VAR <- matrix(,tot_t,n)
  
  # set starting values 
  max_lag <- length(Phi(1))
  VAR[1:max_lag,] <- matrix(0,max_lag,n)
  
  # loop, simulate VAR
  for (t in (max_lag+1):(tot_t)){
    
    var_coef <- Phi(t)
    lags <- 1:length(var_coef)
    innovation <- rmvnorm(1,rep(0,n), Sigma(t))
    filtered <- rowSums(do.call(cbind,lapply(seq_along(lags),function(i)var_coef[[i]] %*% (VAR[t-lags[i],]))))
    VAR[t,] <- filtered + innovation
  }
  
  return(VAR[-(1:burnin),])
}


tv_MA <- function(Theta, Sigam, tt){
  #' Simulate time varying MA 
  #' 
  #' Simulate draws from vector moving-average with time varying parameters
  #' 
  #'@param Theta function of t, returns a list of (n x n) VMA loading matrices
  #'@param Sigma function of t, returns (n x n) covariance matrix for innovations 
  #'@param tt number of time points to simulate

  # set params
  n <- ncol(Sigam(1))
  VMA <- matrix(,tt,n)
  
  # generate innovations
  q1 <- length(Theta(1))
  innovations <- matrix(,tt + q1,n)
  for (t in 1:(tt+q1)){
    innovations[t,] <- rmvnorm(1,rep(0,n),Sigam(t))
  }

  # loop, simulate VMA
  for (t in 1:tt){
    vma_coef <- Theta(t)
    lags <- 1:length(vma_coef)
    filtered <- rowSums(do.call(cbind,lapply(seq_along(lags),function(i)vma_coef[[i]] %*% (innovations[t+q1-lags[i],]))))
    innov <- innovations[t+q1,]
    VMA[t,] <- filtered + innov
  }
  return(VMA)
}

