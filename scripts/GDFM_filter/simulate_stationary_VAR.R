library(docstring) # document nicely 
library(mvtnorm) # sample from  multivariate normal
library(Matrix) # fast matrix computations
library(pracma) #sample from orthonormal matrices

random_SVD_loadings <- function(n,p){
  #'Random SVD loadings 
  #'
  #'Randomly sample from the space of loading matrices which would generate a 
  #'stationary VAR. 
  #'
  #'@param n dimension of VAR
  #'@param p number of lags
  
  # SVD mats
  U <- randortho(n, type = "unitary")
  V <- randortho(n*p, type = "unitary")
  S <- matrix(0,n,n*p)
  S[1:n,1:n] <- diag(runif(n))
  
  # reconstruct loadings & extract
  phi <- U%*%S%*%V
  var_coef <- list()
  for (i in 0:(p-1)){
    var_coef[[i+1]] <- Re(phi[,(n*i+1):(n*i+n)])
  }
  return(var_coef)
}

simulate_VAR <- function(var_coef, R, t, burnin = 1000){
  #' Simulate Stationary VAR
  #' 
  #' Simulate draws from vector auto-regression given 
  #'
  #'@param var_coef a list of (Q+1)*(Q+1) VAR matrices for each lag
  #'@param R (Q+1)*Q weighting matrix for innovations 
  #'@param t number of time points to simulate
  #'@param burnin number of initial points to discard to achieve statiuonarity
  
  # tests
  if (rankMatrix(R) != ncol(var_coef[[1]])-1) stop("R should have rank Q")
  if (any(sapply(var_coef,function(x) length(unique(dim(x))) > 1 ))) stop ("Non-square graph detected")
  
  # set params
  Q <- ncol(var_coef[[1]])-1
  tot_t <- t + burnin
  max_lag <- length(var_coef)
  lags <- 1:max_lag
  
  # set starting vals
  starting_vals <- matrix(0,max_lag,Q+1)
  VAR <- matrix(,tot_t,Q+1)
  VAR[1:max_lag,] <- starting_vals
  
  # loop, simulate VAR
  for (t in (max_lag+1):(tot_t)){
    innovation <- R %*% c(rmvnorm(1,rep(0,Q), diag(1,Q)))
    filtered <- rowSums(do.call(cbind,lapply(seq_along(lags),function(i)var_coef[[i]] %*% (VAR[t-lags[i],]))))
    VAR[t,] <- filtered + innovation
  }
  
  return(VAR[-(1:burnin),])
}



simulate_GDFM <- function(Q,K,lags,t){
  #'Simulates Generalised Dynamic Factor MOdel 
  #'
  #'Simulates GDFM via block-wise VARs
  #'
  #'@param Q number of shocks driving common component
  #'@param K number of VAR blocks - determines n via n = K x (Q+1)
  #'@param lags number of lags in VAR representation of GDFM
  #'@param t number of time points to simulate
  
  GDFM <- list()
  chi = xi = matrix(0,t,K*(Q+1))
  GDFM["AR_coef"] <- list()
  GDFM["innov_mat"] <- list()
  
  for (k in 1:K){
    GDFM$AR_coef[[k]] <- random_SVD_loadings(Q+1,lags)
    GDFM$innov_mat[[k]] <- matrix(rnorm(Q*(Q+1)),(Q+1)) 
  }
  
  # return(GDFM)
  
  for (k in 0:(K-1)){
    chi[,((Q+1)*k+1):((Q+1)*k+(Q+1))] <- simulate_VAR(GDFM$AR_coef[[k+1]], GDFM$innov_mat[[k+1]],t)
    xi[,((Q+1)*k+1):((Q+1)*k+(Q+1))] <- matrix(rnorm(t*(Q+1)),t,(Q+1))
  }
  
  print("done")
  GDFM["chi"] <- list(chi)
  GDFM["xi"] <- list(xi)
  GDFM["panel"] <- list(chi + xi)
  
  return(GDFM)
}