## Simulate large blockwise VAR

library(docstring)

simulate_blockwise_var <- function(n,t,Q,p,seed){
  #' Simulate Blockwise VAR
  #' 
  #' @param n number of time series in panel
  #' @param t number of time points to simulate
  #' @param Q block dimension 
  #' @param p maximum lag oprder
  #' @param seed seed for rnadom number generation
  
  
  res <- list()
  mat_polynomials <- list()
  
  for (i in 1:)
  
}

Q <- 5
p <- 2 
mat_polynomials <- list()

for (i in 1:p){
  mat_polynomials[[i]] <- matrix(rnorm((Q+1)*(Q+1)),(Q+1))
}