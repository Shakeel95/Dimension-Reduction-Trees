Phi <- function(t){
  #' Phi
  #' 
  #'Time varying AR loadings for tv-VARMA model
  #' 
  #' @param t time
  
  mat <- matrix(0,4,4)
  bloc.1 <- toeplitz(0.5^(1:2))
  bloc.2 <- toeplitz((-0.5)^(1:2))
  
  if (t < 1100){
    mat[1:2,1:2] <- bloc.1
    mat[3:4,3:4] <- bloc.2
    return(list(mat))
  } else {
    mat[1:2,1:2] <- bloc.2
    mat[3:4,3:4] <- bloc.1
    return(list(mat))
  }
}

Theta <- function(t){
  #' Theta
  #' 
  #' Time varying MA loadings for tv-VARMA model
  #' 
  #' @param t time
  
  mat <- matrix(0,4,4)
  bloc.1 <- toeplitz(0.5^(1:2))
  bloc.2 <- toeplitz((-0.5)^(1:2))
  
  if (t < 1000){
    mat[1:2,1:2] <- bloc.1
    mat[3:4,3:4] <- bloc.2
    return(list(mat))
  } else {
    mat[1:2,1:2] <- bloc.2
    mat[3:4,3:4] <- bloc.1
    return(list(mat))
  }
}

Sigma <- function(t){
  #'Sigma
  #'
  #'Time varying variance matrix for innovations in VAR model
  #'
  #'@param t 
  
  return(0.1*diag(4))
}