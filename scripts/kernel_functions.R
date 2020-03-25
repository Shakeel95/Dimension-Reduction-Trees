kern <- function(x1,x2,measure){
  #'Needs a name! 
  #'
  #'
  #' @param x1
  #' @param x2
  #' @param similarity

  if(length(x1) != length(x2)) stop("x1 and x2 should have same dimension")

  N = length(x1)
  l = ceiling(N**(1/2))
  b = floor(N/l)

  theta <- c()
  for (j in 1:b){
    est.theta <- measure(x1[(1+(j-1)*l):(j*l)],x2[(1+(j-1)*l):(j*l)])
    theta <- c(theta,est.theta)
  }
  
  omega <- c()
  for (j in 1:b){
    if (j == 1){
      est.omega <- (theta[j] - theta[j+1] + 1)**(-2)
      omega <- c(omega, est.omega)
    } else if (j == b) {
      est.omega <- (theta[j] - theta[j-1] + 1)**(-2)
      omega <- c(omega, est.omega)
    } else {
      est.omega <- min((theta[j] - theta[j+1] + 1)**(-2),(theta[j] - theta[j-1] + 1)**(-2))
      omega <- c(omega, est.omega)
    }
  }
  
  omega <- omega/sum(omega)
  
  return(sum(omega*theta))
}

abs.cov <- function(x,y){return(abs(cov(x,y)))}
abs.cor <- function(x,y){return(abs(cov(x,y)))}