\name{TCS}
\alias{TCS}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
%%  ~~function to do ... ~~
Treelet Covariance Smoothing
}
\description{
%%  ~~ A concise (1-5 lines) description of what the function does. ~~
This function thresholds values in the treelet estimated covariance and returns a smoothed estimate of a covariance matrix.
}
\usage{
TCS(basis, cov, lambda)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{basis}{
%%     ~~Describe \code{basis} here~~
the orthonormal treelet basis calculated at a specific level \eqn{\ell} of the tree.
}
  \item{cov}{
%%     ~~Describe \code{cov} here~~
the corresponding covariance matrix calculated at level \eqn{\ell} of the tree.  The covariances in this matrix are those between the weights (orthogonal projections onto local basis vectors) in the basis expansion of the data vector.
}
  \item{lambda}{
%%     ~~Describe \code{lambda} here~~
a positive thresholding coefficient.  Any element of the matrix \code{cov} that is less than this coefficient in absolute value will be set to zero.
}
}
\details{
%%  ~~ If necessary, more details than the description above ~~
This function implements the TCS method presented in the Crossett et al arXiv paper.  The arguments \code{basis} and \code{cov} should be obtained from the \code{\link{Run_JTree}} function.  The \code{TCS} function is written so that it does not calculate the treelet basis within the function but asks for it as an argument so that the subsampling method presented in the arXiv paper, or another method to obtain a reasonable value of lambda, can be implemented.
}
\value{
%%  ~Describe the value returned
%%  If it is a LIST, use
%%  \item{comp1 }{Description of 'comp1'}
%%  \item{comp2 }{Description of 'comp2'}
%% ...
\item{smooth}{the smoothed estimate of the covariance matrix.}
}
\references{
%% ~put references to the literature/web site here ~
\href{http://arxiv.org/abs/1208.2253}{arXiv:1208.2253v1 [stat.AP]}

Lee, AB, Nadler, B, Wasserman, L (2008).  Treelets - an adaptive multi-scale basis for sparse unordered data. The Annals of Applied Statistics 2: 435-471. \url{http://www.stat.cmu.edu/~annlee/AOAS137.pdf}
}
\author{
Trent Gaugler \email{gauglert@lafayette.edu}
}
%\note{
%%%  ~~further notes~~
%}

%% ~Make other sections like Warning with \section{Warning }{....} ~

\seealso{
%% ~~objects to See Also as \code{\link{help}}, ~~~
\code{\link{Build_JTree}}, \code{\link{JTree_Basis}}, \code{\link{Run_JTree}}
}
\examples{
data(Ahat)
out=Run_JTree(Ahat,49,49)
basis=out$basis[[49]]
cov=out$TreeCovs[[49]]

temp=TCS(basis,cov,.04)
#The value .04 above is arbitrary, and the user 
#should carefully select this value.  One approach
#is the subsampling method outlined in the Crossett et al
#arXiv paper.  The value in 'temp' is the smoothed estimate
#of the relationship matrix.

}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
%\keyword{ ~kwd1 }
%\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
