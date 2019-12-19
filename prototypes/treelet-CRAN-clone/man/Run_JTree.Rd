\name{Run_JTree}
\alias{Run_JTree}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
%%  ~~function to do ... ~~
Treelet basis/hierarchical tree construction
}
\description{
%%  ~~ A concise (1-5 lines) description of what the function does. ~~
Returns information on the simultaneous construction of the treelet orthonormal basis and hierarchical tree, including which nodes were merged at each step and the basis at each specified step of the construction.
}
\usage{
Run_JTree(X, maxlev, whichsave)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{X}{
%%     ~~Describe \code{X} here~~
the covariance matrix of the data.  For example, if using this function on genetics data to improve estimates of heritability, as in the Crossett et al arXiv paper, this argument will be the estimated additive genetic relationship matrix \eqn{\hat{A}}.
}
  \item{maxlev}{
%%     ~~Describe \code{maxlev} here~~
the maximum height of the tree.  This must be an integer between 1 and \code{nrow(X)}-1.
}
  \item{whichsave}{
%%     ~~Describe \code{whichsave} here~~
a vector containing the levels of the tree, specified as integers between 1 and \code{maxlev}, for which you want to save the basis functions and the covariance matrix. 
}
}
\details{
%%  ~~ If necessary, more details than the description above ~~
This function serves as a wrapper for the functions \code{\link{Build_JTree}} and \code{\link{JTree_Basis}}, which build the hierarchical tree and calculate the basis and covariance matrix at each level, respectively. 
}
\value{
%%  ~Describe the value returned
%%  If it is a LIST, use
%%  \item{comp1 }{Description of 'comp1'}
%%  \item{comp2 }{Description of 'comp2'}
%% ...
a list with components
\item{basis}{This is a list with \code{maxlev} elements.  Only those elements that are specified in the \code{whichsave} argument will be non-null entries in the list.  For the non-null entries, this is the orthonormal treelet basis calculated at that level of the tree.}
\item{Zpos}{A matrix of dimension \code{maxlev} x 2.  Each row records which two nodes/clusters of the tree were combined at each step in its construction.}
\item{T}{This is a list with \code{maxlev} elements, where each element is a 2x2 Jacobi rotation matrix for each step of the treelet algorithm.}
\item{PCidx}{A matrix of dimension \code{maxlev} x 2, where each row is a permutation of \eqn{(1,2)} indicating which of the two nodes/clusters merged at that step is the sum variable (value of 1) and which is the difference (value of 2).}
\item{all_nodes}{A matrix of dimension \code{maxlev} x \code{nrow(X)} giving node/cluster labels at each step of the treelet algorithm.  A label of zero indicates a node/cluster that was merged with another node/cluster and was the difference variable.}
\item{TreeCovs}{This is a list with \code{maxlev} elements.  Only those elements that are specified in the \code{whichsave} argument will be non-null entries in the list.  For the non-null entries, this is the covariance matrix calculated at that level of the tree.  The covariances in this matrix are those between the weights (orthogonal projections onto local basis vectors) in the basis expansion of the data vector.}
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
\code{\link{Build_JTree}}, \code{\link{JTree_Basis}}, \code{\link{TCS}}
}
\examples{
data(Ahat)
out=Run_JTree(Ahat,49,49)
#The information in out$Zpos[1,] and out$all_nodes[1,]
#both show which two individuals were the first merged 
#in the tree.  The remaining rows give information
#on subsequent merges in the tree.

basis=out$basis[[49]]
cov=out$TreeCovs[[49]]
temp=basis%*%cov%*%t(basis)
#This is how you can use the basis and cov output
#to reconstruct the estimated relationship matrix.
#See how close temp and the original Ahat are:
Ahat1=round(Ahat,14)
temp1=round(temp,14)
sum(Ahat1!=temp1)
#In this example, we do start seeing discrepancies in the 15th digit and beyond.

}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
%\keyword{ ~kwd1 }
%\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
