\name{Build_JTree}
\alias{Build_JTree}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
%%  ~~function to do ... ~~
Constructs the hierarchical tree for the Treelet algorithm
}
\description{
%%  ~~ A concise (1-5 lines) description of what the function does. ~~
Returns information on the construction of the treelet hierarchical tree on which the basis functions are supported.  
}
\usage{
Build_JTree(C, cc, maxlev, whichsave)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{C}{
%%     ~~Describe \code{C} here~~
the covariance matrix of the data.  For example, if using this function on genetics data to improve estimates of heritability, as in the Crossett et al arXiv paper, this argument will be the estimated additive genetic relationship matrix \eqn{\hat{A}}.
}
  \item{cc}{
%%     ~~Describe \code{cc} here~~
the correlation matrix of the data.
}
  \item{maxlev}{
%%     ~~Describe \code{maxlevel} here~~
the maximum height of the tree.  This must be an integer between 1 and \code{nrow(X)}-1.
}
  \item{whichsave}{
%%     ~~Describe \code{whichsave} here~~
a vector containing the levels of the tree, specified as integers between 1 and \code{maxlev}, for which you want to save the basis functions and the covariance matrix. 
}
}
%\details{
%%%  ~~ If necessary, more details than the description above ~~
%}
\value{
%%  ~Describe the value returned
%%  If it is a LIST, use
%%  \item{comp1 }{Description of 'comp1'}
%%  \item{comp2 }{Description of 'comp2'}
%% ...
a list with components
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
\code{\link{Run_JTree}}, \code{\link{JTree_Basis}}, \code{\link{TCS}}
}
%\examples{
%##---- Should be DIRECTLY executable !! ----
%##-- ==>  Define data, use random,
%##--	or do  help(data=index)  for the standard data sets.
%
%}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
%\keyword{ ~kwd1 }
%\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
