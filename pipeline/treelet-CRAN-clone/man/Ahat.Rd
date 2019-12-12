\name{Ahat}
\alias{Ahat}
\docType{data}
\title{
Estimated additive genetic relationship matrix for 50 simulated individuals
}
\description{
Using the simulated data from the family pedigrees in the Crossett et al arXiv paper, five families were randomly selected, and 10 individuals from the 15 related individuals were randomly selected independently in each family.  The \eqn{\hat{A}} matrix was then calculated via the GCTA software of Yang et al.  In the resulting 50 x 50 relationship matrix, element \eqn{\hat{A}_{ij}} represents the estimated kinship coefficient between individuals \eqn{i} and \eqn{j}.
}
\usage{data(Ahat)}
%\format{
%  The format is:
% num [1:50, 1:50] 0.9841 -0.0202 -0.0311 0.0332 0.0477 ...
%}
%\details{
%%  ~~ If necessary, more details than the __description__ above ~~
%}
\source{
\href{http://arxiv.org/abs/1208.2253}{arXiv:1208.2253v1 [stat.AP]}
}
\references{
\href{http://arxiv.org/abs/1208.2253}{arXiv:1208.2253v1 [stat.AP]}

Lee, AB, Nadler, B, Wasserman, L (2008).  Treelets - an adaptive multi-scale basis for sparse unordered data. The Annals of Applied Statistics 2: 435-471. \url{http://www.stat.cmu.edu/~annlee/AOAS137.pdf}

Yang J, Lee SH, Goddard ME and Visscher PM. GCTA: a tool for Genome-wide Complex Trait Analysis. Am J Hum Genet. 2011 Jan 88(1): 76-82. \href{http://www.ncbi.nlm.nih.gov/pubmed/21167468}{[PubMed ID: 21167468]}
}
%\examples{
%data(Ahat)
%## maybe str(Ahat) ; plot(Ahat) ...
%}
%\keyword{datasets}
