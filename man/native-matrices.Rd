\name{native-matrices}
\alias{Mat}
\alias{SpMat}
\alias{eval_mat}
\alias{t.Mat}
\alias{\%*\%.Mat}
\alias{solve.Mat}
\alias{as.matrix.Mat}
\alias{dim.Mat}
\alias{print.Mat}
\title{Native dense and sparse matrix expressions}
\description{
Double-precision matrices evaluated by the datamunge C++ core. Dense matrices
use row-major storage; sparse matrices use compressed sparse rows (CSR).
}
\usage{
Mat(x)
SpMat(x)
eval_mat(x)
\method{t}{Mat}(x)
\method{solve}{Mat}(a, b, ...)
\method{as.matrix}{Mat}(x, ...)
\method{dim}{Mat}(x)
\method{print}{Mat}(x, ...)
}
\arguments{
\item{x}{An R numeric matrix for constructors, or a native matrix expression for methods.}
\item{a}{A native coefficient matrix.}
\item{b}{An optional native right-hand-side matrix.}
\item{...}{Reserved for generic compatibility. The solve method rejects extra arguments.}
}
\details{
Constructors copy R matrices into owned native storage. Zero entries are omitted
from CSR storage. Native transposes and matrix products are deferred; expressions
keep their operands alive across garbage collection. Printing or converting to
an R matrix evaluates the expression. Use \code{eval_mat()} to retain a native snapshot.

\code{solve(A)} creates a deferred square inverse action. The expression
\code{solve(A) \%*\% B \%*\% C} computes \code{B \%*\% C} before solving,
without an explicit inverse or the solved intermediate. Sparse products remain
CSR; coefficient matrices and right-hand sides become dense at the solve boundary.
\code{solve(A, b)} immediately solves by LU for square matrices or Householder QR
for tall, full-column-rank matrices. Invalid dimensions and nonfinite solve
inputs raise errors; singular and rank-deficient systems fail on evaluation.
Normal equations square the condition number; prefer the direct tall QR solve
when numerical conditioning matters.
}
\value{
Constructors, transposes, products, solves and \code{eval_mat()} return native
\code{Mat} or \code{SpMat} handles. \code{as.matrix()} returns an R numeric matrix;
\code{dim()} returns an integer pair and \code{print()} invisibly returns its argument.
}
\examples{
X <- Mat(matrix(c(1, 0, 1, 1, 1, 2), ncol = 2, byrow = TRUE))
y <- Mat(matrix(c(1, 3, 5), ncol = 1))
beta <- solve(t(X) \%*\% X) \%*\% t(X) \%*\% y
as.matrix(beta) # coefficients 1 and 2
as.matrix(solve(X, y))
SpMat(diag(c(2, 4)))
}
