# Double-precision native matrix handles. S3 methods preserve ordinary R syntax
# while all products, transposes and solves execute in the canonical C++ core.
.dm_matrix <- function(handle) {
  structure(list(handle = handle), class = c(if (handle$is_sparse()) "SpMat", "Mat"))
}

#' Construct an owned native dense matrix from an R numeric matrix.
#' Expressions own their operands. Printing or as.matrix() evaluates; eval_mat()
#' keeps a snapshot. Binding scalars are double precision.
#' @param x A numeric matrix, with R's usual column-major layout.
#' @return A Mat handle with native row-major storage.
#' @export
Mat <- function(x) {
  if (!is.matrix(x) || !is.numeric(x)) stop("x must be a numeric matrix")
  .dm_matrix(LinalgMatrix(as.numeric(t(x)), nrow(x), ncol(x), FALSE))
}

#' Construct an owned native CSR matrix, omitting zero coefficients.
#' @param x A numeric matrix.
#' @return A SpMat handle.
#' @export
SpMat <- function(x) {
  if (!is.matrix(x) || !is.numeric(x)) stop("x must be a numeric matrix")
  .dm_matrix(LinalgMatrix(as.numeric(t(x)), nrow(x), ncol(x), TRUE))
}

#' @export
t.Mat <- function(x) .dm_matrix(x$handle$transpose())

#' @export
`%*%.Mat` <- function(x, y) {
  if (!inherits(x, "Mat") || !inherits(y, "Mat")) stop("operands must be datamunge matrices")
  .dm_matrix(x$handle$multiply(y$handle))
}

#' Defer an inverse action, or solve a square/tall system using native LU/QR.
#' solve(A) %*% B %*% C solves A Z = B C without forming an inverse or A^-1 B.
#' @param a A native Mat or SpMat coefficient matrix.
#' @param b Optional native right-hand-side matrix; omitted for a deferred action.
#' @param ... Reserved for generic compatibility; additional arguments are rejected.
#' @return A deferred inverse action or an owned dense solution.
#' @export
solve.Mat <- function(a, b, ...) {
  if (length(list(...))) stop("unused solve arguments")
  if (missing(b)) return(.dm_matrix(a$handle$inverse_action()))
  if (!inherits(b, "Mat")) stop("b must be a datamunge matrix")
  .dm_matrix(a$handle$solve(b$handle))
}

#' Evaluate a native matrix expression into an owned snapshot.
#' @param x A native Mat or SpMat expression.
#' @return A materialised native Mat or SpMat.
#' @export
eval_mat <- function(x) .dm_matrix(x$handle$eval())

#' @export
as.matrix.Mat <- function(x, ...) {
  matrix(x$handle$values(), nrow = x$handle$rows(), ncol = x$handle$cols(), byrow = TRUE)
}

#' @export
dim.Mat <- function(x) as.integer(c(x$handle$rows(), x$handle$cols()))

#' @export
print.Mat <- function(x, ...) {
  cat(x$handle$text(), "\n", sep = "")
  invisible(x)
}
