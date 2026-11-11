# Short names matching the C++ API (dm::BSR, dm::PSR, dm::RGBA): aliases of the R6 class generators,
# so BSR$new(...) builds a BSplineRegression.
BSR <- BSplineRegression
PSR <- PSplineRegression
RGBA <- RGB

# The SWIG bindings' STL container wrappers, for older code: containers now cross as plain
# R values (vectors as atomic vectors or lists, pairs as list(first =, second =)).
DVector <- function(x = numeric()) as.numeric(x)
IVector <- function(x = integer()) as.integer(x)
SVector <- function(x = character()) as.character(x)
DPair <- function(first, second) list(first = as.numeric(first), second = as.numeric(second))
IPair <- function(first, second) list(first = as.integer(first), second = as.integer(second))
