\name{native-hierarchical-splines}
\alias{native-hierarchical-splines}
\alias{BSplineBasis_open_uniform}
\alias{BSplineBasisVector}
\alias{HierarchicalBSpline}
\alias{TruncatedHierarchicalBSpline}
\alias{TensorBSpline}
\title{Native hierarchical B-spline patches and penalized fits}
\description{
The generated bindings expose the canonical C++ spline bases, tensor splines,
and hierarchical solvers. A \code{BSplineBasisVector} holds one basis per axis;
sample points are a list of numeric coordinate vectors. These objects own
their native storage and release it when collected.
}
\details{
\code{HierarchicalBSpline(lower, upper)} creates an empty additive patch
hierarchy with root cell 0. Each bound is a numeric vector with one entry per
axis. \code{BSplineBasis_open_uniform(degree, basis_size)} creates a basis on
[0, 1]; use this local coordinate domain for patch bases.

\code{model$fit_cell(cell, bases, points, values)} fits an unpenalized patch.
\code{model$fit_cell_penalized(cell, bases, points, values, lambdas, order)}
adds a coefficient difference penalty with one nonnegative lambda per axis
and difference order 2 by default. Pass raw observations, not residuals:
the solver subtracts existing ancestor contributions. The target cell must
not already contain a patch. Points must lie inside that cell. Invalid
arguments and singular fits raise R errors.

\code{model$refine(cell, slot)} returns a child cell ID. In one dimension,
slot 0 is the lower half and slot 1 is the upper half.
\code{model$evaluate(point)} sums patches along the path containing the point.
Independent patches do not impose continuity at cell boundaries.

Hierarchical penalized (HP) fits use \code{fit_cell_penalized}; there is no
separate HP class. This differs from hp refinement with variable polynomial
degree. \code{TruncatedHierarchicalBSpline} instead uses a nested, truncated
basis and a joint fit. The playground contains a complete C++/Python/R
comparison with global \code{PSplineRegression}.
}
\examples{
bases <- BSplineBasisVector()
invisible(bases$push_back(BSplineBasis_open_uniform(3L, 6L)))
model <- HierarchicalBSpline(c(0), c(1))
invisible(model$fit_cell_penalized(
  0L, bases, list(c(0), c(1)), c(0, 2), c(100), 2L))
round(model$evaluate(c(0.5)), 6) # 1
}
