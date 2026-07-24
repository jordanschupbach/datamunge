#pragma once

// Convenience umbrella header for datamunge::algebra -- computational (symbolic) algebra:
// univariate polynomial arithmetic, GCD algorithms (Euclidean, extended, pseudo-remainder
// sequence), Lagrange/Newton interpolation, resultants/discriminants, real-root isolation
// (Sturm's theorem, Descartes' rule of signs), square-free factorization (Yun's algorithm),
// factorization over GF(p) (Berlekamp's algorithm), all-complex-roots polynomial root-finding
// (Durand-Kerner) and exact rational-root search (the Rational Root Theorem), a numeric
// real/complex factorization built on both, modular integer arithmetic (incl. the Chinese
// Remainder Theorem), rational functions, multivariate polynomials with monomial orderings and
// a Groebner-basis solver (Buchberger's algorithm), a symbolic expression tree with
// differentiation, integration, simplification, and evaluation, and two calculus utilities
// built on that tree (adaptive-Simpson numeric definite integration and Taylor series).

#include <datamunge/algebra/calculus.hpp>
#include <datamunge/algebra/descartes.hpp>
#include <datamunge/algebra/expression.hpp>
#include <datamunge/algebra/factorization.hpp>
#include <datamunge/algebra/groebner.hpp>
#include <datamunge/algebra/interpolation.hpp>
#include <datamunge/algebra/modular.hpp>
#include <datamunge/algebra/monomial_order.hpp>
#include <datamunge/algebra/multivariate_polynomial.hpp>
#include <datamunge/algebra/poly_factor_gf_p.hpp>
#include <datamunge/algebra/poly_gcd.hpp>
#include <datamunge/algebra/polynomial.hpp>
#include <datamunge/algebra/polynomial_roots.hpp>
#include <datamunge/algebra/rational_function.hpp>
#include <datamunge/algebra/resultant.hpp>
#include <datamunge/algebra/square_free.hpp>
#include <datamunge/algebra/sturm.hpp>
