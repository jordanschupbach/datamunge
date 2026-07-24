#pragma once

namespace datamunge::fem {

/// @brief A user-supplied scalar source-term function of one spatial variable, f(x).
///        Director-enabled extension point (same pattern as datamunge::ode::RHS and
///        datamunge::optim::ArbitraryFunction elsewhere in this codebase): subclass and
///        override evaluate() in C++, or, via SWIG directors, in any binding with director
///        support. Bindings without director support must instead use
///        FEM1D::solve_builtin()'s fixed set of named source terms.
class ScalarField1D {
public:
    virtual ~ScalarField1D() = default;
    virtual double evaluate(double x) = 0;
};

/// @brief A user-supplied scalar source-term function of two spatial variables, f(x, y).
///        Same director-enabled pattern as ScalarField1D; FEM2D::solve_builtin() is the
///        fallback for bindings without director support.
class ScalarField2D {
public:
    virtual ~ScalarField2D() = default;
    virtual double evaluate(double x, double y) = 0;
};

} // namespace datamunge::fem
