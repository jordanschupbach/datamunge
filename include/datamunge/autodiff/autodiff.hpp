#pragma once

// Convenience umbrella header -- includes every autodiff building block plus the
// generic driver functions built on top of them.

#include <datamunge/autodiff/dual.hpp>
#include <datamunge/autodiff/hyperdual.hpp>
#include <datamunge/autodiff/tape.hpp>
#include <datamunge/linalg/dense.hpp>

#include <cstddef>
#include <utility>
#include <vector>

namespace datamunge::autodiff {

/// @brief Forward-mode derivative of a scalar function f: R -> R at @p x.
///        @p f must be callable as `Dual f(Dual)`.
template <typename F>
double derivative(F&& f, double x) {
    return f(Dual(x, 1.0)).derivative();
}

/// @brief Forward-mode gradient of f: R^n -> R at @p x (one forward pass per input
///        dimension). @p f must be callable as `template<typename T> T f(const std::vector<T>&)`.
template <typename F>
std::vector<double> gradient_forward(F&& f, const std::vector<double>& x) {
    const std::size_t n = x.size();
    std::vector<double> grad(n);
    for (std::size_t i = 0; i < n; ++i) {
        std::vector<Dual> seeded(n);
        for (std::size_t j = 0; j < n; ++j) seeded[j] = Dual(x[j], j == i ? 1.0 : 0.0);
        grad[i] = f(seeded).derivative();
    }
    return grad;
}

/// @brief Reverse-mode value and gradient of f: R^n -> R at @p x, computed in a single
///        forward build of the tape plus one backward sweep. @p f must be callable as
///        `template<typename T> T f(const std::vector<T>&)`.
template <typename F>
std::pair<double, std::vector<double>> gradient_reverse(F&& f, const std::vector<double>& x) {
    Tape tape;
    std::vector<Var> leaves;
    leaves.reserve(x.size());
    for (const double xi : x) leaves.emplace_back(tape, xi);
    const Var y = f(leaves);
    const auto adjoint = tape.backward(y.index());
    std::vector<double> grad(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) grad[i] = adjoint[leaves[i].index()];
    return {y.value(), grad};
}

/// @brief Gradient of f: R^n -> R via reverse mode -- the efficient default for the
///        common case of a scalar output with many inputs (e.g. a loss function).
template <typename F>
std::vector<double> gradient(F&& f, const std::vector<double>& x) {
    return gradient_reverse(std::forward<F>(f), x).second;
}

/// @brief Forward-mode Jacobian of f: R^n -> R^m at @p x (one forward pass per input
///        dimension, yielding a full column of the Jacobian each time). @p f must be
///        callable as `template<typename T> std::vector<T> f(const std::vector<T>&)`.
template <typename F>
linalg::DenseMatrix<double> jacobian_forward(F&& f, const std::vector<double>& x) {
    const std::size_t n = x.size();
    linalg::DenseMatrix<double> jac;
    for (std::size_t i = 0; i < n; ++i) {
        std::vector<Dual> seeded(n);
        for (std::size_t j = 0; j < n; ++j) seeded[j] = Dual(x[j], j == i ? 1.0 : 0.0);
        const std::vector<Dual> outputs = f(seeded);
        if (i == 0) jac = linalg::DenseMatrix<double>(outputs.size(), n, 0.0);
        for (std::size_t r = 0; r < outputs.size(); ++r) jac(r, i) = outputs[r].derivative();
    }
    return jac;
}

/// @brief Hessian of f: R^n -> R at @p x via second-order forward mode (HyperDual),
///        exact to machine precision -- one evaluation per unique (i, j) pair, exploiting
///        symmetry. @p f must be callable as `template<typename T> T f(const std::vector<T>&)`.
template <typename F>
linalg::DenseMatrix<double> hessian(F&& f, const std::vector<double>& x) {
    const std::size_t n = x.size();
    linalg::DenseMatrix<double> H(n, n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i; j < n; ++j) {
            std::vector<HyperDual> seeded(n);
            for (std::size_t k = 0; k < n; ++k)
                seeded[k] = HyperDual(x[k], k == i ? 1.0 : 0.0, k == j ? 1.0 : 0.0, 0.0);
            const double h = f(seeded).eps1eps2();
            H(i, j) = h;
            H(j, i) = h;
        }
    }
    return H;
}

} // namespace datamunge::autodiff
