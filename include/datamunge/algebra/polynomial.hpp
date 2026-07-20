#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace datamunge::algebra {

/// @brief Dense univariate polynomial over the reals, stored as coefficients in ascending
///        degree order (coefficients()[i] is the coefficient of x^i). Always kept trimmed --
///        no trailing (highest-degree) zero coefficients, except for the zero polynomial itself,
///        which is stored as the single coefficient {0.0}.
class Polynomial {
 public:
    Polynomial() : coeffs_{0.0} {}
    explicit Polynomial(double constant) : coeffs_{constant} { trim(); }
    explicit Polynomial(std::vector<double> coeffs) : coeffs_(std::move(coeffs)) {
        if (coeffs_.empty()) coeffs_.push_back(0.0);
        trim();
    }

    /// @brief Degree of the polynomial; the zero polynomial has degree 0 (there is no
    ///        universally-agreed convention for its "true" degree of -infinity, and 0 keeps
    ///        every coefficient()/loop bound in this module well-defined).
    [[nodiscard]] int degree() const { return static_cast<int>(coeffs_.size()) - 1; }

    [[nodiscard]] bool is_zero() const { return coeffs_.size() == 1 && coeffs_[0] == 0.0; }

    /// @brief Coefficient of x^i; 0.0 for any i outside [0, degree()] (including negative i).
    [[nodiscard]] double coefficient(int i) const {
        if (i < 0 || static_cast<std::size_t>(i) >= coeffs_.size()) return 0.0;
        return coeffs_[static_cast<std::size_t>(i)];
    }

    [[nodiscard]] const std::vector<double>& coefficients() const { return coeffs_; }

    /// @brief Evaluates the polynomial at x via Horner's method.
    [[nodiscard]] double evaluate(double x) const {
        double result = 0.0;
        for (auto it = coeffs_.rbegin(); it != coeffs_.rend(); ++it) result = result * x + *it;
        return result;
    }

    [[nodiscard]] Polynomial derivative() const {
        if (degree() <= 0) return Polynomial(0.0);
        std::vector<double> d(coeffs_.size() - 1);
        for (std::size_t i = 1; i < coeffs_.size(); ++i) d[i - 1] = coeffs_[i] * static_cast<double>(i);
        return Polynomial(std::move(d));
    }

    /// @brief The antiderivative with zero constant term (i.e. integral from 0 to x, as a
    ///        polynomial in x).
    [[nodiscard]] Polynomial antiderivative() const {
        std::vector<double> a(coeffs_.size() + 1, 0.0);
        for (std::size_t i = 0; i < coeffs_.size(); ++i) a[i + 1] = coeffs_[i] / static_cast<double>(i + 1);
        return Polynomial(std::move(a));
    }

    [[nodiscard]] Polynomial add(const Polynomial& other) const {
        std::vector<double> r(std::max(coeffs_.size(), other.coeffs_.size()), 0.0);
        for (std::size_t i = 0; i < coeffs_.size(); ++i) r[i] += coeffs_[i];
        for (std::size_t i = 0; i < other.coeffs_.size(); ++i) r[i] += other.coeffs_[i];
        return Polynomial(std::move(r));
    }

    [[nodiscard]] Polynomial subtract(const Polynomial& other) const {
        std::vector<double> r(std::max(coeffs_.size(), other.coeffs_.size()), 0.0);
        for (std::size_t i = 0; i < coeffs_.size(); ++i) r[i] += coeffs_[i];
        for (std::size_t i = 0; i < other.coeffs_.size(); ++i) r[i] -= other.coeffs_[i];
        return Polynomial(std::move(r));
    }

    [[nodiscard]] Polynomial multiply(const Polynomial& other) const {
        if (is_zero() || other.is_zero()) return Polynomial(0.0);
        std::vector<double> r(coeffs_.size() + other.coeffs_.size() - 1, 0.0);
        for (std::size_t i = 0; i < coeffs_.size(); ++i)
            for (std::size_t j = 0; j < other.coeffs_.size(); ++j) r[i + j] += coeffs_[i] * other.coeffs_[j];
        return Polynomial(std::move(r));
    }

    [[nodiscard]] Polynomial scale(double factor) const {
        std::vector<double> r = coeffs_;
        for (auto& c : r) c *= factor;
        return Polynomial(std::move(r));
    }

    [[nodiscard]] Polynomial negate() const { return scale(-1.0); }

    /// @brief Polynomial long division: returns {quotient, remainder} such that
    ///        `*this == quotient.multiply(divisor).add(remainder)` and
    ///        `remainder.degree() < divisor.degree()` (or remainder is zero).
    /// @throws std::invalid_argument if divisor is the zero polynomial.
    [[nodiscard]] std::pair<Polynomial, Polynomial> divmod(const Polynomial& divisor) const {
        if (divisor.is_zero()) throw std::invalid_argument("division by the zero polynomial");
        const int dividend_degree = degree();
        const int divisor_degree = divisor.degree();
        if (is_zero() || dividend_degree < divisor_degree) return {Polynomial(0.0), *this};

        std::vector<double> remainder = coeffs_;
        std::vector<double> quotient(static_cast<std::size_t>(dividend_degree - divisor_degree + 1), 0.0);
        const double lead = divisor.coeffs_.back();
        for (int k = dividend_degree - divisor_degree; k >= 0; --k) {
            const double coef = remainder[static_cast<std::size_t>(k + divisor_degree)] / lead;
            quotient[static_cast<std::size_t>(k)] = coef;
            for (int j = 0; j <= divisor_degree; ++j) {
                remainder[static_cast<std::size_t>(k + j)] -= coef * divisor.coeffs_[static_cast<std::size_t>(j)];
            }
        }
        return {Polynomial(std::move(quotient)), Polynomial(std::move(remainder))};
    }

    [[nodiscard]] std::string to_string() const {
        std::ostringstream out;
        bool first = true;
        for (int i = degree(); i >= 0; --i) {
            const double c = coefficient(i);
            if (c == 0.0 && !(first && i == 0)) continue;
            if (!first) out << (c < 0 ? " - " : " + ");
            else if (c < 0) out << "-";
            const double mag = std::fabs(c);
            if (i == 0 || mag != 1.0) out << mag;
            if (i > 0) {
                out << "x";
                if (i > 1) out << "^" << i;
            }
            first = false;
        }
        return out.str();
    }

    /// @brief Member (not free-function) equality, matching Point2D's convention -- SWIG maps
    ///        a free `operator==` to a single module-level `__eq__` in the Python backend,
    ///        which collides across every type that defines one (e.g. image::Pixel's), while
    ///        a member operator becomes a per-class `__eq__` with no such collision.
    [[nodiscard]] bool operator==(const Polynomial& other) const { return coeffs_ == other.coeffs_; }
    [[nodiscard]] bool operator!=(const Polynomial& other) const { return !(*this == other); }

 private:
    void trim() {
        while (coeffs_.size() > 1 && coeffs_.back() == 0.0) coeffs_.pop_back();
    }

    std::vector<double> coeffs_;
};

/// @brief The quotient from a.divmod(b) -- a binding-friendly alternative to divmod() itself,
///        which returns a std::pair<Polynomial, Polynomial> that this codebase avoids exposing
///        directly to SWIG bindings (matching the vector-returning-alternative pattern used
///        elsewhere for similarly pair/map-shaped C++ APIs).
[[nodiscard]] inline Polynomial poly_quotient(const Polynomial& a, const Polynomial& b) { return a.divmod(b).first; }

/// @brief The remainder from a.divmod(b); see poly_quotient() for why this exists alongside
///        the pair-returning divmod() method.
[[nodiscard]] inline Polynomial poly_remainder(const Polynomial& a, const Polynomial& b) { return a.divmod(b).second; }

} // namespace datamunge::algebra
