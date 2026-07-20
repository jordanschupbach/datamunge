#pragma once

#include <datamunge/algebra/poly_gcd.hpp>
#include <datamunge/algebra/polynomial.hpp>

#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace datamunge::algebra {

/// @brief A ratio of two polynomials (numerator / denominator), kept automatically reduced by
///        their GCD -- e.g. (x^2-1)/(x-1) simplifies to x+1 -- with the denominator normalized
///        to monic.
class RationalFunction {
 public:
    RationalFunction() : numerator_(0.0), denominator_(1.0) {}

    RationalFunction(Polynomial numerator, Polynomial denominator)
        : numerator_(std::move(numerator)), denominator_(std::move(denominator)) {
        if (denominator_.is_zero()) throw std::invalid_argument("denominator must be nonzero");
        reduce();
    }

    [[nodiscard]] const Polynomial& numerator() const { return numerator_; }
    [[nodiscard]] const Polynomial& denominator() const { return denominator_; }

    /// @throws std::domain_error if x is a pole (denominator evaluates to zero at x).
    [[nodiscard]] double evaluate(double x) const {
        const double d = denominator_.evaluate(x);
        if (d == 0.0) throw std::domain_error("rational function has a pole at x");
        return numerator_.evaluate(x) / d;
    }

    [[nodiscard]] RationalFunction add(const RationalFunction& other) const {
        return RationalFunction(numerator_.multiply(other.denominator_).add(other.numerator_.multiply(denominator_)),
                                 denominator_.multiply(other.denominator_));
    }

    [[nodiscard]] RationalFunction subtract(const RationalFunction& other) const {
        return RationalFunction(numerator_.multiply(other.denominator_).subtract(other.numerator_.multiply(denominator_)),
                                 denominator_.multiply(other.denominator_));
    }

    [[nodiscard]] RationalFunction multiply(const RationalFunction& other) const {
        return RationalFunction(numerator_.multiply(other.numerator_), denominator_.multiply(other.denominator_));
    }

    /// @throws std::invalid_argument if other is the zero rational function.
    [[nodiscard]] RationalFunction divide(const RationalFunction& other) const {
        if (other.numerator_.is_zero()) throw std::invalid_argument("division by a zero rational function");
        return RationalFunction(numerator_.multiply(other.denominator_), denominator_.multiply(other.numerator_));
    }

    [[nodiscard]] std::string to_string() const {
        std::ostringstream out;
        out << "(" << numerator_.to_string() << ") / (" << denominator_.to_string() << ")";
        return out.str();
    }

 private:
    void reduce() {
        const Polynomial g = poly_gcd(numerator_, denominator_);
        if (g.degree() > 0) {
            numerator_ = numerator_.divmod(g).first;
            denominator_ = denominator_.divmod(g).first;
        }
        const double lead = denominator_.coefficient(denominator_.degree());
        if (lead != 1.0) {
            numerator_ = numerator_.scale(1.0 / lead);
            denominator_ = denominator_.scale(1.0 / lead);
        }
    }

    Polynomial numerator_;
    Polynomial denominator_;
};

} // namespace datamunge::algebra
