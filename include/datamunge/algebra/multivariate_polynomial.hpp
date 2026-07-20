#pragma once

#include <datamunge/algebra/monomial_order.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace datamunge::algebra {

/// @brief Sparse multivariate polynomial over the reals in a fixed number of variables (x0,
///        x1, ...), stored internally as a map from exponent vector to coefficient. Exposed to
///        bindings via parallel arrays (exponents()/coefficients()) rather than the internal
///        map, matching this module's std::map-avoidance convention for SWIG-facing types.
class MultivariatePolynomial {
 public:
    MultivariatePolynomial() : num_variables_(0) {}

    explicit MultivariatePolynomial(int num_variables) : num_variables_(num_variables) {
        if (num_variables < 0) throw std::invalid_argument("num_variables must be nonnegative");
    }

    /// @param num_variables Number of variables every exponent vector must have.
    /// @param exponents One entry per term; each inner vector has length num_variables.
    /// @param coefficients Parallel to exponents; terms with equal exponent vectors are
    ///        summed, and any resulting zero coefficient is dropped.
    MultivariatePolynomial(int num_variables, const std::vector<std::vector<int>>& exponents,
                            const std::vector<double>& coefficients)
        : num_variables_(num_variables) {
        if (num_variables < 0) throw std::invalid_argument("num_variables must be nonnegative");
        if (exponents.size() != coefficients.size())
            throw std::invalid_argument("exponents/coefficients must be the same size");
        for (std::size_t i = 0; i < exponents.size(); ++i) {
            if (static_cast<int>(exponents[i].size()) != num_variables)
                throw std::invalid_argument("every exponent vector must have length num_variables");
            add_term(exponents[i], coefficients[i]);
        }
    }

    [[nodiscard]] int num_variables() const { return num_variables_; }
    [[nodiscard]] int num_terms() const { return static_cast<int>(terms_.size()); }
    [[nodiscard]] bool is_zero() const { return terms_.empty(); }

    [[nodiscard]] std::vector<std::vector<int>> exponents() const {
        std::vector<std::vector<int>> result;
        result.reserve(terms_.size());
        for (const auto& term : terms_) result.push_back(term.first);
        return result;
    }

    [[nodiscard]] std::vector<double> coefficients() const {
        std::vector<double> result;
        result.reserve(terms_.size());
        for (const auto& term : terms_) result.push_back(term.second);
        return result;
    }

    [[nodiscard]] double evaluate(const std::vector<double>& point) const {
        if (static_cast<int>(point.size()) != num_variables_)
            throw std::invalid_argument("point must have num_variables entries");
        double total = 0.0;
        for (const auto& term : terms_) {
            double value = term.second;
            for (int v = 0; v < num_variables_; ++v)
                value *= std::pow(point[static_cast<std::size_t>(v)], term.first[static_cast<std::size_t>(v)]);
            total += value;
        }
        return total;
    }

    [[nodiscard]] MultivariatePolynomial add(const MultivariatePolynomial& other) const {
        check_compatible(other);
        MultivariatePolynomial result = *this;
        for (const auto& term : other.terms_) result.add_term(term.first, term.second);
        return result;
    }

    [[nodiscard]] MultivariatePolynomial subtract(const MultivariatePolynomial& other) const {
        check_compatible(other);
        MultivariatePolynomial result = *this;
        for (const auto& term : other.terms_) result.add_term(term.first, -term.second);
        return result;
    }

    [[nodiscard]] MultivariatePolynomial multiply(const MultivariatePolynomial& other) const {
        check_compatible(other);
        MultivariatePolynomial result(num_variables_);
        for (const auto& a : terms_) {
            for (const auto& b : other.terms_) {
                std::vector<int> exp(static_cast<std::size_t>(num_variables_));
                for (int v = 0; v < num_variables_; ++v) {
                    const std::size_t sv = static_cast<std::size_t>(v);
                    exp[sv] = a.first[sv] + b.first[sv];
                }
                result.add_term(exp, a.second * b.second);
            }
        }
        return result;
    }

    [[nodiscard]] MultivariatePolynomial scale(double factor) const {
        MultivariatePolynomial result(num_variables_);
        for (const auto& term : terms_) result.add_term(term.first, term.second * factor);
        return result;
    }

    /// @throws std::invalid_argument if this is the zero polynomial.
    [[nodiscard]] std::vector<int> leading_exponent(MonomialOrder order) const {
        if (terms_.empty()) throw std::invalid_argument("the zero polynomial has no leading term");
        auto best = terms_.begin();
        for (auto it = std::next(terms_.begin()); it != terms_.end(); ++it) {
            if (detail::compare_monomials(it->first, best->first, order) > 0) best = it;
        }
        return best->first;
    }

    [[nodiscard]] double leading_coefficient(MonomialOrder order) const { return terms_.at(leading_exponent(order)); }

    [[nodiscard]] std::string to_string() const {
        if (terms_.empty()) return "0";

        std::vector<std::pair<std::vector<int>, double>> sorted(terms_.begin(), terms_.end());
        std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) {
            return detail::compare_monomials(a.first, b.first, MonomialOrder::Grevlex) > 0;
        });

        std::ostringstream out;
        bool first = true;
        for (const auto& term : sorted) {
            const double coef = term.second;
            if (!first) out << (coef < 0 ? " - " : " + ");
            else if (coef < 0) out << "-";
            const double mag = std::fabs(coef);

            std::ostringstream vars;
            bool any_var = false;
            for (int v = 0; v < num_variables_; ++v) {
                const int e = term.first[static_cast<std::size_t>(v)];
                if (e == 0) continue;
                any_var = true;
                vars << "x" << v;
                if (e > 1) vars << "^" << e;
            }
            if (!any_var || mag != 1.0) out << mag;
            out << vars.str();
            first = false;
        }
        return out.str();
    }

 private:
    void check_compatible(const MultivariatePolynomial& other) const {
        if (num_variables_ != other.num_variables_) throw std::invalid_argument("polynomials must have the same number of variables");
    }

    void add_term(const std::vector<int>& exp, double coef) {
        auto it = terms_.find(exp);
        if (it == terms_.end()) {
            if (coef != 0.0) terms_.emplace(exp, coef);
        } else {
            it->second += coef;
            if (it->second == 0.0) terms_.erase(it);
        }
    }

    int num_variables_;
    std::map<std::vector<int>, double> terms_;
};

} // namespace datamunge::algebra
