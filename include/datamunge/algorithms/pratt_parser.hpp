#pragma once

/// \file pratt_parser.hpp
/// \brief A Pratt (top-down operator-precedence) parser/evaluator for arithmetic expressions.
///
/// Vaughan Pratt's 1973 parser replaces one-function-per-precedence-level (as in recursive
/// descent) with a single expression routine driven by numeric *binding powers*. Each
/// operator gets a left and right binding power; the core loop keeps consuming operators as
/// long as their left binding power exceeds the current context's minimum:
/// \f[
///   \text{parse\_expr}(\text{min\_bp}):\ \text{lhs} \gets \text{atom};\
///   \textbf{while } \text{lbp}(\text{op}) > \text{min\_bp}:\ \text{lhs} \gets
///   \text{op}(\text{lhs},\ \text{parse\_expr}(\text{rbp}(\text{op}))).
/// \f]
/// Left-associative operators use \f$\text{rbp}=\text{lbp}\f$; right-associative ones (like
/// exponentiation) use \f$\text{rbp}=\text{lbp}-1\f$, so a lower right power lets the operator
/// re-associate to the right. This one idea captures precedence, associativity, prefix, and
/// infix operators uniformly, which is why Pratt parsing is popular for extensible expression
/// languages (it is the parser behind many real-world interpreters). This implementation
/// evaluates to a =double= and supports right-associative \f$\wedge\f$ (power).

#include <cctype>
#include <cmath>
#include <stdexcept>
#include <string>

namespace datamunge::algorithms {

namespace detail {

class PrattEvaluator {
   public:
    explicit PrattEvaluator(const std::string& text) : s_(text) {}

    double run() {
        double v = parse_expr(0);
        if (peek() != '\0') throw std::runtime_error("pratt: trailing characters");
        return v;
    }

   private:
    const std::string& s_;
    std::size_t pos_ = 0;

    void skip_ws() {
        while (pos_ < s_.size() && std::isspace(static_cast<unsigned char>(s_[pos_]))) ++pos_;
    }
    char peek() {
        skip_ws();
        return pos_ < s_.size() ? s_[pos_] : '\0';
    }

    // Left binding power of an infix operator; 0 means "not an infix operator".
    static int lbp(char op) {
        switch (op) {
            case '+': case '-': return 10;
            case '*': case '/': return 20;
            case '^': return 30;
            default: return 0;
        }
    }
    // Right binding power: equal to lbp for left-assoc; lbp-1 for right-assoc '^'.
    static int rbp(char op) { return op == '^' ? lbp(op) - 1 : lbp(op); }

    double parse_expr(int min_bp) {
        double lhs = atom();
        for (;;) {
            char op = peek();
            int power = lbp(op);
            if (power == 0 || power <= min_bp) break;
            ++pos_;  // consume operator
            double rhs = parse_expr(rbp(op));
            switch (op) {
                case '+': lhs += rhs; break;
                case '-': lhs -= rhs; break;
                case '*': lhs *= rhs; break;
                case '/': lhs /= rhs; break;
                case '^': lhs = std::pow(lhs, rhs); break;
            }
        }
        return lhs;
    }

    // atom = number | '(' expr ')' | ('-'|'+') atom  (prefix unary)
    double atom() {
        char c = peek();
        if (c == '(') {
            ++pos_;
            double v = parse_expr(0);
            if (peek() != ')') throw std::runtime_error("pratt: expected ')'");
            ++pos_;
            return v;
        }
        if (c == '-') { ++pos_; return -atom(); }
        if (c == '+') { ++pos_; return atom(); }
        skip_ws();
        std::size_t start = pos_;
        while (pos_ < s_.size() &&
               (std::isdigit(static_cast<unsigned char>(s_[pos_])) || s_[pos_] == '.'))
            ++pos_;
        if (pos_ == start) throw std::runtime_error("pratt: expected a number");
        return std::stod(s_.substr(start, pos_ - start));
    }
};

}  // namespace detail

/// \brief Parse and evaluate an arithmetic expression by Pratt precedence-climbing.
/// Supports +, -, *, /, right-associative ^ (power), unary minus, and parentheses.
/// \throws std::runtime_error on malformed input.
inline double pratt_eval(const std::string& expression) {
    detail::PrattEvaluator ev(expression);
    return ev.run();
}

}  // namespace datamunge::algorithms
