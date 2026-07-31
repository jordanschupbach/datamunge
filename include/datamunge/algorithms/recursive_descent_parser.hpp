#pragma once

/// \file recursive_descent_parser.hpp
/// \brief A recursive-descent parser/evaluator for arithmetic expressions.
///
/// Recursive descent is the most direct way to parse an *LL(k)* grammar: write one function
/// per nonterminal, and let the call stack mirror the grammar's structure. For arithmetic we
/// encode precedence directly in the grammar,
/// \f[
///   \text{expr} \to \text{term}\;((+\!\mid\!-)\;\text{term})^\ast,\quad
///   \text{term} \to \text{factor}\;((\ast\!\mid/)\;\text{factor})^\ast,\quad
///   \text{factor} \to \text{number}\mid(\;\text{expr}\;)\mid-\text{factor},
/// \f]
/// so that =expr= handles the lowest-precedence operators, =term= the next, and =factor= the
/// atoms and parentheses. Because =factor= can recurse back into =expr= through parentheses,
/// the parser handles arbitrary nesting. This is the technique behind most hand-written
/// compilers and interpreters: transparent, easy to extend, and producing good error
/// messages -- at the cost of not handling left-recursive grammars (which would loop forever).
///
/// This implementation evaluates directly to a =double= as it parses, and throws on malformed
/// input.

#include <cctype>
#include <stdexcept>
#include <string>

namespace datamunge::algorithms {

namespace detail {

/// Single-pass recursive-descent evaluator over a character buffer.
class RecursiveDescentEvaluator {
   public:
    explicit RecursiveDescentEvaluator(const std::string& text) : s_(text) {}

    double run() {
        double v = expr();
        skip_ws();
        if (pos_ != s_.size())
            throw std::runtime_error("recursive_descent: trailing characters at position " +
                                     std::to_string(pos_));
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

    // expr = term (('+' | '-') term)*
    double expr() {
        double v = term();
        for (;;) {
            char c = peek();
            if (c == '+') { ++pos_; v += term(); }
            else if (c == '-') { ++pos_; v -= term(); }
            else return v;
        }
    }
    // term = factor (('*' | '/') factor)*
    double term() {
        double v = factor();
        for (;;) {
            char c = peek();
            if (c == '*') { ++pos_; v *= factor(); }
            else if (c == '/') { ++pos_; v /= factor(); }
            else return v;
        }
    }
    // factor = number | '(' expr ')' | '-' factor
    double factor() {
        char c = peek();
        if (c == '(') {
            ++pos_;
            double v = expr();
            if (peek() != ')') throw std::runtime_error("recursive_descent: expected ')'");
            ++pos_;
            return v;
        }
        if (c == '-') { ++pos_; return -factor(); }
        if (c == '+') { ++pos_; return factor(); }
        return number();
    }
    double number() {
        skip_ws();
        std::size_t start = pos_;
        while (pos_ < s_.size() &&
               (std::isdigit(static_cast<unsigned char>(s_[pos_])) || s_[pos_] == '.'))
            ++pos_;
        if (pos_ == start) throw std::runtime_error("recursive_descent: expected a number");
        return std::stod(s_.substr(start, pos_ - start));
    }
};

}  // namespace detail

/// \brief Parse and evaluate an arithmetic expression by recursive descent.
/// Supports +, -, *, /, unary minus, parentheses, and decimal numbers.
/// \throws std::runtime_error on malformed input.
inline double recursive_descent_eval(const std::string& expression) {
    detail::RecursiveDescentEvaluator ev(expression);
    return ev.run();
}

}  // namespace datamunge::algorithms
