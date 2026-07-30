#pragma once

/// \file shunting_yard.hpp
/// \brief Dijkstra's shunting-yard algorithm: convert an infix arithmetic expression
///        to postfix (Reverse Polish Notation), and evaluate it (Dijkstra 1961).
///
/// Infix notation (`3 + 4 * 2`) needs precedence and parenthesis rules to parse;
/// *postfix* notation (`3 4 2 * +`) needs none -- it evaluates with a single stack. The
/// *shunting-yard algorithm* converts one to the other in a single left-to-right pass
/// using an operator stack, so named because operators are shunted onto a side track
/// like railway cars. Numbers go straight to the output; an operator pops higher-or-
/// equal-precedence operators to the output before being pushed (respecting left/right
/// associativity); parentheses gate the popping. It is the classic bridge between the
/// notation humans write and the stack machine a computer runs, and the seed of
/// operator-precedence parsing.

#include <cctype>
#include <cmath>
#include <stack>
#include <string>
#include <vector>

namespace datamunge::algorithms {

namespace detail {
inline int  precedence(char op) { return (op == '+' || op == '-') ? 1 : (op == '*' || op == '/') ? 2 : (op == '^') ? 3 : 0; }
inline bool right_assoc(char op) { return op == '^'; }
inline bool is_op(char c) { return c == '+' || c == '-' || c == '*' || c == '/' || c == '^'; }
}  // namespace detail

/// \brief Convert an infix expression to space-separated postfix (RPN).
///
/// Supports multi-digit non-negative numbers, the binary operators + - * / ^ (with ^
/// right-associative), and parentheses. Tokens may be separated by spaces or run
/// together.
inline std::string shunting_yard_to_postfix(const std::string& infix) {
    std::string      output;
    std::stack<char> ops;
    auto             emit = [&](const std::string& tok) {
        if (!output.empty()) output += ' ';
        output += tok;
    };
    for (std::size_t i = 0; i < infix.size();) {
        const char c = infix[i];
        if (std::isspace(static_cast<unsigned char>(c))) { ++i; continue; }
        if (std::isdigit(static_cast<unsigned char>(c)) || c == '.') {
            std::string num;
            while (i < infix.size() && (std::isdigit(static_cast<unsigned char>(infix[i])) || infix[i] == '.')) num += infix[i++];
            emit(num);
        } else if (detail::is_op(c)) {
            while (!ops.empty() && detail::is_op(ops.top()) &&
                   (detail::precedence(ops.top()) > detail::precedence(c) ||
                    (detail::precedence(ops.top()) == detail::precedence(c) && !detail::right_assoc(c)))) {
                emit(std::string(1, ops.top()));
                ops.pop();
            }
            ops.push(c);
            ++i;
        } else if (c == '(') {
            ops.push(c);
            ++i;
        } else if (c == ')') {
            while (!ops.empty() && ops.top() != '(') { emit(std::string(1, ops.top())); ops.pop(); }
            if (!ops.empty()) ops.pop();  // discard '('
            ++i;
        } else {
            ++i;  // skip unknown
        }
    }
    while (!ops.empty()) { emit(std::string(1, ops.top())); ops.pop(); }
    return output;
}

/// \brief Evaluate a space-separated postfix (RPN) expression to a double.
inline double evaluate_rpn(const std::string& postfix) {
    std::stack<double> st;
    std::size_t        i = 0;
    while (i < postfix.size()) {
        if (std::isspace(static_cast<unsigned char>(postfix[i]))) { ++i; continue; }
        const char c = postfix[i];
        if (detail::is_op(c) && (i + 1 >= postfix.size() || postfix[i + 1] == ' ' || postfix[i + 1] == '\0')) {
            const double b = st.top(); st.pop();
            const double a = st.top(); st.pop();
            double       r = 0.0;
            switch (c) {
                case '+': r = a + b; break;
                case '-': r = a - b; break;
                case '*': r = a * b; break;
                case '/': r = a / b; break;
                case '^': r = std::pow(a, b); break;
            }
            st.push(r);
            ++i;
        } else {
            std::string num;
            while (i < postfix.size() && postfix[i] != ' ') num += postfix[i++];
            st.push(std::stod(num));
        }
    }
    return st.empty() ? 0.0 : st.top();
}

/// Convenience: parse and evaluate an infix expression directly.
inline double shunting_yard_evaluate(const std::string& infix) {
    return evaluate_rpn(shunting_yard_to_postfix(infix));
}

}  // namespace datamunge::algorithms
