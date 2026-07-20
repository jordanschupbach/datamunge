#pragma once

#include <cmath>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace datamunge::algebra {

namespace detail {

enum class ExprOp { Const, Var, Add, Sub, Mul, Div, Pow, Neg, Sin, Cos, Exp, Log };

struct ExprNode {
    ExprOp op = ExprOp::Const;
    double value = 0.0; // used when op == Const
    std::string name;   // used when op == Var
    std::shared_ptr<ExprNode> left;
    std::shared_ptr<ExprNode> right; // unused for unary ops (Neg, Sin, Cos, Exp, Log)
};
using ExprNodePtr = std::shared_ptr<ExprNode>;

} // namespace detail

/// @brief A symbolic expression tree over +, -, *, /, ^ (power) and sin/cos/exp/log, supporting
///        symbolic differentiation, basic algebraic simplification, and numeric evaluation.
///        Value type wrapping a shared, immutable node tree -- copies are cheap and structural
///        sharing across copies is always safe since nodes are never mutated after
///        construction.
class Expr {
 public:
    Expr() : node_(make_const(0.0)) {}

    [[nodiscard]] static Expr constant(double value) { return Expr(make_const(value)); }
    [[nodiscard]] static Expr variable(const std::string& name) { return Expr(make_var(name)); }

    [[nodiscard]] Expr add(const Expr& other) const { return binary(detail::ExprOp::Add, other); }
    [[nodiscard]] Expr subtract(const Expr& other) const { return binary(detail::ExprOp::Sub, other); }
    [[nodiscard]] Expr multiply(const Expr& other) const { return binary(detail::ExprOp::Mul, other); }
    [[nodiscard]] Expr divide(const Expr& other) const { return binary(detail::ExprOp::Div, other); }
    [[nodiscard]] Expr power(const Expr& exponent) const { return binary(detail::ExprOp::Pow, exponent); }
    [[nodiscard]] Expr negate() const { return unary(detail::ExprOp::Neg); }
    [[nodiscard]] Expr sin() const { return unary(detail::ExprOp::Sin); }
    [[nodiscard]] Expr cos() const { return unary(detail::ExprOp::Cos); }
    [[nodiscard]] Expr exp() const { return unary(detail::ExprOp::Exp); }
    [[nodiscard]] Expr log() const { return unary(detail::ExprOp::Log); }

    /// @brief The symbolic derivative with respect to `var` (any other variable name is
    ///        treated as a constant).
    /// @throws std::invalid_argument if the expression contains `base ^ exponent` where
    ///         `exponent` is not a constant (only the constant-exponent power rule is
    ///         supported).
    [[nodiscard]] Expr differentiate(const std::string& var) const { return Expr(diff(node_, var)); }

    /// @brief Basic algebraic simplification: constant folding, plus identities like x+0, x*1,
    ///        x*0, x/1, x^0, x^1. Not a full computer-algebra simplifier (no factoring,
    ///        collecting like terms, or trig identities).
    [[nodiscard]] Expr simplify() const { return Expr(simplify_node(node_)); }

    /// @brief Evaluates the expression, binding each entry of `names` to the corresponding
    ///        entry of `values` (parallel arrays, must be the same length).
    /// @throws std::invalid_argument if names/values sizes disagree or a variable in the
    ///         expression has no binding.
    [[nodiscard]] double evaluate(const std::vector<std::string>& names, const std::vector<double>& values) const {
        if (names.size() != values.size()) throw std::invalid_argument("names/values must be the same size");
        return eval(node_, names, values);
    }

    [[nodiscard]] std::string to_string() const { return to_str(node_); }

 private:
    explicit Expr(detail::ExprNodePtr node) : node_(std::move(node)) {}

    [[nodiscard]] Expr binary(detail::ExprOp op, const Expr& other) const { return Expr(node2(op, node_, other.node_)); }
    [[nodiscard]] Expr unary(detail::ExprOp op) const { return Expr(node1(op, node_)); }

    [[nodiscard]] static detail::ExprNodePtr make_const(double v) {
        auto n = std::make_shared<detail::ExprNode>();
        n->op = detail::ExprOp::Const;
        n->value = v;
        return n;
    }
    [[nodiscard]] static detail::ExprNodePtr make_var(const std::string& name) {
        auto n = std::make_shared<detail::ExprNode>();
        n->op = detail::ExprOp::Var;
        n->name = name;
        return n;
    }
    [[nodiscard]] static detail::ExprNodePtr node1(detail::ExprOp op, detail::ExprNodePtr a) {
        auto n = std::make_shared<detail::ExprNode>();
        n->op = op;
        n->left = std::move(a);
        return n;
    }
    [[nodiscard]] static detail::ExprNodePtr node2(detail::ExprOp op, detail::ExprNodePtr a, detail::ExprNodePtr b) {
        auto n = std::make_shared<detail::ExprNode>();
        n->op = op;
        n->left = std::move(a);
        n->right = std::move(b);
        return n;
    }

    [[nodiscard]] static double eval(const detail::ExprNodePtr& n, const std::vector<std::string>& names, const std::vector<double>& values) {
        using detail::ExprOp;
        switch (n->op) {
            case ExprOp::Const:
                return n->value;
            case ExprOp::Var:
                for (std::size_t i = 0; i < names.size(); ++i)
                    if (names[i] == n->name) return values[i];
                throw std::invalid_argument("no binding provided for variable '" + n->name + "'");
            case ExprOp::Add:
                return eval(n->left, names, values) + eval(n->right, names, values);
            case ExprOp::Sub:
                return eval(n->left, names, values) - eval(n->right, names, values);
            case ExprOp::Mul:
                return eval(n->left, names, values) * eval(n->right, names, values);
            case ExprOp::Div:
                return eval(n->left, names, values) / eval(n->right, names, values);
            case ExprOp::Pow:
                return std::pow(eval(n->left, names, values), eval(n->right, names, values));
            case ExprOp::Neg:
                return -eval(n->left, names, values);
            case ExprOp::Sin:
                return std::sin(eval(n->left, names, values));
            case ExprOp::Cos:
                return std::cos(eval(n->left, names, values));
            case ExprOp::Exp:
                return std::exp(eval(n->left, names, values));
            case ExprOp::Log:
                return std::log(eval(n->left, names, values));
        }
        throw std::logic_error("unreachable");
    }

    [[nodiscard]] static detail::ExprNodePtr diff(const detail::ExprNodePtr& n, const std::string& var) {
        using detail::ExprOp;
        switch (n->op) {
            case ExprOp::Const:
                return make_const(0.0);
            case ExprOp::Var:
                return make_const(n->name == var ? 1.0 : 0.0);
            case ExprOp::Add:
                return node2(ExprOp::Add, diff(n->left, var), diff(n->right, var));
            case ExprOp::Sub:
                return node2(ExprOp::Sub, diff(n->left, var), diff(n->right, var));
            case ExprOp::Neg:
                return node1(ExprOp::Neg, diff(n->left, var));
            case ExprOp::Mul:
                // Product rule: (fg)' = f'g + fg'.
                return node2(ExprOp::Add, node2(ExprOp::Mul, diff(n->left, var), n->right),
                             node2(ExprOp::Mul, n->left, diff(n->right, var)));
            case ExprOp::Div: {
                // Quotient rule: (f/g)' = (f'g - fg') / g^2.
                auto numerator = node2(ExprOp::Sub, node2(ExprOp::Mul, diff(n->left, var), n->right),
                                       node2(ExprOp::Mul, n->left, diff(n->right, var)));
                auto denominator = node2(ExprOp::Mul, n->right, n->right);
                return node2(ExprOp::Div, numerator, denominator);
            }
            case ExprOp::Pow: {
                // Power rule for a constant exponent: d/dx f^c = c * f^(c-1) * f'.
                if (n->right->op != ExprOp::Const)
                    throw std::invalid_argument("differentiation of a non-constant exponent is not supported");
                const double c = n->right->value;
                auto reduced_power = node2(ExprOp::Pow, n->left, make_const(c - 1.0));
                return node2(ExprOp::Mul, node2(ExprOp::Mul, make_const(c), reduced_power), diff(n->left, var));
            }
            case ExprOp::Sin:
                return node2(ExprOp::Mul, node1(ExprOp::Cos, n->left), diff(n->left, var));
            case ExprOp::Cos:
                return node2(ExprOp::Mul, node1(ExprOp::Neg, node1(ExprOp::Sin, n->left)), diff(n->left, var));
            case ExprOp::Exp:
                return node2(ExprOp::Mul, node1(ExprOp::Exp, n->left), diff(n->left, var));
            case ExprOp::Log:
                return node2(ExprOp::Div, diff(n->left, var), n->left);
        }
        throw std::logic_error("unreachable");
    }

    [[nodiscard]] static detail::ExprNodePtr simplify_node(const detail::ExprNodePtr& n) {
        using detail::ExprOp;
        if (n->op == ExprOp::Const || n->op == ExprOp::Var) return n;

        const std::vector<std::string> no_names;
        const std::vector<double> no_values;

        if (n->right) {
            auto l = simplify_node(n->left);
            auto r = simplify_node(n->right);
            if (l->op == ExprOp::Const && r->op == ExprOp::Const) return make_const(eval(node2(n->op, l, r), no_names, no_values));
            switch (n->op) {
                case ExprOp::Add:
                    if (l->op == ExprOp::Const && l->value == 0.0) return r;
                    if (r->op == ExprOp::Const && r->value == 0.0) return l;
                    break;
                case ExprOp::Sub:
                    if (r->op == ExprOp::Const && r->value == 0.0) return l;
                    break;
                case ExprOp::Mul:
                    if ((l->op == ExprOp::Const && l->value == 0.0) || (r->op == ExprOp::Const && r->value == 0.0)) return make_const(0.0);
                    if (l->op == ExprOp::Const && l->value == 1.0) return r;
                    if (r->op == ExprOp::Const && r->value == 1.0) return l;
                    break;
                case ExprOp::Div:
                    if (r->op == ExprOp::Const && r->value == 1.0) return l;
                    break;
                case ExprOp::Pow:
                    if (r->op == ExprOp::Const && r->value == 1.0) return l;
                    if (r->op == ExprOp::Const && r->value == 0.0) return make_const(1.0);
                    break;
                default:
                    break;
            }
            return node2(n->op, l, r);
        }

        auto l = simplify_node(n->left);
        if (l->op == ExprOp::Const) return make_const(eval(node1(n->op, l), no_names, no_values));
        return node1(n->op, l);
    }

    [[nodiscard]] static std::string to_str(const detail::ExprNodePtr& n) {
        using detail::ExprOp;
        std::ostringstream out;
        switch (n->op) {
            case ExprOp::Const:
                out << n->value;
                break;
            case ExprOp::Var:
                out << n->name;
                break;
            case ExprOp::Add:
                out << "(" << to_str(n->left) << " + " << to_str(n->right) << ")";
                break;
            case ExprOp::Sub:
                out << "(" << to_str(n->left) << " - " << to_str(n->right) << ")";
                break;
            case ExprOp::Mul:
                out << "(" << to_str(n->left) << " * " << to_str(n->right) << ")";
                break;
            case ExprOp::Div:
                out << "(" << to_str(n->left) << " / " << to_str(n->right) << ")";
                break;
            case ExprOp::Pow:
                out << "(" << to_str(n->left) << " ^ " << to_str(n->right) << ")";
                break;
            case ExprOp::Neg:
                out << "-(" << to_str(n->left) << ")";
                break;
            case ExprOp::Sin:
                out << "sin(" << to_str(n->left) << ")";
                break;
            case ExprOp::Cos:
                out << "cos(" << to_str(n->left) << ")";
                break;
            case ExprOp::Exp:
                out << "exp(" << to_str(n->left) << ")";
                break;
            case ExprOp::Log:
                out << "log(" << to_str(n->left) << ")";
                break;
        }
        return out.str();
    }

    detail::ExprNodePtr node_;
};

} // namespace datamunge::algebra
