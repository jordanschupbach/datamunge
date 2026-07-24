#pragma once

#include <cmath>
#include <memory>
#include <optional>
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

    /// @brief The symbolic antiderivative with respect to `var` (any other variable name is
    ///        treated as a constant, matching differentiate()). Supports: the sum/difference
    ///        and constant-multiple rules; the power rule x^c (any real c != -1) and its log
    ///        special case (c == -1); sin/cos/exp/log of any expression that is affine
    ///        (a*var + b) in var, via the standard substitution result for each; a constant
    ///        divided by an affine expression (reduces to the log rule); and, for a product of
    ///        two var-dependent factors, generic tabular integration by parts -- reusing
    ///        differentiate() to peel one factor down to the constant 0 across at most 15
    ///        repeated derivatives (terminates for any genuine polynomial factor), paired with
    ///        repeated antiderivatives of the other factor. Not a full computer-algebra
    ///        integrator: general rational functions (a non-constant denominator that isn't
    ///        affine), a variable exponent, and products/quotients outside the forms above
    ///        (e.g. sin(x)*cos(x), which never reduces to zero under repeated differentiation)
    ///        are not attempted.
    /// @throws std::invalid_argument when no rule above applies.
    [[nodiscard]] Expr integrate(const std::string& var) const { return Expr(integrate_node(node_, var)); }

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

    [[nodiscard]] static bool contains_var(const detail::ExprNodePtr& n, const std::string& var) {
        using detail::ExprOp;
        switch (n->op) {
            case ExprOp::Const:
                return false;
            case ExprOp::Var:
                return n->name == var;
            case ExprOp::Neg:
            case ExprOp::Sin:
            case ExprOp::Cos:
            case ExprOp::Exp:
            case ExprOp::Log:
                return contains_var(n->left, var);
            default: // Add, Sub, Mul, Div, Pow
                return contains_var(n->left, var) || contains_var(n->right, var);
        }
    }

    /// @brief Rebuilds `n` with every occurrence of `Var(var)` replaced by `Const(value)`,
    ///        leaving any other variable name untouched (used by try_linear_form() below to
    ///        probe "is this expression affine in var" without needing bindings for every other
    ///        free variable the way evaluate() would).
    [[nodiscard]] static detail::ExprNodePtr substitute_const(const detail::ExprNodePtr& n, const std::string& var,
                                                              double value) {
        if (n->op == detail::ExprOp::Var) return n->name == var ? make_const(value) : n;
        if (n->op == detail::ExprOp::Const) return n;
        if (n->right) return node2(n->op, substitute_const(n->left, var, value), substitute_const(n->right, var, value));
        return node1(n->op, substitute_const(n->left, var, value));
    }

    struct LinearForm {
        double a = 0.0;
        double b = 0.0;
        bool ok = false;
    };

    /// @brief Detects whether `n` is affine in var (n == a*var + b for constants a, b), via
    ///        differentiate()+simplify() (a is the derivative, iff it folds to a single
    ///        constant -- true everywhere iff n is genuinely affine, given exact constant
    ///        folding) and substitute_const()+simplify() at var=0 (b is the resulting
    ///        constant, iff every other name in n also disappeared -- i.e. n has no other free
    ///        variables). Used by the linear-argument substitution rules in integrate_node().
    [[nodiscard]] static LinearForm try_linear_form(const detail::ExprNodePtr& n, const std::string& var) {
        const detail::ExprNodePtr derivative = simplify_node(diff(n, var));
        if (derivative->op != detail::ExprOp::Const) return {};
        const detail::ExprNodePtr at_zero = simplify_node(substitute_const(n, var, 0.0));
        if (at_zero->op != detail::ExprOp::Const) return {};
        return {derivative->value, at_zero->value, true};
    }

    static constexpr int kMaxIntegrationByPartsTerms = 15;

    /// @brief Tabular integration by parts for p_factor * t_factor: attempts to integrate
    ///        t_factor repeatedly while differentiating p_factor repeatedly, summing
    ///        (-1)^i * p_factor^(i) * (i+1)-th antiderivative of t_factor until p_factor's
    ///        i-th derivative simplifies to the constant 0 (a genuine polynomial always
    ///        reaches this within its degree; anything else fails within
    ///        kMaxIntegrationByPartsTerms steps, at which point this returns std::nullopt so
    ///        the caller can try the two factors in the opposite role, or give up).
    [[nodiscard]] static std::optional<detail::ExprNodePtr> try_tabular_by_parts(const detail::ExprNodePtr& p_factor,
                                                                                 const detail::ExprNodePtr& t_factor,
                                                                                 const std::string& var) {
        using detail::ExprOp;
        detail::ExprNodePtr t_integral;
        try {
            t_integral = integrate_node(t_factor, var);
        } catch (const std::invalid_argument&) {
            return std::nullopt;
        }

        detail::ExprNodePtr p_i = p_factor;
        detail::ExprNodePtr result;
        double sign = 1.0;
        for (int i = 0; i < kMaxIntegrationByPartsTerms; ++i) {
            const detail::ExprNodePtr signed_p = sign > 0.0 ? p_i : node1(ExprOp::Neg, p_i);
            const detail::ExprNodePtr term = node2(ExprOp::Mul, signed_p, t_integral);
            result = result ? node2(ExprOp::Add, result, term) : term;

            const detail::ExprNodePtr p_next = simplify_node(diff(p_i, var));
            if (p_next->op == ExprOp::Const && p_next->value == 0.0) return result;

            try {
                t_integral = integrate_node(t_integral, var);
            } catch (const std::invalid_argument&) {
                return std::nullopt;
            }
            p_i = p_next;
            sign = -sign;
        }
        return std::nullopt;
    }

    [[nodiscard]] static detail::ExprNodePtr integrate_node(const detail::ExprNodePtr& n, const std::string& var) {
        using detail::ExprOp;

        if (!contains_var(n, var)) return node2(ExprOp::Mul, n, make_var(var));

        switch (n->op) {
            case ExprOp::Const:
            case ExprOp::Var:
                // n->name == var is guaranteed here (the contains_var() guard above already
                // excludes both Const and any other-named Var).
                return node2(ExprOp::Div, node2(ExprOp::Pow, n, make_const(2.0)), make_const(2.0));
            case ExprOp::Add:
                return node2(ExprOp::Add, integrate_node(n->left, var), integrate_node(n->right, var));
            case ExprOp::Sub:
                return node2(ExprOp::Sub, integrate_node(n->left, var), integrate_node(n->right, var));
            case ExprOp::Neg:
                return node1(ExprOp::Neg, integrate_node(n->left, var));
            case ExprOp::Mul: {
                if (!contains_var(n->left, var)) return node2(ExprOp::Mul, n->left, integrate_node(n->right, var));
                if (!contains_var(n->right, var)) return node2(ExprOp::Mul, integrate_node(n->left, var), n->right);
                if (auto ibp = try_tabular_by_parts(n->left, n->right, var)) return *ibp;
                if (auto ibp = try_tabular_by_parts(n->right, n->left, var)) return *ibp;
                throw std::invalid_argument("Expr::integrate: unsupported product of two var-dependent factors");
            }
            case ExprOp::Div: {
                if (!contains_var(n->right, var)) return node2(ExprOp::Div, integrate_node(n->left, var), n->right);
                if (!contains_var(n->left, var)) {
                    const LinearForm lf = try_linear_form(n->right, var);
                    if (lf.ok && lf.a != 0.0) {
                        return node2(ExprOp::Mul, node2(ExprOp::Div, n->left, make_const(lf.a)),
                                     node1(ExprOp::Log, n->right));
                    }
                }
                throw std::invalid_argument("Expr::integrate: division by a non-constant, non-affine "
                                             "denominator is not supported");
            }
            case ExprOp::Pow: {
                if (n->right->op != ExprOp::Const)
                    throw std::invalid_argument("Expr::integrate: integration of a non-constant exponent is not "
                                                 "supported");
                const LinearForm lf = try_linear_form(n->left, var);
                if (!lf.ok)
                    throw std::invalid_argument("Expr::integrate: power of a non-affine base is not supported");
                const double c = n->right->value;
                if (c == -1.0) return node2(ExprOp::Div, node1(ExprOp::Log, n->left), make_const(lf.a));
                return node2(ExprOp::Div, node2(ExprOp::Pow, n->left, make_const(c + 1.0)),
                             make_const(lf.a * (c + 1.0)));
            }
            case ExprOp::Sin: {
                const LinearForm lf = try_linear_form(n->left, var);
                if (!lf.ok) throw std::invalid_argument("Expr::integrate: sin of a non-affine argument is not supported");
                return node2(ExprOp::Div, node1(ExprOp::Neg, node1(ExprOp::Cos, n->left)), make_const(lf.a));
            }
            case ExprOp::Cos: {
                const LinearForm lf = try_linear_form(n->left, var);
                if (!lf.ok) throw std::invalid_argument("Expr::integrate: cos of a non-affine argument is not supported");
                return node2(ExprOp::Div, node1(ExprOp::Sin, n->left), make_const(lf.a));
            }
            case ExprOp::Exp: {
                const LinearForm lf = try_linear_form(n->left, var);
                if (!lf.ok) throw std::invalid_argument("Expr::integrate: exp of a non-affine argument is not supported");
                return node2(ExprOp::Div, node1(ExprOp::Exp, n->left), make_const(lf.a));
            }
            case ExprOp::Log: {
                const LinearForm lf = try_linear_form(n->left, var);
                if (!lf.ok) throw std::invalid_argument("Expr::integrate: log of a non-affine argument is not supported");
                const detail::ExprNodePtr inner_over_a = node2(ExprOp::Div, n->left, make_const(lf.a));
                const detail::ExprNodePtr log_minus_one =
                    node2(ExprOp::Sub, node1(ExprOp::Log, n->left), make_const(1.0));
                return node2(ExprOp::Mul, inner_over_a, log_minus_one);
            }
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
