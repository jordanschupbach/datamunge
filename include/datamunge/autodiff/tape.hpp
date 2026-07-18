#pragma once

#include <array>
#include <cstddef>
#include <vector>

namespace datamunge::autodiff {

/// @brief A reverse-mode ("backpropagation") computation tape -- an append-only
///        Wengert list. Every Var operation records one node holding its value and
///        the local derivative(s) with respect to its immediate parent node(s); a
///        single backward() sweep then yields the derivative of the seeded output
///        with respect to every node (including every leaf) in one reverse pass.
class Tape {
public:
    Tape() = default;

    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] double value_at(std::size_t index) const;

    /// @brief Records an independent variable (a leaf with no parents).
    std::size_t new_leaf(double value);

    /// @brief Records y = f(a) with local derivative dy/da.
    std::size_t record_unary(double value, std::size_t a, double grad_a);

    /// @brief Records y = f(a, b) with local derivatives dy/da, dy/db.
    std::size_t record_binary(double value, std::size_t a, double grad_a, std::size_t b, double grad_b);

    /// @brief Reverse sweep seeded at node @p output; returns d(output)/d(node) for every node index.
    [[nodiscard]] std::vector<double> backward(std::size_t output) const;

private:
    struct Node {
        double value{0.0};
        std::size_t n_parents{0};
        std::array<std::size_t, 2> parents{0, 0};
        std::array<double, 2> local_grads{0.0, 0.0};
    };

    std::vector<Node> nodes_;
};

/// @brief A handle into a Tape: a value together with the index of the node that
///        produced it. Building an expression out of Var operations records the
///        expression graph onto the underlying Tape as a side effect.
class Var {
public:
    Var(Tape& tape, double value); // records a new leaf
    Var(Tape& tape, std::size_t index, double value) noexcept;

    [[nodiscard]] double value() const noexcept { return value_; }
    [[nodiscard]] std::size_t index() const noexcept { return index_; }
    [[nodiscard]] Tape& tape() const noexcept { return *tape_; }

private:
    Tape* tape_;
    std::size_t index_;
    double value_;
};

Var operator+(const Var& a, const Var& b);
Var operator-(const Var& a, const Var& b);
Var operator*(const Var& a, const Var& b);
Var operator/(const Var& a, const Var& b);
Var operator-(const Var& a);

Var operator+(const Var& a, double b);
Var operator+(double a, const Var& b);
Var operator-(const Var& a, double b);
Var operator-(double a, const Var& b);
Var operator*(const Var& a, double b);
Var operator*(double a, const Var& b);
Var operator/(const Var& a, double b);
Var operator/(double a, const Var& b);

Var pow(const Var& x, double p);
Var exp(const Var& x);
Var log(const Var& x);
Var sqrt(const Var& x);
Var sin(const Var& x);
Var cos(const Var& x);
Var tan(const Var& x);
Var tanh(const Var& x);
Var abs(const Var& x);

} // namespace datamunge::autodiff
