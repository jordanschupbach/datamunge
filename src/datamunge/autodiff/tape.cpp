#include <datamunge/autodiff/tape.hpp>

#include <cmath>
#include <stdexcept>

namespace datamunge::autodiff {

std::size_t Tape::size() const noexcept { return nodes_.size(); }

double Tape::value_at(const std::size_t index) const {
    if (index >= nodes_.size()) throw std::out_of_range("Tape::value_at: index out of range");
    return nodes_[index].value;
}

std::size_t Tape::new_leaf(const double value) {
    Node node;
    node.value = value;
    node.n_parents = 0;
    nodes_.push_back(node);
    return nodes_.size() - 1;
}

std::size_t Tape::record_unary(const double value, const std::size_t a, const double grad_a) {
    Node node;
    node.value = value;
    node.n_parents = 1;
    node.parents[0] = a;
    node.local_grads[0] = grad_a;
    nodes_.push_back(node);
    return nodes_.size() - 1;
}

std::size_t Tape::record_binary(const double value, const std::size_t a, const double grad_a, const std::size_t b,
                                 const double grad_b) {
    Node node;
    node.value = value;
    node.n_parents = 2;
    node.parents = {a, b};
    node.local_grads = {grad_a, grad_b};
    nodes_.push_back(node);
    return nodes_.size() - 1;
}

std::vector<double> Tape::backward(const std::size_t output) const {
    if (output >= nodes_.size()) throw std::out_of_range("Tape::backward: output index out of range");
    std::vector<double> adjoint(nodes_.size(), 0.0);
    adjoint[output] = 1.0;
    for (std::size_t i = nodes_.size(); i-- > 0;) {
        const Node& node = nodes_[i];
        for (std::size_t k = 0; k < node.n_parents; ++k) adjoint[node.parents[k]] += node.local_grads[k] * adjoint[i];
    }
    return adjoint;
}

namespace {
void check_same_tape(const Var& a, const Var& b) {
    if (&a.tape() != &b.tape())
        throw std::invalid_argument("autodiff::Var: operands belong to different Tapes");
}
} // namespace

Var::Var(Tape& tape, const double value) : tape_(&tape), index_(tape.new_leaf(value)), value_(value) {}

Var::Var(Tape& tape, const std::size_t index, const double value) noexcept
    : tape_(&tape), index_(index), value_(value) {}

Var operator+(const Var& a, const Var& b) {
    check_same_tape(a, b);
    const double v = a.value() + b.value();
    return Var(a.tape(), a.tape().record_binary(v, a.index(), 1.0, b.index(), 1.0), v);
}

Var operator-(const Var& a, const Var& b) {
    check_same_tape(a, b);
    const double v = a.value() - b.value();
    return Var(a.tape(), a.tape().record_binary(v, a.index(), 1.0, b.index(), -1.0), v);
}

Var operator*(const Var& a, const Var& b) {
    check_same_tape(a, b);
    const double v = a.value() * b.value();
    return Var(a.tape(), a.tape().record_binary(v, a.index(), b.value(), b.index(), a.value()), v);
}

Var operator/(const Var& a, const Var& b) {
    check_same_tape(a, b);
    const double v = a.value() / b.value();
    const double grad_a = 1.0 / b.value();
    const double grad_b = -a.value() / (b.value() * b.value());
    return Var(a.tape(), a.tape().record_binary(v, a.index(), grad_a, b.index(), grad_b), v);
}

Var operator-(const Var& a) { return Var(a.tape(), a.tape().record_unary(-a.value(), a.index(), -1.0), -a.value()); }

Var operator+(const Var& a, const double b) {
    const double v = a.value() + b;
    return Var(a.tape(), a.tape().record_unary(v, a.index(), 1.0), v);
}
Var operator+(const double a, const Var& b) { return b + a; }

Var operator-(const Var& a, const double b) {
    const double v = a.value() - b;
    return Var(a.tape(), a.tape().record_unary(v, a.index(), 1.0), v);
}
Var operator-(const double a, const Var& b) {
    const double v = a - b.value();
    return Var(b.tape(), b.tape().record_unary(v, b.index(), -1.0), v);
}

Var operator*(const Var& a, const double b) {
    const double v = a.value() * b;
    return Var(a.tape(), a.tape().record_unary(v, a.index(), b), v);
}
Var operator*(const double a, const Var& b) { return b * a; }

Var operator/(const Var& a, const double b) {
    const double v = a.value() / b;
    return Var(a.tape(), a.tape().record_unary(v, a.index(), 1.0 / b), v);
}
Var operator/(const double a, const Var& b) {
    const double v = a / b.value();
    const double grad = -a / (b.value() * b.value());
    return Var(b.tape(), b.tape().record_unary(v, b.index(), grad), v);
}

Var pow(const Var& x, const double p) {
    const double v = std::pow(x.value(), p);
    const double grad = p * std::pow(x.value(), p - 1.0);
    return Var(x.tape(), x.tape().record_unary(v, x.index(), grad), v);
}

Var exp(const Var& x) {
    const double v = std::exp(x.value());
    return Var(x.tape(), x.tape().record_unary(v, x.index(), v), v);
}

Var log(const Var& x) {
    const double v = std::log(x.value());
    return Var(x.tape(), x.tape().record_unary(v, x.index(), 1.0 / x.value()), v);
}

Var sqrt(const Var& x) {
    const double v = std::sqrt(x.value());
    return Var(x.tape(), x.tape().record_unary(v, x.index(), 1.0 / (2.0 * v)), v);
}

Var sin(const Var& x) {
    const double v = std::sin(x.value());
    return Var(x.tape(), x.tape().record_unary(v, x.index(), std::cos(x.value())), v);
}

Var cos(const Var& x) {
    const double v = std::cos(x.value());
    return Var(x.tape(), x.tape().record_unary(v, x.index(), -std::sin(x.value())), v);
}

Var tan(const Var& x) {
    const double v = std::tan(x.value());
    const double c = std::cos(x.value());
    return Var(x.tape(), x.tape().record_unary(v, x.index(), 1.0 / (c * c)), v);
}

Var tanh(const Var& x) {
    const double v = std::tanh(x.value());
    return Var(x.tape(), x.tape().record_unary(v, x.index(), 1.0 - v * v), v);
}

Var abs(const Var& x) {
    const double v = std::abs(x.value());
    return Var(x.tape(), x.tape().record_unary(v, x.index(), x.value() >= 0.0 ? 1.0 : -1.0), v);
}

} // namespace datamunge::autodiff
