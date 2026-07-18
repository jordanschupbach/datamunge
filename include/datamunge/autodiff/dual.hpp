#pragma once

#include <cmath>

namespace datamunge::autodiff {

/// @brief A first-order forward-mode dual number: tracks a value together with its
///        derivative with respect to a single seed variable through every operation.
///        Evaluating a function on a Dual seeded with derivative=1 computes both the
///        function's value and its exact derivative in one forward pass.
class Dual {
public:
    Dual() = default;
    explicit Dual(double value, double derivative = 0.0) : value_(value), derivative_(derivative) {}

    [[nodiscard]] double value() const noexcept { return value_; }
    [[nodiscard]] double derivative() const noexcept { return derivative_; }

    Dual operator+(const Dual& o) const { return Dual(value_ + o.value_, derivative_ + o.derivative_); }
    Dual operator-(const Dual& o) const { return Dual(value_ - o.value_, derivative_ - o.derivative_); }
    Dual operator-() const { return Dual(-value_, -derivative_); }
    Dual operator*(const Dual& o) const {
        return Dual(value_ * o.value_, derivative_ * o.value_ + value_ * o.derivative_);
    }
    Dual operator/(const Dual& o) const {
        return Dual(value_ / o.value_, (derivative_ * o.value_ - value_ * o.derivative_) / (o.value_ * o.value_));
    }

    Dual operator+(double s) const { return Dual(value_ + s, derivative_); }
    Dual operator-(double s) const { return Dual(value_ - s, derivative_); }
    Dual operator*(double s) const { return Dual(value_ * s, derivative_ * s); }
    Dual operator/(double s) const { return Dual(value_ / s, derivative_ / s); }

    friend Dual operator+(double s, const Dual& d) { return d + s; }
    friend Dual operator-(double s, const Dual& d) { return Dual(s - d.value_, -d.derivative_); }
    friend Dual operator*(double s, const Dual& d) { return d * s; }
    friend Dual operator/(double s, const Dual& d) { return Dual(s, 0.0) / d; }

private:
    double value_{0.0};
    double derivative_{0.0};
};

inline Dual pow(const Dual& x, double p) {
    const double v = std::pow(x.value(), p);
    const double local = p * std::pow(x.value(), p - 1.0);
    return Dual(v, local * x.derivative());
}
inline Dual exp(const Dual& x) {
    const double v = std::exp(x.value());
    return Dual(v, v * x.derivative());
}
inline Dual log(const Dual& x) { return Dual(std::log(x.value()), x.derivative() / x.value()); }
inline Dual sqrt(const Dual& x) {
    const double v = std::sqrt(x.value());
    return Dual(v, x.derivative() / (2.0 * v));
}
inline Dual sin(const Dual& x) { return Dual(std::sin(x.value()), std::cos(x.value()) * x.derivative()); }
inline Dual cos(const Dual& x) { return Dual(std::cos(x.value()), -std::sin(x.value()) * x.derivative()); }
inline Dual tan(const Dual& x) {
    const double c = std::cos(x.value());
    return Dual(std::tan(x.value()), x.derivative() / (c * c));
}
inline Dual tanh(const Dual& x) {
    const double v = std::tanh(x.value());
    return Dual(v, (1.0 - v * v) * x.derivative());
}
inline Dual abs(const Dual& x) {
    return Dual(std::abs(x.value()), x.value() >= 0.0 ? x.derivative() : -x.derivative());
}

} // namespace datamunge::autodiff
