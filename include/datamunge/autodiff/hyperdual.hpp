#pragma once

#include <cmath>

namespace datamunge::autodiff {

/// @brief A second-order forward-mode dual number carrying a value along with derivatives
///        along two independent seed directions and their exact mixed second partial --
///        i.e. x = x0 + eps1*dx1 + eps2*dx2 + eps1*eps2*dx12 with eps1^2 = eps2^2 = 0.
///        Seeding eps1 along e_i and eps2 along e_j and evaluating f yields
///        d^2f/dx_i dx_j exactly (to machine precision, unlike finite differences).
class HyperDual {
public:
    HyperDual() = default;
    explicit HyperDual(double value, double eps1 = 0.0, double eps2 = 0.0, double eps1eps2 = 0.0)
        : value_(value), eps1_(eps1), eps2_(eps2), eps1eps2_(eps1eps2) {}

    [[nodiscard]] double value() const noexcept { return value_; }
    [[nodiscard]] double eps1() const noexcept { return eps1_; }
    [[nodiscard]] double eps2() const noexcept { return eps2_; }
    [[nodiscard]] double eps1eps2() const noexcept { return eps1eps2_; }

    HyperDual operator+(const HyperDual& o) const {
        return HyperDual(value_ + o.value_, eps1_ + o.eps1_, eps2_ + o.eps2_, eps1eps2_ + o.eps1eps2_);
    }
    HyperDual operator-(const HyperDual& o) const {
        return HyperDual(value_ - o.value_, eps1_ - o.eps1_, eps2_ - o.eps2_, eps1eps2_ - o.eps1eps2_);
    }
    HyperDual operator-() const { return HyperDual(-value_, -eps1_, -eps2_, -eps1eps2_); }
    HyperDual operator*(const HyperDual& o) const {
        return HyperDual(value_ * o.value_, value_ * o.eps1_ + eps1_ * o.value_, value_ * o.eps2_ + eps2_ * o.value_,
                          value_ * o.eps1eps2_ + eps1_ * o.eps2_ + eps2_ * o.eps1_ + eps1eps2_ * o.value_);
    }
    HyperDual operator/(const HyperDual& o) const;

    HyperDual operator+(double s) const { return HyperDual(value_ + s, eps1_, eps2_, eps1eps2_); }
    HyperDual operator-(double s) const { return HyperDual(value_ - s, eps1_, eps2_, eps1eps2_); }
    HyperDual operator*(double s) const { return HyperDual(value_ * s, eps1_ * s, eps2_ * s, eps1eps2_ * s); }
    HyperDual operator/(double s) const { return HyperDual(value_ / s, eps1_ / s, eps2_ / s, eps1eps2_ / s); }

    friend HyperDual operator+(double s, const HyperDual& d) { return d + s; }
    friend HyperDual operator-(double s, const HyperDual& d) {
        return HyperDual(s - d.value_, -d.eps1_, -d.eps2_, -d.eps1eps2_);
    }
    friend HyperDual operator*(double s, const HyperDual& d) { return d * s; }
    friend HyperDual operator/(double s, const HyperDual& d) { return HyperDual(s) / d; }

private:
    double value_{0.0};
    double eps1_{0.0};
    double eps2_{0.0};
    double eps1eps2_{0.0};
};

/// @brief Composes a scalar function f, given f(u0), f'(u0), f''(u0), with a HyperDual
///        argument via the (second-order) chain rule -- the shared machinery behind every
///        unary math function below.
inline HyperDual hyperdual_chain(double f0, double f1, double f2, const HyperDual& x) {
    return HyperDual(f0, f1 * x.eps1(), f1 * x.eps2(), f2 * x.eps1() * x.eps2() + f1 * x.eps1eps2());
}

inline HyperDual reciprocal(const HyperDual& x) {
    const double v = x.value();
    return hyperdual_chain(1.0 / v, -1.0 / (v * v), 2.0 / (v * v * v), x);
}

inline HyperDual HyperDual::operator/(const HyperDual& o) const { return (*this) * reciprocal(o); }

inline HyperDual pow(const HyperDual& x, double p) {
    const double v = x.value();
    return hyperdual_chain(std::pow(v, p), p * std::pow(v, p - 1.0), p * (p - 1.0) * std::pow(v, p - 2.0), x);
}
inline HyperDual exp(const HyperDual& x) {
    const double v = std::exp(x.value());
    return hyperdual_chain(v, v, v, x);
}
inline HyperDual log(const HyperDual& x) {
    const double v = x.value();
    return hyperdual_chain(std::log(v), 1.0 / v, -1.0 / (v * v), x);
}
inline HyperDual sqrt(const HyperDual& x) {
    const double v = x.value();
    const double s = std::sqrt(v);
    return hyperdual_chain(s, 0.5 / s, -0.25 / (v * s), x);
}
inline HyperDual sin(const HyperDual& x) {
    const double v = x.value();
    return hyperdual_chain(std::sin(v), std::cos(v), -std::sin(v), x);
}
inline HyperDual cos(const HyperDual& x) {
    const double v = x.value();
    return hyperdual_chain(std::cos(v), -std::sin(v), -std::cos(v), x);
}
inline HyperDual tan(const HyperDual& x) {
    const double v = x.value();
    const double t = std::tan(v);
    const double sec2 = 1.0 + t * t;
    return hyperdual_chain(t, sec2, 2.0 * t * sec2, x);
}
inline HyperDual tanh(const HyperDual& x) {
    const double v = x.value();
    const double t = std::tanh(v);
    return hyperdual_chain(t, 1.0 - t * t, -2.0 * t * (1.0 - t * t), x);
}

} // namespace datamunge::autodiff
