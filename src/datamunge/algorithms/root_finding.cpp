#include <datamunge/algorithms/root_finding.hpp>

#include <cmath>
#include <functional>

namespace datamunge::algorithms {

namespace {
bool same_sign(double x, double y) { return (x < 0.0) == (y < 0.0); }
} // namespace

RootResult bisection(const std::function<double(double)>& f, double a, double b, double tol, int max_iter) {
    double     fa = f(a);
    double     fb = f(b);
    RootResult r;
    if (fa == 0.0) return RootResult{true, a, 0, 0.0};
    if (fb == 0.0) return RootResult{true, b, 0, 0.0};
    if (same_sign(fa, fb)) return r; // no bracketed sign change

    for (int i = 1; i <= max_iter; ++i) {
        const double c  = 0.5 * (a + b);
        const double fc = f(c);
        r.iterations    = i;
        r.root          = c;
        r.residual      = std::fabs(fc);
        if (std::fabs(fc) < tol || 0.5 * (b - a) < tol) {
            r.converged = true;
            return r;
        }
        if (same_sign(fc, fa)) {
            a  = c;
            fa = fc;
        } else {
            b  = c;
            fb = fc;
        }
    }
    return r;
}

RootResult newton_raphson(const std::function<double(double)>& f, const std::function<double(double)>& df,
                          double x0, double tol, int max_iter) {
    RootResult r;
    double     x = x0;
    for (int i = 1; i <= max_iter; ++i) {
        const double fx = f(x);
        r.iterations    = i;
        r.root          = x;
        r.residual      = std::fabs(fx);
        if (std::fabs(fx) < tol) {
            r.converged = true;
            return r;
        }
        const double d = df(x);
        if (d == 0.0) return r; // flat tangent: cannot proceed
        x = x - fx / d;
    }
    r.root     = x;
    r.residual = std::fabs(f(x));
    if (r.residual < tol) r.converged = true;
    return r;
}

RootResult secant(const std::function<double(double)>& f, double x0, double x1, double tol, int max_iter) {
    RootResult r;
    double     f0 = f(x0);
    double     f1 = f(x1);
    for (int i = 1; i <= max_iter; ++i) {
        r.iterations = i;
        r.root       = x1;
        r.residual   = std::fabs(f1);
        if (std::fabs(f1) < tol) {
            r.converged = true;
            return r;
        }
        const double denom = f1 - f0;
        if (denom == 0.0) return r; // horizontal secant: cannot proceed
        const double x2 = x1 - f1 * (x1 - x0) / denom;
        x0              = x1;
        f0              = f1;
        x1              = x2;
        f1              = f(x2);
    }
    r.root     = x1;
    r.residual = std::fabs(f1);
    if (r.residual < tol) r.converged = true;
    return r;
}

RootResult ridders(const std::function<double(double)>& f, double a, double b, double tol, int max_iter) {
    double     fa = f(a);
    double     fb = f(b);
    RootResult r;
    if (fa == 0.0) return RootResult{true, a, 0, 0.0};
    if (fb == 0.0) return RootResult{true, b, 0, 0.0};
    if (same_sign(fa, fb)) return r;

    for (int i = 1; i <= max_iter; ++i) {
        const double c  = 0.5 * (a + b);
        const double fc = f(c);
        const double s  = std::sqrt(fc * fc - fa * fb);
        if (s == 0.0) return r;
        // Exponential correction toward the root.
        const double d  = c + (c - a) * ((fa >= fb ? 1.0 : -1.0) * fc / s);
        const double fd = f(d);
        r.iterations    = i;
        r.root          = d;
        r.residual      = std::fabs(fd);
        if (std::fabs(fd) < tol) {
            r.converged = true;
            return r;
        }
        // Re-bracket the root among a, c, d.
        if (!same_sign(fc, fd)) {
            a  = c;
            fa = fc;
            b  = d;
            fb = fd;
        } else if (!same_sign(fa, fd)) {
            b  = d;
            fb = fd;
        } else {
            a  = d;
            fa = fd;
        }
        if (std::fabs(b - a) < tol) {
            r.converged = true;
            return r;
        }
    }
    return r;
}

} // namespace datamunge::algorithms
