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

RootResult false_position(const std::function<double(double)>& f, double a, double b, double tol, int max_iter) {
    double     fa = f(a);
    double     fb = f(b);
    RootResult r;
    if (fa == 0.0) return RootResult{true, a, 0, 0.0};
    if (fb == 0.0) return RootResult{true, b, 0, 0.0};
    if (same_sign(fa, fb)) return r;

    int side = 0; // +1 if we last replaced b (kept a), -1 if we last replaced a (kept b)
    for (int i = 1; i <= max_iter; ++i) {
        const double c  = (fb * a - fa * b) / (fb - fa); // secant x-intercept
        const double fc = f(c);
        r.iterations    = i;
        r.root          = c;
        r.residual      = std::fabs(fc);
        if (std::fabs(fc) < tol) {
            r.converged = true;
            return r;
        }
        if (same_sign(fc, fb)) {           // c on the b side -> replace b, keep a
            b  = c;
            fb = fc;
            if (side == +1) fa *= 0.5;      // Illinois: a retained again, halve its weight
            side = +1;
        } else {                            // replace a, keep b
            a  = c;
            fa = fc;
            if (side == -1) fb *= 0.5;
            side = -1;
        }
    }
    return r;
}

RootResult halley(const std::function<double(double)>& f, const std::function<double(double)>& df,
                  const std::function<double(double)>& d2f, double x0, double tol, int max_iter) {
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
        const double d     = df(x);
        const double d2    = d2f(x);
        const double denom = 2.0 * d * d - fx * d2;
        if (denom == 0.0) return r;
        x = x - 2.0 * fx * d / denom;
    }
    r.root     = x;
    r.residual = std::fabs(f(x));
    if (r.residual < tol) r.converged = true;
    return r;
}

RootResult muller(const std::function<double(double)>& f, double x0, double x1, double x2, double tol,
                  int max_iter) {
    RootResult r;
    double     f0 = f(x0), f1 = f(x1), f2 = f(x2);
    for (int i = 1; i <= max_iter; ++i) {
        r.iterations = i;
        r.root       = x2;
        r.residual   = std::fabs(f2);
        if (std::fabs(f2) < tol) {
            r.converged = true;
            return r;
        }
        const double h0 = x1 - x0, h1 = x2 - x1;
        const double d0 = (f1 - f0) / h0, d1 = (f2 - f1) / h1;
        const double a  = (d1 - d0) / (h1 + h0); // parabola through the three points
        const double b  = a * h1 + d1;
        const double c  = f2;
        double       disc = b * b - 4.0 * a * c;
        if (disc < 0.0) disc = 0.0; // clamp to target real roots
        const double sq    = std::sqrt(disc);
        const double denom = (std::fabs(b + sq) > std::fabs(b - sq)) ? (b + sq) : (b - sq);
        if (denom == 0.0) return r;
        const double x3 = x2 - 2.0 * c / denom;
        x0 = x1; f0 = f1;
        x1 = x2; f1 = f2;
        x2 = x3; f2 = f(x3);
    }
    r.root     = x2;
    r.residual = std::fabs(f2);
    if (r.residual < tol) r.converged = true;
    return r;
}

RootResult itp(const std::function<double(double)>& f, double a, double b, double tol, int max_iter) {
    double     fa = f(a);
    double     fb = f(b);
    RootResult r;
    if (fa == 0.0) return RootResult{true, a, 0, 0.0};
    if (fb == 0.0) return RootResult{true, b, 0, 0.0};
    if (same_sign(fa, fb)) return r;

    const double eps   = tol;
    const double k1    = 0.2 / (b - a); // scale-dependent truncation parameter
    const double k2    = 2.0;
    const int    n0    = 1;
    const int    n_half = static_cast<int>(std::ceil(std::log2((b - a) / (2.0 * eps))));
    const int    n_max  = n_half + n0;

    for (int j = 0; j < max_iter && (b - a) > 2.0 * eps; ++j) {
        const double xh     = 0.5 * (a + b);                        // bisection midpoint
        const double r_proj = eps * std::pow(2.0, static_cast<double>(n_max - j)) - 0.5 * (b - a);
        const double xf     = (fb * a - fa * b) / (fb - fa);        // interpolation (regula falsi)
        const double sigma  = (xh - xf) >= 0.0 ? 1.0 : -1.0;
        const double delta  = k1 * std::pow(b - a, k2);
        const double xt     = (delta <= std::fabs(xh - xf)) ? xf + sigma * delta : xh; // truncation
        const double xitp   = (std::fabs(xt - xh) <= r_proj) ? xt : xh - sigma * r_proj; // projection
        const double fitp   = f(xitp);
        r.iterations        = j + 1;
        r.root              = xitp;
        r.residual          = std::fabs(fitp);
        if (std::fabs(fitp) < tol) {
            r.converged = true;
            return r;
        }
        if (same_sign(fitp, fa)) { a = xitp; fa = fitp; }
        else { b = xitp; fb = fitp; }
    }
    const double c = 0.5 * (a + b);
    r.root         = c;
    r.residual     = std::fabs(f(c));
    r.converged    = (b - a) <= 2.0 * eps || r.residual < tol;
    return r;
}

} // namespace datamunge::algorithms
