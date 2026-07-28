#include <datamunge/algorithms/ellipsoid.hpp>

#include <cmath>
#include <cstddef>
#include <vector>

namespace datamunge::algorithms {

EllipsoidResult ellipsoid_minimize(const std::function<double(const std::vector<double>&)>&              f,
                                   const std::function<std::vector<double>(const std::vector<double>&)>& subgradient,
                                   std::vector<double> center, double radius, int max_iter, double tol) {
    const std::size_t n = center.size();
    EllipsoidResult   result;
    result.x     = center;
    result.value = f(center);

    // P is the ellipsoid shape matrix: E = { x : (x-c)^T P^{-1} (x-c) <= 1 }, start P = radius^2 I.
    std::vector<std::vector<double>> P(n, std::vector<double>(n, 0.0));
    for (std::size_t i = 0; i < n; ++i) P[i][i] = radius * radius;

    // 1-D case: the update below divides by n^2-1; fall back to sign-driven bisection.
    if (n == 1) {
        double lo = center[0] - radius, hi = center[0] + radius;
        for (int it = 0; it < max_iter; ++it) {
            const double mid = 0.5 * (lo + hi);
            const std::vector<double> pt{mid};
            const double fv = f(pt);
            if (fv < result.value) { result.value = fv; result.x = pt; }
            const double g = subgradient(pt)[0];
            result.iterations = it + 1;
            if (hi - lo < tol) break;
            if (g > 0.0) hi = mid; else lo = mid; // minimizer is where subgradient changes sign
        }
        return result;
    }

    const double nd = static_cast<double>(n);
    for (int it = 0; it < max_iter; ++it) {
        result.iterations = it + 1;
        const std::vector<double> g = subgradient(center);

        // Pg = P * g
        std::vector<double> Pg(n, 0.0);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j) Pg[i] += P[i][j] * g[j];
        double gPg = 0.0;
        for (std::size_t i = 0; i < n; ++i) gPg += g[i] * Pg[i];
        if (gPg <= tol * tol) break; // subgradient (in the ellipsoid metric) negligible -> at optimum

        const double inv = 1.0 / std::sqrt(gPg);
        std::vector<double> Pg_hat(n);
        for (std::size_t i = 0; i < n; ++i) Pg_hat[i] = Pg[i] * inv; // P g / sqrt(g^T P g)

        // Centre update: c <- c - 1/(n+1) * Pg_hat.
        for (std::size_t i = 0; i < n; ++i) center[i] -= Pg_hat[i] / (nd + 1.0);

        // Shape update: P <- n^2/(n^2-1) * ( P - 2/(n+1) * Pg_hat Pg_hat^T ).
        const double scale = (nd * nd) / (nd * nd - 1.0);
        const double coef  = 2.0 / (nd + 1.0);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j)
                P[i][j] = scale * (P[i][j] - coef * Pg_hat[i] * Pg_hat[j]);

        const double fv = f(center);
        if (fv < result.value) {
            result.value = fv;
            result.x     = center;
        }
    }
    return result;
}

} // namespace datamunge::algorithms
