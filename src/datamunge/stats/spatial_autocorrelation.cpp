/// @file spatial_autocorrelation.cpp
/// @brief Neighbour weights, weight constants and Moran/Geary moments and permutation tests.
#include <datamunge/random/random.hpp>
#include <datamunge/stats/spatial_autocorrelation.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>

namespace datamunge::stats {
/// @brief Implementation helpers local to this translation unit.
namespace {
/// @brief Dense row-major matrix of doubles.
using Matrix = linalg::DenseMatrix<double>;

/// @brief Euclidean distance between two rows.
/// @param c Coordinates.
/// @param i First row.
/// @param j Second row.
/// @return The distance.
double distance(const Matrix& c, std::size_t i, std::size_t j) {
    double s = 0;
    for (std::size_t a = 0; a < c.cols(); ++a)
        s += (c(i, a) - c(j, a)) * (c(i, a) - c(j, a));
    return std::sqrt(s);
}

/// @brief Rejects nonfinite coordinates.
/// @param c Coordinates.
/// @param who Function name.
void check_coordinates(const Matrix& c, const char* who) {
    if (c.rows() == 0 || c.cols() == 0 || !std::ranges::all_of(c.data(), [](double v) { return std::isfinite(v); }))
        throw std::invalid_argument(std::string(who) + ": coordinates must be finite and nonempty");
}

/// @brief Weight constants of Cliff and Ord.
struct WeightConstants {
    double s0{}; ///< Sum of weights.
    double s1{}; ///< Half the sum of (w_ij + w_ji)^2.
    double s2{}; ///< Sum of (row sum + column sum)^2.
};

/// @brief Validates the inputs and returns the weight constants.
/// @param x Values.
/// @param w Weights.
/// @return S0, S1, S2.
WeightConstants validate(const std::vector<double>& x, const Matrix& w) {
    const std::size_t n = x.size();
    if (n < 4 || w.rows() != n || w.cols() != n)
        throw std::invalid_argument("spatial autocorrelation: need four or more values and an n-by-n weight matrix");
    if (!std::ranges::all_of(x, [](double v) { return std::isfinite(v); }) ||
        !std::ranges::all_of(w.data(), [](double v) { return std::isfinite(v); }))
        throw std::invalid_argument("spatial autocorrelation: values and weights must be finite");
    WeightConstants k;
    std::vector<double> rows(n, 0.0), cols(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        if (w(i, i) != 0)
            throw std::invalid_argument("spatial autocorrelation: the weight diagonal must be zero");
        for (std::size_t j = 0; j < n; ++j) {
            k.s0 += w(i, j);
            k.s1 += 0.5 * (w(i, j) + w(j, i)) * (w(i, j) + w(j, i));
            rows[i] += w(i, j);
            cols[j] += w(i, j);
        }
    }
    for (std::size_t i = 0; i < n; ++i)
        k.s2 += (rows[i] + cols[i]) * (rows[i] + cols[i]);
    if (!(k.s0 != 0))
        throw std::invalid_argument("spatial autocorrelation: weights sum to zero");
    return k;
}

/// @brief Moran's I of values (any permutation).
/// @param z Centred values.
/// @param w Weights.
/// @param s0 Weight sum.
/// @param zz Sum of squares.
/// @return I.
double moran_statistic(const std::vector<double>& z, const Matrix& w, double s0, double zz) {
    const std::size_t n = z.size();
    double s = 0;
    for (std::size_t i = 0; i < n; ++i) {
        double lag = 0;
        for (std::size_t j = 0; j < n; ++j)
            lag += w(i, j) * z[j];
        s += z[i] * lag;
    }
    return static_cast<double>(n) / s0 * s / zz;
}

/// @brief Geary's C of values (any permutation).
/// @param x Values.
/// @param w Weights.
/// @param s0 Weight sum.
/// @param zz Sum of squared deviations.
/// @return C.
double geary_statistic(const std::vector<double>& x, const Matrix& w, double s0, double zz) {
    const std::size_t n = x.size();
    double s = 0;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            if (w(i, j) != 0)
                s += w(i, j) * (x[i] - x[j]) * (x[i] - x[j]);
    return (static_cast<double>(n) - 1) * s / (2 * s0 * zz);
}

/// @brief Upper standard normal tail.
/// @param z Deviate.
/// @return P(Z > z).
double upper(double z) { return 0.5 * std::erfc(z / std::sqrt(2.0)); }

/// @brief Normal p-value for a deviate signed towards positive autocorrelation.
/// @param z Deviate.
/// @param alt Alternative.
/// @return The p-value.
double normal_p(double z, SpatialAlternative alt) {
    switch (alt) {
    case SpatialAlternative::positive:
        return upper(z);
    case SpatialAlternative::negative:
        return upper(-z);
    case SpatialAlternative::two_sided:
        return std::min(1.0, 2 * upper(std::abs(z)));
    }
    return upper(z);
}

/// @brief Fills the permutation fields of a test.
/// @param t Test (updated).
/// @param x Values.
/// @param o Options.
/// @param stat Statistic of a permuted vector.
/// @param positive_is_large True for Moran (large values mean positive autocorrelation).
template <typename Stat>
void permutation_test(SpatialAutocorrelationTest& t, const std::vector<double>& x,
                      const SpatialAutocorrelationOptions& o, Stat stat, bool positive_is_large) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    t.permutation_p_value = t.permutation_mean = t.permutation_sd = nan;
    if (o.permutations == 0)
        return;
    random::SplitMix64 rng(o.seed);
    std::vector<double> perm = x;
    std::size_t above = 0, below = 0;
    double mean = 0, sq = 0;
    const double tol = 1e-12 * std::max(1.0, std::abs(t.statistic));
    for (std::size_t s = 0; s < o.permutations; ++s) {
        for (std::size_t i = perm.size() - 1; i > 0; --i) {
            const auto j = static_cast<std::size_t>(rng.uniform01() * static_cast<double>(i + 1));
            std::swap(perm[i], perm[std::min(j, i)]);
        }
        const double v = stat(perm);
        mean += v;
        sq += v * v;
        above += v >= t.statistic - tol;
        below += v <= t.statistic + tol;
    }
    const double m = static_cast<double>(o.permutations);
    t.permutation_mean = mean / m;
    t.permutation_sd = o.permutations > 1 ? std::sqrt((sq - mean * mean / m) / (m - 1)) : nan;
    const double pos = (1.0 + static_cast<double>(positive_is_large ? above : below)) / (m + 1);
    const double neg = (1.0 + static_cast<double>(positive_is_large ? below : above)) / (m + 1);
    t.permutation_p_value = o.alternative == SpatialAlternative::positive   ? pos
                            : o.alternative == SpatialAlternative::negative ? neg
                                                                            : std::min(1.0, 2 * std::min(pos, neg));
}

/// @brief Centred values, sum of squares and kurtosis.
struct Moments {
    std::vector<double> z; ///< Centred values.
    double zz{};           ///< Sum of squares.
    double k{};            ///< n sum z^4 / (sum z^2)^2.
};

/// @brief Computes the moments used by both statistics.
/// @param x Values.
/// @return Centred values, sum of squares and kurtosis.
Moments moments(const std::vector<double>& x) {
    Moments m;
    const double mean = std::accumulate(x.begin(), x.end(), 0.0) / static_cast<double>(x.size());
    double z4 = 0;
    for (const double v : x) {
        m.z.push_back(v - mean);
        m.zz += (v - mean) * (v - mean);
        z4 += std::pow(v - mean, 4);
    }
    if (!(m.zz > 0))
        throw std::invalid_argument("spatial autocorrelation: values are constant");
    m.k = static_cast<double>(x.size()) * z4 / (m.zz * m.zz);
    return m;
}
} // namespace

linalg::DenseMatrix<double> knn_weights(const linalg::DenseMatrix<double>& coordinates, std::size_t k) {
    check_coordinates(coordinates, "knn_weights");
    const std::size_t n = coordinates.rows();
    if (k == 0 || k >= n)
        throw std::invalid_argument("knn_weights: k must be in [1, n)");
    Matrix w(n, n, 0.0);
    std::vector<std::size_t> order(n);
    std::vector<double> d(n);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j)
            d[j] = j == i ? std::numeric_limits<double>::infinity() : distance(coordinates, i, j);
        std::iota(order.begin(), order.end(), 0);
        std::ranges::stable_sort(order, [&](std::size_t a, std::size_t b) { return d[a] < d[b]; });
        for (std::size_t r = 0; r < k; ++r)
            w(i, order[r]) = 1;
    }
    return w;
}

linalg::DenseMatrix<double> distance_band_weights(const linalg::DenseMatrix<double>& coordinates, double lower, double upper) {
    check_coordinates(coordinates, "distance_band_weights");
    if (!(lower >= 0) || !(upper > lower) || !std::isfinite(upper))
        throw std::invalid_argument("distance_band_weights: need 0 <= lower < upper");
    const std::size_t n = coordinates.rows();
    Matrix w(n, n, 0.0);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = i + 1; j < n; ++j) {
            const double dij = distance(coordinates, i, j);
            if (dij > lower && dij <= upper)
                w(i, j) = w(j, i) = 1;
        }
    return w;
}

linalg::DenseMatrix<double> row_standardize(const linalg::DenseMatrix<double>& weights) {
    if (weights.rows() != weights.cols())
        throw std::invalid_argument("row_standardize: weights must be square");
    Matrix w = weights;
    for (std::size_t i = 0; i < w.rows(); ++i) {
        double s = 0;
        for (std::size_t j = 0; j < w.cols(); ++j) {
            if (w(i, j) < 0)
                throw std::invalid_argument("row_standardize: weights must be nonnegative");
            s += w(i, j);
        }
        if (s > 0)
            for (std::size_t j = 0; j < w.cols(); ++j)
                w(i, j) /= s;
    }
    return w;
}

SpatialAutocorrelationTest morans_i(const std::vector<double>& values,
                                    const linalg::DenseMatrix<double>& weights,
                                    const SpatialAutocorrelationOptions& options) {
    const WeightConstants c = validate(values, weights);
    const Moments m = moments(values);
    const double n = static_cast<double>(values.size()), s02 = c.s0 * c.s0;
    SpatialAutocorrelationTest t;
    t.statistic = moran_statistic(m.z, weights, c.s0, m.zz);
    t.expectation = -1 / (n - 1);
    if (options.randomisation) {
        const double a = n * (c.s1 * (n * n - 3 * n + 3) - n * c.s2 + 3 * s02);
        const double b = m.k * (c.s1 * (n * n - n) - 2 * n * c.s2 + 6 * s02);
        t.variance = (a - b) / ((n - 1) * (n - 2) * (n - 3) * s02) - t.expectation * t.expectation;
    } else {
        t.variance = (n * n * c.s1 - n * c.s2 + 3 * s02) / (s02 * (n * n - 1)) - t.expectation * t.expectation;
    }
    t.z = (t.statistic - t.expectation) / std::sqrt(t.variance);
    t.p_value = normal_p(t.z, options.alternative);
    permutation_test(
        t, values, options,
        [&](const std::vector<double>& p) {
            const double mean = std::accumulate(p.begin(), p.end(), 0.0) / n;
            std::vector<double> z(p.size());
            for (std::size_t i = 0; i < p.size(); ++i)
                z[i] = p[i] - mean;
            return moran_statistic(z, weights, c.s0, m.zz);
        },
        true);
    return t;
}

SpatialAutocorrelationTest gearys_c(const std::vector<double>& values,
                                    const linalg::DenseMatrix<double>& weights,
                                    const SpatialAutocorrelationOptions& options) {
    const WeightConstants c = validate(values, weights);
    const Moments m = moments(values);
    const double n = static_cast<double>(values.size()), s02 = c.s0 * c.s0, k = m.k;
    SpatialAutocorrelationTest t;
    t.statistic = geary_statistic(values, weights, c.s0, m.zz);
    t.expectation = 1;
    if (options.randomisation) {
        double v = (n - 1) * c.s1 * (n * n - 3 * n + 3 - k * (n - 1));
        v -= 0.25 * (n - 1) * c.s2 * (n * n + 3 * n - 6 - k * (n * n - n + 2));
        v += s02 * (n * n - 3 - k * (n - 1) * (n - 1));
        t.variance = v / (n * (n - 2) * (n - 3) * s02);
    } else {
        t.variance = ((2 * c.s1 + c.s2) * (n - 1) - 4 * s02) / (2 * (n + 1) * s02);
    }
    t.z = (t.expectation - t.statistic) / std::sqrt(t.variance);
    t.p_value = normal_p(t.z, options.alternative);
    permutation_test(
        t, values, options, [&](const std::vector<double>& p) { return geary_statistic(p, weights, c.s0, m.zz); },
        false);
    return t;
}

/// @brief Implementation helpers for the local statistics.
namespace {
/// @brief Checks the inputs of a local statistic.
/// @param x Values.
/// @param w Weights.
/// @param allow_diagonal Whether nonzero diagonal weights are allowed.
void validate_local(const std::vector<double>& x, const linalg::DenseMatrix<double>& w, bool allow_diagonal) {
    const std::size_t n = x.size();
    if (n < 4 || w.rows() != n || w.cols() != n)
        throw std::invalid_argument("local statistic: need four or more values and an n-by-n weight matrix");
    if (!std::ranges::all_of(x, [](double v) { return std::isfinite(v); }) ||
        !std::ranges::all_of(w.data(), [](double v) { return std::isfinite(v); }))
        throw std::invalid_argument("local statistic: values and weights must be finite");
    if (!allow_diagonal)
        for (std::size_t i = 0; i < n; ++i)
            if (w(i, i) != 0)
                throw std::invalid_argument("local statistic: the weight diagonal must be zero");
}

/// @brief Normal p-value for a deviate under an alternative (positive means large statistic).
/// @param z Deviate.
/// @param alt Alternative.
/// @return The p-value.
double local_p(double z, SpatialAlternative alt) { return normal_p(z, alt); }

/// @brief Conditional permutation summary of one location.
struct PermutationSummary {
    double p{};    ///< Permutation p-value.
    double mean{}; ///< Mean of the permuted statistic.
    double sd{};   ///< SD of the permuted statistic.
};

/// @brief Runs conditional permutations for location i.
/// @param i Location.
/// @param pool_values Values permuted into the neighbours (all locations).
/// @param w Weights.
/// @param r Number of permutations.
/// @param alt Alternative.
/// @param observed Observed statistic.
/// @param map Maps the permuted neighbour lag to the statistic.
/// @param rng Random source.
/// @return Permutation p-value and moments.
template <typename Map>
PermutationSummary conditional_permutation(std::size_t i, const std::vector<double>& pool_values,
                                           const linalg::DenseMatrix<double>& w, std::size_t r,
                                           SpatialAlternative alt, double observed, Map map,
                                           random::SplitMix64& rng) {
    const std::size_t n = pool_values.size();
    std::vector<double> neighbour_weights;
    for (std::size_t j = 0; j < n; ++j)
        if (j != i && w(i, j) != 0)
            neighbour_weights.push_back(w(i, j));
    std::vector<std::size_t> pool;
    for (std::size_t j = 0; j < n; ++j)
        if (j != i)
            pool.push_back(j);
    const std::size_t k = neighbour_weights.size();
    std::size_t ge = 0, le = 0;
    double sum = 0, sq = 0;
    const double tol = 1e-12 * std::max(1.0, std::abs(observed));
    for (std::size_t s = 0; s < r; ++s) {
        double lag = 0;
        for (std::size_t a = 0; a < k; ++a) {
            const std::size_t b =
                a + std::min(pool.size() - a - 1,
                             static_cast<std::size_t>(rng.uniform01() * static_cast<double>(pool.size() - a)));
            std::swap(pool[a], pool[b]);
            lag += neighbour_weights[a] * pool_values[pool[a]];
        }
        const double v = map(lag);
        sum += v;
        sq += v * v;
        ge += v >= observed - tol;
        le += v <= observed + tol;
    }
    const double m = static_cast<double>(r);
    PermutationSummary out;
    out.mean = sum / m;
    out.sd = r > 1 ? std::sqrt(std::max(0.0, (sq - sum * sum / m) / (m - 1))) : std::numeric_limits<double>::quiet_NaN();
    const double up = (1.0 + static_cast<double>(ge)) / (m + 1), down = (1.0 + static_cast<double>(le)) / (m + 1);
    out.p = alt == SpatialAlternative::positive   ? up
            : alt == SpatialAlternative::negative ? down
                                                  : (1.0 + static_cast<double>(std::min(ge, le))) / (m + 1);
    return out;
}

/// @brief Quadrant from a deviation and its spatial lag (zero counts as low, as spdep).
/// @param z Deviation.
/// @param lag Lag of deviations.
/// @return The quadrant.
LisaQuadrant quadrant_of(double z, double lag) {
    if (z > 0)
        return lag > 0 ? LisaQuadrant::high_high : LisaQuadrant::high_low;
    return lag > 0 ? LisaQuadrant::low_high : LisaQuadrant::low_low;
}
} // namespace

std::vector<LocalStatistic> local_morans_i(const std::vector<double>& values,
                                           const linalg::DenseMatrix<double>& weights,
                                           const LocalStatisticOptions& options) {
    validate_local(values, weights, false);
    const Moments mo = moments(values);
    const std::size_t n = values.size();
    const double nd = static_cast<double>(n), m2 = mo.zz / nd;
    double z4 = 0;
    for (const double v : mo.z)
        z4 += v * v * v * v;
    const double b2 = (z4 / nd) / (m2 * m2);
    const double a_coef = (nd - b2) / (nd - 1), b_coef = (2 * b2 - nd) / ((nd - 1) * (nd - 2));
    random::SplitMix64 rng(options.seed);
    const double nan = std::numeric_limits<double>::quiet_NaN();
    std::vector<LocalStatistic> out(n);
    for (std::size_t i = 0; i < n; ++i) {
        double lag = 0, wi = 0, wi2 = 0;
        for (std::size_t j = 0; j < n; ++j) {
            lag += weights(i, j) * mo.z[j];
            wi += weights(i, j);
            wi2 += weights(i, j) * weights(i, j);
        }
        const double zi = mo.z[i];
        LocalStatistic& s = out[i];
        s.statistic = zi / m2 * lag;
        if (options.conditional) {
            s.expectation = -(zi * zi * wi) / ((nd - 1) * m2);
            s.variance = (zi / m2) * (zi / m2) * (nd / (nd - 2)) * (wi2 - wi * wi / (nd - 1)) *
                         (m2 - zi * zi / (nd - 1));
        } else {
            s.expectation = -wi / (nd - 1);
            s.variance = a_coef * wi2 + b_coef * (wi * wi - wi2) - s.expectation * s.expectation;
        }
        s.z = (s.statistic - s.expectation) / std::sqrt(s.variance);
        s.p_value = local_p(s.z, options.alternative);
        s.quadrant = quadrant_of(zi, lag);
        s.permutation_p_value = s.permutation_mean = s.permutation_sd = nan;
        if (options.permutations > 0) {
            const auto ps = conditional_permutation(
                i, mo.z, weights, options.permutations, options.alternative, s.statistic,
                [&](double l) { return zi / m2 * l; }, rng);
            s.permutation_p_value = ps.p;
            s.permutation_mean = ps.mean;
            s.permutation_sd = ps.sd;
        }
    }
    return out;
}

std::vector<LocalStatistic> local_getis_ord(const std::vector<double>& values,
                                            const linalg::DenseMatrix<double>& weights, bool star,
                                            const LocalStatisticOptions& options) {
    validate_local(values, weights, star);
    const Moments mo = moments(values);
    const std::size_t n = values.size();
    const double nd = static_cast<double>(n);
    double total = 0, total2 = 0;
    for (const double v : values) {
        total += v;
        total2 += v * v;
    }
    random::SplitMix64 rng(options.seed);
    const double nan = std::numeric_limits<double>::quiet_NaN();
    std::vector<LocalStatistic> out(n);
    for (std::size_t i = 0; i < n; ++i) {
        double lag = 0, wi = 0, s1i = 0, zlag = 0;
        for (std::size_t j = 0; j < n; ++j) {
            if (!star && j == i)
                continue;
            lag += weights(i, j) * values[j];
            zlag += j == i ? 0.0 : weights(i, j) * mo.z[j];
            wi += weights(i, j);
            s1i += weights(i, j) * weights(i, j);
        }
        double xbar = 0, s2 = 0, denom = 0, vg = 0;
        if (star) {
            xbar = total / nd;
            s2 = mo.zz / nd;
            vg = s2 * (nd * s1i - wi * wi) / (nd - 1);
            denom = total;
        } else {
            xbar = (total - values[i]) / (nd - 1);
            s2 = (total2 - values[i] * values[i]) / (nd - 1) - xbar * xbar;
            vg = s2 * ((nd - 1) * s1i - wi * wi) / (nd - 2);
            denom = total - values[i];
        }
        const double eg = wi * xbar;
        LocalStatistic& s = out[i];
        s.statistic = lag / denom;
        s.expectation = eg / denom;
        s.variance = vg / (denom * denom);
        s.z = (lag - eg) / std::sqrt(vg);
        s.p_value = local_p(s.z, options.alternative);
        s.quadrant = quadrant_of(mo.z[i], zlag);
        s.permutation_p_value = s.permutation_mean = s.permutation_sd = nan;
        if (options.permutations > 0) {
            const double self = star ? weights(i, i) * values[i] : 0.0;
            const auto ps = conditional_permutation(
                i, values, weights, options.permutations, options.alternative, s.statistic,
                [&](double l) { return (self + l) / denom; }, rng);
            s.permutation_p_value = ps.p;
            s.permutation_mean = ps.mean;
            s.permutation_sd = ps.sd;
        }
    }
    return out;
}

} // namespace datamunge::stats
