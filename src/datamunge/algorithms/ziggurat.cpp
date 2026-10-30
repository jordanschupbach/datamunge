/// @file ziggurat.cpp
/// @brief Ziggurat table construction, sampling, path diagnostics and Gaussian measurement simulation.
#include <datamunge/algorithms/ziggurat.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace datamunge::algorithms {
ZigguratNormal::ZigguratNormal(std::size_t layers) : n_(layers) {
    if (layers != 128 && layers != 256) throw std::invalid_argument("ZigguratNormal: layers must be 128 or 256");
    x_.resize(n_ + 1); ratio_.resize(n_);
    const double r = layers == 128 ? 3.442619855899 : 3.6541528853610088;
    const double v = layers == 128 ? 9.91256303526217e-3 : 0.00492867323399;
    double f = std::exp(-0.5 * r * r);
    x_[0] = v / f; x_[1] = r; x_[n_] = 0;
    for (std::size_t i = 2; i < n_; ++i) {
        x_[i] = std::sqrt(-2 * std::log(v / x_[i - 1] + f));
        f = std::exp(-0.5 * x_[i] * x_[i]);
    }
    for (std::size_t i = 0; i < n_; ++i) ratio_[i] = x_[i + 1] / x_[i];
}

double ZigguratNormal::sample(std::mt19937_64& rng) const { return sample_impl(rng, nullptr); }

double ZigguratNormal::sample_impl(std::mt19937_64& rng, Counters* counts) const {
    std::uniform_real_distribution<double> unit(0, 1);
    std::uniform_int_distribution<std::size_t> layer(0, n_ - 1);
    for (;;) {
        if (counts) ++counts->proposals;
        const double u = 2 * unit(rng) - 1;
        const auto i = layer(rng);
        if (std::abs(u) < ratio_[i]) {
            if (counts) ++counts->fast;
            return u * x_[i];
        }
        if (i == 0) {
            if (counts) ++counts->tail;
            return tail(rng, u < 0, counts);
        }
        const double xr = u * x_[i];
        const double f0 = std::exp(-0.5 * (x_[i] * x_[i] - xr * xr));
        const double f1 = std::exp(-0.5 * (x_[i + 1] * x_[i + 1] - xr * xr));
        if (f1 + unit(rng) * (f0 - f1) < 1) {
            if (counts) ++counts->wedge;
            return xr;
        }
        if (counts) ++counts->rejected;
    }
}

double ZigguratNormal::tail(std::mt19937_64& rng, bool negative, Counters* counts) const {
    std::uniform_real_distribution<double> unit(0, 1);
    double e, y;
    do {
        if (counts) ++counts->tail_trials;
        e = -std::log1p(-unit(rng)) / x_[1];
        y = -std::log1p(-unit(rng));
    } while (2 * y < e * e);
    return negative ? -x_[1] - e : x_[1] + e;
}

std::vector<double> ZigguratNormal::draw(std::size_t count, std::uint64_t seed, double mean, double sd) const {
    if (!std::isfinite(mean) || !std::isfinite(sd) || sd < 0)
        throw std::invalid_argument("ZigguratNormal::draw: invalid mean or sd");
    std::mt19937_64 rng(seed);
    std::vector<double> values(count, mean);
    if (sd != 0) for (double& value : values) value += sd * sample(rng);
    return values;
}

dstruct::DataFrame ZigguratNormal::table() const {
    std::vector<double> layer, outer, inner, lower, upper, area;
    for (std::size_t i = 0; i < n_; ++i) {
        layer.push_back(static_cast<double>(i)); outer.push_back(x_[i]); inner.push_back(x_[i + 1]);
        lower.push_back(i == 0 ? 0 : std::exp(-0.5 * x_[i] * x_[i]));
        upper.push_back(std::exp(-0.5 * x_[i + 1] * x_[i + 1]));
        area.push_back(x_[i] * (upper.back() - lower.back()));
    }
    dstruct::DataFrame out;
    out.add_column("layer", layer); out.add_column("outer", outer); out.add_column("inner", inner);
    out.add_column("lower", lower); out.add_column("upper", upper);
    out.add_column("rectangle_area", area); out.add_column("fast_fraction", ratio_);
    return out;
}

graphics::RPlot ZigguratNormal::plot_envelope(std::size_t stride) const {
    if (stride == 0) throw std::invalid_argument("plot_envelope: stride must be positive");
    auto plot = graphics::RPlot::create();
    const auto data = table();
    plot.polygon({0, x_[1], x_[1], 0}, {0, 0, data.double_at("upper", 0), data.double_at("upper", 0)},
                 {180, 180, 180}, false);
    for (std::size_t i = 1; i < n_; ++i) {
        if ((i - 1) % stride != 0) continue;
        const double lo = data.double_at("lower", i), hi = data.double_at("upper", i);
        plot.polygon({0, x_[i], x_[i], 0}, {lo, lo, hi, hi}, {96, 165, 250}, false);
    }
    std::vector<double> x, y;
    for (int k = 0; k <= 500; ++k) { x.push_back(4.5 * k / 500); y.push_back(std::exp(-0.5 * x.back() * x.back())); }
    plot.line(x, y, "exp(-x²/2)", {220, 38, 38}, 2);
    plot.title("Selected equal-area Ziggurat proposal rectangles").x_label("positive standard-normal coordinate")
        .y_label("unnormalised density").size(900, 480);
    return plot;
}

dstruct::DataFrame ZigguratNormal::diagnostics(std::size_t count, std::uint64_t seed) const {
    if (count == 0) throw std::invalid_argument("diagnostics: count must be positive");
    Counters counts;
    std::mt19937_64 rng(seed);
    for (std::size_t i = 0; i < count; ++i) (void)sample_impl(rng, &counts);
    const double v = x_[0] * std::exp(-0.5 * x_[1] * x_[1]);
    dstruct::DataFrame out;
    out.add_column("layers", std::vector<double>{static_cast<double>(n_)});
    out.add_column("draws", std::vector<double>{static_cast<double>(count)});
    out.add_column("proposals", std::vector<double>{static_cast<double>(counts.proposals)});
    out.add_column("fast", std::vector<double>{static_cast<double>(counts.fast)});
    out.add_column("wedge", std::vector<double>{static_cast<double>(counts.wedge)});
    out.add_column("tail", std::vector<double>{static_cast<double>(counts.tail)});
    out.add_column("rejected", std::vector<double>{static_cast<double>(counts.rejected)});
    out.add_column("tail_trials", std::vector<double>{static_cast<double>(counts.tail_trials)});
    out.add_column("fast_per_proposal", std::vector<double>{static_cast<double>(counts.fast) / counts.proposals});
    out.add_column("expected_fast_per_proposal", std::vector<double>{std::accumulate(ratio_.begin(), ratio_.end(), 0.0) / n_});
    out.add_column("expected_acceptance", std::vector<double>{std::sqrt(std::acos(-1.0) / 2) / (n_ * v)});
    return out;
}

dstruct::DataFrame ZigguratNormal::simulate_measurements(const dstruct::DataFrame& frame, const std::string& mean_column,
                                                        const std::string& se_column, std::size_t count,
                                                        std::uint64_t seed) const {
    if (count < 2) throw std::invalid_argument("simulate_measurements: need at least two draws");
    for (const auto* name : {"mc_mean", "mc_sd", "mc_positive", "normal_positive", "mc_positive_se"})
        if (frame.has_column(name)) throw std::invalid_argument("simulate_measurements: output column exists");
    (void)frame.get_column(mean_column); (void)frame.get_column(se_column);
    std::vector<double> means, sds, positive, theoretical, errors;
    std::mt19937_64 rng(seed);
    for (std::size_t r = 0; r < frame.nrows(); ++r) {
        const double mu = frame.double_at(mean_column, r), sd = frame.double_at(se_column, r);
        if (!std::isfinite(mu) || !std::isfinite(sd) || sd < 0)
            throw std::invalid_argument("simulate_measurements: invalid mean or standard error");
        double mean = 0, m2 = 0;
        std::size_t npositive = 0;
        for (std::size_t j = 0; j < count; ++j) {
            const double value = sd == 0 ? mu : mu + sd * sample(rng);
            const double delta = value - mean;
            mean += delta / static_cast<double>(j + 1); m2 += delta * (value - mean);
            npositive += value > 0;
        }
        const double p = sd == 0 ? (mu > 0 ? 1 : 0) : 0.5 * std::erfc(-mu / (sd * std::sqrt(2.0)));
        means.push_back(mean); sds.push_back(std::sqrt(m2 / static_cast<double>(count - 1)));
        positive.push_back(static_cast<double>(npositive) / count); theoretical.push_back(p);
        errors.push_back(std::sqrt(p * (1 - p) / count));
    }
    auto out = frame;
    out.add_column("mc_mean", means); out.add_column("mc_sd", sds); out.add_column("mc_positive", positive);
    out.add_column("normal_positive", theoretical); out.add_column("mc_positive_se", errors);
    return out;
}
} // namespace datamunge::algorithms
