#pragma once

/// @file ziggurat.hpp
/// @brief Equal-area rejection sampling for a standard normal, with table and path diagnostics.
#include <datamunge/dstruct/dataframe.hpp>
#include <datamunge/plot/plot.hpp>

#include <cstddef>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

namespace datamunge::algorithms {

/// @brief Reusable 128- or 256-layer Ziggurat sampler for the standard normal distribution.
///
/// Tables use f(x)=exp(-x*x/2), zero-based strips from the tail to the peak,
/// and separate uniform draws for the horizontal coordinate and strip index.
/// Construction uses O(layers) storage/time. Sampling has expected O(1) cost
/// and unbounded rejection-loop worst case. Const methods share no mutable state;
/// callers must not share a mutable engine between concurrent calls.
/// Pseudorandom sequences depend on the standard library's uniform distributions.
///
/// @par Example
/// @code
/// const dm::algorithms::ZigguratNormal sampler(256);
/// std::mt19937_64 rng(42);
/// const double z = sampler.sample(rng);
/// @endcode
/// @par References
/// - G. Marsaglia and W. W. Tsang (2000), The Ziggurat Method for Generating Random Variables,
///   Journal of Statistical Software 5(8), https://doi.org/10.18637/jss.v005.i08
/// - J. A. Doornik (2005), An Improved Ziggurat Method to Generate Normal Random Samples,
///   https://doornik.com/research/ziggurat.pdf (separate index and coordinate draws).
/// @see draw(), table(), diagnostics()
class ZigguratNormal {
 public:
    /// @brief Precompute the normal envelope for a supported layer count.
    /// @param layers Number of strips, exactly 128 or 256 (default 256).
    /// @throws std::invalid_argument If layers is unsupported, before allocating tables.
    /// @par Example
    /// @code
    /// const dm::algorithms::ZigguratNormal sampler(128);
    /// const auto n = sampler.layers(); // 128
    /// @endcode
    /// @par References
    /// - G. Marsaglia and W. W. Tsang (2000), https://doi.org/10.18637/jss.v005.i08
    /// @see table(), tail_boundary()
    explicit ZigguratNormal(std::size_t layers = 256);

    /// @brief Number of envelope strips.
    /// @return 128 or 256, as passed to construction.
    [[nodiscard]] std::size_t layers() const noexcept { return n_; }

    /// @brief Start of the positive Gaussian tail, in standard deviations.
    /// @return r, approximately 3.443 (128 strips) or 3.654 (256 strips).
    [[nodiscard]] double tail_boundary() const noexcept { return x_[1]; }

    /// @brief Draw one N(0,1) variate, advancing the caller's engine.
    /// @param rng MT19937-64 engine, advanced by a variable number of draws.
    /// @return One standard-normal draw; tables remain unchanged.
    /// @par Example
    /// @code
    /// const dm::algorithms::ZigguratNormal sampler;
    /// std::mt19937_64 rng(2024);
    /// const double z = sampler.sample(rng);
    /// @endcode
    /// @par References
    /// - G. Marsaglia and W. W. Tsang (2000), https://doi.org/10.18637/jss.v005.i08
    /// @see draw(), diagnostics()
    double sample(std::mt19937_64& rng) const;

    /// @brief Draw an owned vector of independent normal variates, like R's rnorm.
    /// @param count Number of draws, including zero.
    /// @param seed Local MT19937-64 seed; default 2024.
    /// @param mean Normal location, finite; default 0.
    /// @param sd Normal standard deviation, finite and non-negative; default 1.
    /// @return count values mean + sd * Z, or constant mean when sd is zero.
    /// @throws std::invalid_argument If location or scale is invalid.
    /// @par Example
    /// @code
    /// const auto x = dm::algorithms::ZigguratNormal().draw(3, 42, 7, 0); // {7,7,7}
    /// @endcode
    /// @par References
    /// - G. Marsaglia and W. W. Tsang (2000), https://doi.org/10.18637/jss.v005.i08
    /// @see sample(), simulate_measurements()
    std::vector<double> draw(std::size_t count, std::uint64_t seed = 2024, double mean = 0, double sd = 1) const;

    /// @brief Inspect the envelope boundaries and exact per-proposal fast-path probabilities.
    ///
    /// For layer i>0 the rectangle has width outer=x[i] and height
    /// f(inner)-f(outer), with inner=x[i+1]. Row zero describes the base
    /// proposal with outer=V/f(r), inner=r, lower=0, upper=f(r).
    /// Its rectangle_area is the area of that proposal rectangle, equal to
    /// the combined area of the physical base rectangle and normal tail.
    /// @return Owned columns layer, outer, inner, lower, upper, rectangle_area,
    ///         fast_fraction; O(layers) time/storage.
    /// @par Example
    /// @code
    /// const auto table = dm::algorithms::ZigguratNormal().table();
    /// const auto rows = table.nrows(); // 256
    /// @endcode
    /// @par References
    /// - J. A. Doornik (2005), https://doornik.com/research/ziggurat.pdf, section 2.
    /// @see plot_envelope(), diagnostics()
    dstruct::DataFrame table() const;

    /// @brief Plot the positive normal kernel and selected envelope rectangles.
    /// @param stride Plot every stride-th non-base layer; positive, default 16.
    /// @return Owned plot; the base rectangle is shown at its physical width r.
    /// @throws std::invalid_argument If stride is zero.
    /// @par Example
    /// @code
    /// const auto plot = dm::algorithms::ZigguratNormal(128).plot_envelope(16);
    /// const bool svg = !plot.to_svg().empty(); // true
    /// @endcode
    /// @par References
    /// - J. A. Doornik (2005), https://doornik.com/research/ziggurat.pdf, figure 1.
    /// @see table()
    graphics::RPlot plot_envelope(std::size_t stride = 16) const;

    /// @brief Count actual proposal, wedge, tail and retry paths over a seeded run.
    ///
    /// Returns one row: layers, draws, proposals, fast, wedge, tail, rejected,
    /// tail_trials, fast_per_proposal, expected_fast_per_proposal and
    /// expected_acceptance. fast+wedge+tail==draws and proposals==draws+rejected.
    /// Tail rejection attempts are counted separately in tail_trials.
    /// @param count Number of returned samples, positive.
    /// @param seed Local engine seed, default 2024; same sequence as draw(count,seed).
    /// @return Owned counts and theoretical probabilities; O(count) expected time.
    /// @throws std::invalid_argument If count is zero.
    /// @par Example
    /// @code
    /// const auto d = dm::algorithms::ZigguratNormal().diagnostics(1000);
    /// const double n = d.double_at("draws", 0); // 1000
    /// @endcode
    /// @par References
    /// - J. A. Doornik (2005), https://doornik.com/research/ziggurat.pdf, section 2.
    /// @see sample(), table()
    dstruct::DataFrame diagnostics(std::size_t count, std::uint64_t seed = 2024) const;

    /// @brief Simulate independent Gaussian measurement replicates around tabulated estimates.
    ///
    /// For each row generate Y*=mean+standard_error*Z; this is a conditional
    /// measurement model, not a fitted hierarchical or posterior distribution.
    /// Retains all input columns and appends mc_mean, mc_sd (sample sd),
    /// mc_positive, normal_positive and mc_positive_se. The last is the
    /// theoretical Bernoulli Monte Carlo SE using the exact normal probability.
    /// O(rows*count) expected time and O(rows) output plus copied input storage.
    /// @param frame Input measurements, borrowed and unchanged.
    /// @param mean_column Numeric finite means, no nulls.
    /// @param se_column Numeric finite non-negative standard errors, no nulls.
    /// @param count Replicates per row, at least two; default 100000.
    /// @param seed Local engine seed; default 2024. Rows consume successive draws.
    /// @return An owned augmented frame. Zero scale gives deterministic replicates.
    /// @throws std::invalid_argument If data/count is invalid or an output column already exists.
    /// @throws std::out_of_range If either named column is absent.
    /// @par Example
    /// @code
    /// const auto schools = dm::datasets::eight_schools();
    /// const auto simulated = dm::algorithms::ZigguratNormal().simulate_measurements(
    ///     schools, "effect", "std_error", 1000);
    /// const auto rows = simulated.nrows(); // 8
    /// @endcode
    /// @par References
    /// - G. Marsaglia and W. W. Tsang (2000), https://doi.org/10.18637/jss.v005.i08
    /// @see draw(), diagnostics()
    dstruct::DataFrame simulate_measurements(const dstruct::DataFrame& frame, const std::string& mean_column,
                                            const std::string& se_column, std::size_t count = 100000,
                                            std::uint64_t seed = 2024) const;

 private:
    /// @brief Optional counters shared by the main and tail rejection loops.
    struct Counters {
        std::uint64_t proposals = 0; ///< Outer strip proposals.
        std::uint64_t fast = 0; ///< Inner-rectangle returns.
        std::uint64_t wedge = 0; ///< Curved-wedge returns.
        std::uint64_t tail = 0; ///< Tail returns.
        std::uint64_t rejected = 0; ///< Rejected outer strip proposals.
        std::uint64_t tail_trials = 0; ///< Tail proposal attempts, including successes.
    };
    /// @brief Main rejection loop, optionally collecting counts without changing its draw sequence.
    /// @param rng Mutable uniform source.
    /// @param counts Optional counters, or nullptr for no instrumentation.
    /// @return One standard normal draw.
    double sample_impl(std::mt19937_64& rng, Counters* counts) const;
    /// @brief Exponential rejection sampling conditional on |Z|>r.
    /// @param rng Mutable uniform source.
    /// @param negative Whether to negate the tail sample.
    /// @param counts Optional counters, or nullptr.
    /// @return Signed normal-tail sample, using log1p to avoid log(0).
    double tail(std::mt19937_64& rng, bool negative, Counters* counts) const;

    std::size_t n_; ///< Supported strip count.
    std::vector<double> x_; ///< Decreasing boundaries, x[0]=V/f(r), x[1]=r, x[n]=0.
    std::vector<double> ratio_; ///< x[i+1]/x[i], the per-strip fast-path threshold.
};
} // namespace datamunge::algorithms
