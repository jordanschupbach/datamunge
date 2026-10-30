/// @file quality.cpp
/// @brief The pseudorandom-generator test battery, bit frequencies and bit periods.

#include <datamunge/algorithms/berlekamp_massey.hpp>
#include <datamunge/random/distributions.hpp>
#include <limits>
#include <datamunge/random/quality.hpp>
#include <datamunge/stats/chi_squared_test.hpp>
#include <datamunge/stats/ks_test.hpp>

#include <cmath>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace datamunge::random {

/// @brief Two-sided normal p-value helper.
namespace {

/// @brief P(|Z| > |z|) for a standard normal Z.
double two_sided(double z) { return 2.0 * (1.0 - normal_cdf(std::fabs(z))); }

} // namespace

std::vector<double> uniform_sample(RandomGenerator& generator, std::size_t n) {
  std::vector<double> u(n);
  for (auto& v : u) v = generator.uniform01();
  return u;
}

std::vector<double> bit_frequencies(RandomGenerator& generator, std::size_t n) {
  if (n == 0) throw std::invalid_argument("bit_frequencies: n must be positive");
  std::vector<double> ones(64, 0.0);
  for (std::size_t t = 0; t < n; ++t) {
    const std::uint64_t v = generator.next_u64();
    for (int b = 0; b < 64; ++b) ones[static_cast<std::size_t>(b)] += static_cast<double>((v >> b) & 1U);
  }
  for (auto& f : ones) f /= static_cast<double>(n);
  return ones;
}

dstruct::DataFrame bit_linear_complexities(const std::vector<std::uint64_t>& outputs,
                                           std::size_t training_size,
                                           const std::vector<int>& positions) {
  if (training_size < 2 || training_size >= outputs.size())
    throw std::invalid_argument("bit_linear_complexities: need training prefix >=2 and nonempty suffix");
  if (training_size > static_cast<std::size_t>(std::numeric_limits<int>::max()/2))
    throw std::length_error("bit_linear_complexities: training prefix exceeds solver's index limit");
  for (const int bit : positions)
    if (bit < 0 || bit > 63) throw std::invalid_argument("bit_linear_complexities: invalid bit position");
  std::vector<double> bits, train, heldout, lengths, errors, rates, frequencies;
  for (const int bit : positions) {
    std::vector<int> prefix(training_size);
    for (std::size_t t=0; t<training_size; ++t) prefix[t] = (outputs[t] >> bit) & 1U;
    const auto fit = algorithms::berlekamp_massey(prefix);
    std::size_t mistakes=0, ones=0;
    for (const auto word : outputs) ones += (word >> bit) & 1U;
    for (std::size_t t=training_size; t<outputs.size(); ++t) {
      int predicted=0;
      for (int j=1; j<=fit.length; ++j)
        predicted ^= fit.polynomial[j] & ((outputs[t-j] >> bit) & 1U);
      mistakes += predicted != ((outputs[t] >> bit) & 1U);
    }
    bits.push_back(bit); train.push_back(training_size); heldout.push_back(outputs.size()-training_size);
    lengths.push_back(fit.length); errors.push_back(mistakes);
    rates.push_back(static_cast<double>(mistakes)/(outputs.size()-training_size));
    frequencies.push_back(static_cast<double>(ones)/outputs.size());
  }
  dstruct::DataFrame out;
  out.add_column("bit",bits); out.add_column("training_size",train); out.add_column("heldout_size",heldout);
  out.add_column("linear_complexity",lengths); out.add_column("heldout_errors",errors);
  out.add_column("error_rate",rates); out.add_column("one_frequency",frequencies);
  return out;
}

std::vector<std::uint64_t> bit_periods_of(const std::vector<std::uint64_t>& outputs, int bits) {
  if (bits < 1 || bits > 64) throw std::invalid_argument("bit_periods: bits must be in [1, 64]");
  const std::size_t n = outputs.size();
  std::vector<std::uint64_t> periods(static_cast<std::size_t>(bits), 0U);
  for (int b = 0; b < bits; ++b) {
    for (std::size_t p = 1; p <= n / 2; p *= 2) {
      bool periodic = true;
      for (std::size_t t = 0; t + p < n && periodic; ++t)
        periodic = ((outputs[t] ^ outputs[t + p]) >> b & 1U) == 0U;
      if (periodic) {
        periods[static_cast<std::size_t>(b)] = p;
        break;
      }
    }
  }
  return periods;
}

dstruct::DataFrame quality_tests(RandomGenerator& generator, std::size_t n) {
  if (n < 1000) throw std::invalid_argument("quality_tests: n must be at least 1000");
  const double dn = static_cast<double>(n);
  std::vector<std::string> names;
  std::vector<double> statistic, df, p_value;
  const auto add = [&](std::string name, double stat, double dof, double p) {
    names.push_back(std::move(name));
    statistic.push_back(stat);
    df.push_back(dof);
    p_value.push_back(p);
  };

  // 1. Uniformity in 100 cells.
  std::vector<double> cells(100, 0.0);
  for (std::size_t t = 0; t < n; ++t) cells[static_cast<std::size_t>(generator.uniform01() * 100.0)] += 1.0;
  auto chi = stats::chi_squared_goodness_of_fit(cells);
  add("Uniformity, chi-square (100 cells)", chi.statistic, chi.parameter1, chi.p_value);

  // 2. Kolmogorov-Smirnov against Uniform(0, 1).
  const auto ks = stats::ks_test_one_sample_uniform(uniform_sample(generator, n));
  add("Uniformity, Kolmogorov-Smirnov", ks.statistic, 0.0, ks.p_value);

  // 3. Non-overlapping pairs in a 10 x 10 grid.
  std::vector<double> grid(100, 0.0);
  for (std::size_t t = 0; t < n; ++t) {
    const auto a = static_cast<std::size_t>(generator.uniform01() * 10.0);
    const auto b = static_cast<std::size_t>(generator.uniform01() * 10.0);
    grid[a * 10 + b] += 1.0;
  }
  chi = stats::chi_squared_goodness_of_fit(grid);
  add("Serial pairs, chi-square (10 x 10 cells)", chi.statistic, chi.parameter1, chi.p_value);

  // 4. Lag-1 sample correlation of successive uniforms.
  const auto u = uniform_sample(generator, n);
  double mean = 0.0;
  for (double v : u) mean += v;
  mean /= dn;
  double num = 0.0, den = 0.0;
  for (std::size_t t = 0; t < n; ++t) {
    den += (u[t] - mean) * (u[t] - mean);
    if (t + 1 < n) num += (u[t] - mean) * (u[t + 1] - mean);
  }
  const double z_corr = num / den * std::sqrt(dn);
  add("Lag-1 correlation, z = r sqrt(n)", z_corr, 0.0, two_sided(z_corr));

  // 5-6. Monobit tests on the lowest and highest bits; 7. low byte.
  double low = 0.0, high = 0.0;
  std::vector<double> bytes(256, 0.0);
  for (std::size_t t = 0; t < n; ++t) {
    const std::uint64_t v = generator.next_u64();
    low += static_cast<double>(v & 1U);
    high += static_cast<double>(v >> 63U);
  }
  for (std::size_t t = 0; t < n; ++t) bytes[generator.next_u64() & 0xFFU] += 1.0;
  const double z_low = (low - dn / 2) / std::sqrt(dn / 4), z_high = (high - dn / 2) / std::sqrt(dn / 4);
  add("Monobit, lowest bit (z)", z_low, 0.0, two_sided(z_low));
  add("Monobit, highest bit (z)", z_high, 0.0, two_sided(z_high));
  chi = stats::chi_squared_goodness_of_fit(bytes);
  add("Lowest byte, chi-square (256 cells)", chi.statistic, chi.parameter1, chi.p_value);

  using DF = dstruct::DataFrame;
  return DF{{"test", DF::string_column_type(names)},
            {"statistic", DF::numeric_column_type(statistic)},
            {"df", DF::numeric_column_type(df)},
            {"p_value", DF::numeric_column_type(p_value)}};
}

dstruct::DataFrame quality_tests_over_seeds(const GeneratorFactory& factory, std::size_t seeds,
                                            std::size_t n, std::uint64_t first_seed) {
  if (!factory || seeds < 2 || n < 1000)
    throw std::invalid_argument("quality_tests_over_seeds: invalid factory or counts");
  std::vector<std::string> names;
  std::vector<std::vector<double>> pvalues;
  for (std::size_t run=0; run<seeds; ++run) {
    auto generator=factory(first_seed+static_cast<std::uint64_t>(run));
    if (!generator) throw std::invalid_argument("quality_tests_over_seeds: null generator");
    const auto battery=quality_tests(*generator,n);
    if (run==0) { names=battery.get_column<std::string>("test").values(); pvalues.resize(names.size()); }
    for (std::size_t t=0; t<names.size(); ++t) {
      const double p=battery.double_at("p_value",t);
      if (!std::isfinite(p) || p<0 || p>1) throw std::runtime_error("quality_tests_over_seeds: invalid p-value");
      pvalues[t].push_back(p);
    }
  }
  std::vector<double> mean, small, large, statistic, p;
  for (const auto& values:pvalues) {
    mean.push_back(std::accumulate(values.begin(),values.end(),0.0)/seeds);
    double lo=0,hi=0;
    for (double value:values) { lo+=value<0.05; hi+=value>0.95; }
    small.push_back(lo);large.push_back(hi);
    const auto ks=stats::ks_test_one_sample_uniform(values);
    statistic.push_back(ks.statistic);p.push_back(ks.p_value);
  }
  dstruct::DataFrame out;
  out.add_column("test",names);out.add_column("seeds",std::vector<double>(names.size(),seeds));
  out.add_column("draws_per_test",std::vector<double>(names.size(),n));out.add_column("mean_p",mean);
  out.add_column("small_p_count",small);out.add_column("large_p_count",large);
  out.add_column("ks_statistic",statistic);out.add_column("ks_p_value",p);
  return out;
}

graphics::RPlot bit_plot(const std::vector<std::uint64_t>& outputs, int bits, graphics::RGB color) {
  if (bits < 1 || bits > 64) throw std::invalid_argument("bit_plot: bits must be in [1, 64]");
  if (outputs.empty()) throw std::invalid_argument("bit_plot: outputs must not be empty");
  const double n = static_cast<double>(outputs.size());
  auto plot = graphics::RPlot::create();
  plot.x_limits(0, n).y_limits(0, bits).show_grid(false).x_label("output index t").y_label("bit position");
  // One rectangle per maximal run of ones in each row keeps the SVG small.
  for (int b = 0; b < bits; ++b) {
    std::size_t t = 0;
    while (t < outputs.size()) {
      if (((outputs[t] >> b) & 1U) == 0U) {
        ++t;
        continue;
      }
      const std::size_t start = t;
      while (t < outputs.size() && ((outputs[t] >> b) & 1U) != 0U) ++t;
      const double x0 = static_cast<double>(start), x1 = static_cast<double>(t), y0 = b, y1 = b + 0.9;
      plot.polygon({x0, x1, x1, x0}, {y0, y0, y1, y1}, color, true);
    }
  }
  return plot;
}

} // namespace datamunge::random
