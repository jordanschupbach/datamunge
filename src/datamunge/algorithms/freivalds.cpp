/// @file freivalds.cpp
/// @brief One checked numerical kernel for ordinary verification and bounded diagnostics.
#include <algorithm>
#include <cmath>
#include <datamunge/algorithms/freivalds.hpp>
#include <datamunge/algorithms/linear_algebra.hpp>
#include <random>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Private validation, counting and projection implementation.
namespace freivalds_detail {
/// @brief Checked size_t sum.
/// @param a First count.
/// @param b Second count.
/// @return Sum.
/// @throws std::length_error Count overflow.
std::size_t add(std::size_t a, std::size_t b) {
  if (b > SIZE_MAX - a)
    throw std::length_error("Freivalds count overflow");
  return a + b;
}
/// @brief Checked size_t product.
/// @param a First count.
/// @param b Second count.
/// @return Product.
/// @throws std::length_error Count overflow.
std::size_t mul(std::size_t a, std::size_t b) {
  if (a && b > SIZE_MAX / a)
    throw std::length_error("Freivalds count overflow");
  return a * b;
}
/// @brief Validated shape and conservative input-entry count.
struct Shape {
  std::size_t m{};          ///< Left rows.
  std::size_t k{};          ///< Inner dimension.
  std::size_t n{};          ///< Right columns.
  std::size_t entries{};    ///< Actual input rows plus cells.
  bool        compatible{}; ///< All rectangular dimensions agree.
};
/// @brief Validate all shapes before all values and collect dimensions.
/// @param a Left input.
/// @param b Right input.
/// @param c Claimed product.
/// @return Validated metadata.
/// @throws std::invalid_argument Ragged or nonfinite input.
/// @throws std::length_error Input-entry count overflow.
Shape prepare(const FreivaldsMatrix& a, const FreivaldsMatrix& b, const FreivaldsMatrix& c) {
  Shape s{a.size(), a.empty() ? 0 : a[0].size(), b.empty() ? 0 : b[0].size()};
  for (const auto* matrix : {&a, &b, &c}) {
    s.entries        = add(s.entries, matrix->size());
    const auto width = matrix->empty() ? 0 : (*matrix)[0].size();
    for (const auto& row : *matrix) {
      if (row.size() != width)
        throw std::invalid_argument("Freivalds ragged matrix");
      s.entries = add(s.entries, row.size());
    }
  }
  for (const auto* matrix : {&a, &b, &c})
    for (const auto& row : *matrix)
      for (const auto x : row)
        if (!std::isfinite(x))
          throw std::invalid_argument("Freivalds requires finite entries");
  s.compatible = b.size() == s.k && c.size() == s.m && (s.m == 0 || c[0].size() == s.n);
  return s;
}
/// @brief Convert a signed round request to the performed upper bound.
/// @param rounds Requested rounds.
/// @return Nonnegative round count.
std::size_t positive(int rounds) {
  return rounds > 0 ? static_cast<std::size_t>(rounds) : 0;
}
/// @brief Check bounds for all counters before executing any batch.
/// @param s Validated shape.
/// @param rounds Maximum rounds per call.
/// @param calls Maximum compatible calls.
/// @return Nothing.
/// @throws std::length_error A counter or combined term count would overflow.
void bounds(const Shape& s, std::size_t rounds, std::size_t calls = 1) {
  if (!s.compatible)
    return;
  const auto r = mul(rounds, calls);
  mul(r, s.n);
  mul(r, s.m);
  const auto br = mul(s.k, s.n), abr = mul(s.m, s.k), cr = mul(s.m, s.n);
  mul(r, add(add(br, abr), cr));
}
/// @brief Forecast retained input and full requested frame entries.
/// @param s Validated shape.
/// @param rounds Requested rounds.
/// @return Conservative whole trace entries.
/// @throws std::length_error Counts or frame container capacity overflow.
std::size_t entries(const Shape& s, int rounds) {
  const auto t = s.compatible ? positive(rounds) : 0;
  if (t > std::vector<FreivaldsRound>().max_size())
    throw std::length_error("Freivalds frame capacity");
  bounds(s, t);
  return add(s.entries, mul(t, add(add(add(1, s.n), s.k), mul(3, s.m))));
}
/// @brief Enforce an optional whole-request count quota.
/// @param count Checked requested count.
/// @param cap Optional cap;0 means disabled.
/// @return Nothing.
/// @throws std::length_error Count exceeds cap.
void admit(std::size_t count, std::size_t cap) {
  if (cap && count > cap)
    throw std::length_error("Freivalds admission quota exceeded");
}
/// @brief Advance the legacy SplitMix64 state with intentional unsigned wrap.
/// @param s Mutable private RNG state.
/// @return Next64-bit word.
std::uint64_t splitmix(std::uint64_t& s) {
  std::uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
  z               = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z               = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}
/// @brief Reject a nonfinite actual accumulated value or residual.
/// @param x Value produced by the real computation.
/// @return Nothing.
/// @throws std::overflow_error Nonfinite arithmetic.
void finite(double x) {
  if (!std::isfinite(x))
    throw std::overflow_error("Freivalds nonfinite arithmetic");
}
/// @brief Execute the validated kernel once, optionally retaining actual frames.
/// @param a Left input.
/// @param b Right input.
/// @param c Claimed product.
/// @param s Validated metadata.
/// @param rounds Requested rounds.
/// @param seed Initial wrapping RNG state.
/// @param frames Optional output owned history.
/// @param supplied Optional one-round binary challenge; no RNG when supplied.
/// @return Actual verdict and work; incompatible input rejects without execution.
/// @throws std::overflow_error Actual nonfinite arithmetic.
/// @throws std::length_error Counter bounds overflow.
FreivaldsResult run(
    const FreivaldsMatrix&           a,
    const FreivaldsMatrix&           b,
    const FreivaldsMatrix&           c,
    const Shape&                     s,
    int                              rounds,
    std::uint64_t                    seed,
    std::vector<FreivaldsRound>*     frames   = nullptr,
    const std::vector<std::uint8_t>* supplied = nullptr) {
  bounds(s, positive(rounds));
  FreivaldsResult out;
  out.compatible     = s.compatible;
  auto& w            = out.work;
  w.rows             = s.m;
  w.inner            = s.k;
  w.columns          = s.n;
  w.requested_rounds = rounds;
  w.seeded           = !supplied;
  w.seed             = supplied ? 0 : seed;
  w.final_state      = w.seed;
  if (!s.compatible)
    return out;
  out.accepted        = true;
  w.kernel_calls      = 1;
  std::uint64_t state = w.seed;
  for (int r = 0; r < rounds; ++r) {
    const auto          before = state;
    std::vector<double> rv(s.n);
    for (std::size_t j = 0; j < s.n; ++j) {
      rv[j] = supplied ? static_cast<double>((*supplied)[j])
                       : static_cast<double>(splitmix(state) & 1ULL);
      if (supplied)
        ++w.provided_bits;
      else
        ++w.random_words;
    }
    w.final_state = state;
    std::vector<double> br(s.k, 0.0);
    for (std::size_t i = 0; i < s.k; ++i)
      for (std::size_t j = 0; j < s.n; ++j) {
        br[i] += b[i][j] * rv[j];
        ++w.br_terms;
        finite(br[i]);
      }
    std::vector<double> abr(s.m, 0.0), cr(s.m, 0.0);
    for (std::size_t i = 0; i < s.m; ++i) {
      for (std::size_t j = 0; j < s.k; ++j) {
        abr[i] += a[i][j] * br[j];
        ++w.abr_terms;
        finite(abr[i]);
      }
      for (std::size_t j = 0; j < s.n; ++j) {
        cr[i] += c[i][j] * rv[j];
        ++w.cr_terms;
        finite(cr[i]);
      }
    }
    FreivaldsRound f;
    f.iteration    = static_cast<std::size_t>(r);
    f.state_before = before;
    f.state_after  = state;
    for (std::size_t i = 0; i < s.m; ++i) {
      const double difference = std::fabs(abr[i] - cr[i]);
      finite(difference);
      ++w.compared_rows;
      if (frames)
        f.difference.push_back(difference);
      if (difference > kFreivaldsTolerance) {
        out.accepted     = false;
        out.failed_round = static_cast<std::size_t>(r);
        out.failed_row   = i;
        f.accepted       = false;
        f.failing_row    = i;
        break;
      }
    }
    ++w.rounds_completed;
    if (frames) {
      f.rv  = std::move(rv);
      f.br  = std::move(br);
      f.abr = std::move(abr);
      f.cr  = std::move(cr);
      frames->push_back(std::move(f));
    }
    if (!out.accepted)
      break;
  }
  return out;
}
/// @brief Add actual per-call counts into a preflight-bounded aggregate.
/// @param dst Aggregate to mutate.
/// @param src Actual call counts.
/// @return Nothing; preflight bounds establish representability.
void accumulate(FreivaldsCounts& dst, const FreivaldsCounts& src) {
  dst.kernel_calls += src.kernel_calls;
  dst.rounds_completed += src.rounds_completed;
  dst.random_words += src.random_words;
  dst.provided_bits += src.provided_bits;
  dst.br_terms += src.br_terms;
  dst.abr_terms += src.abr_terms;
  dst.cr_terms += src.cr_terms;
  dst.compared_rows += src.compared_rows;
}
/// @brief Copy admitted input and run one capturing kernel.
/// @param a Left input.
/// @param b Right input.
/// @param c Claimed product.
/// @param s Validated metadata.
/// @param rounds Requested rounds.
/// @param seed Initial state.
/// @param max_entries Optional entry quota.
/// @param supplied Optional supplied probe.
/// @return Owned full trace.
/// @throws std::length_error Quota/count bounds.
/// @throws std::overflow_error Actual nonfinite arithmetic.
FreivaldsTrace capture(
    const FreivaldsMatrix&           a,
    const FreivaldsMatrix&           b,
    const FreivaldsMatrix&           c,
    const Shape&                     s,
    int                              rounds,
    std::uint64_t                    seed,
    std::size_t                      max_entries,
    const std::vector<std::uint8_t>* supplied = nullptr) {
  admit(entries(s, rounds), max_entries);
  FreivaldsTrace out;
  out.a                              = a;
  out.b                              = b;
  out.c                              = c;
  static_cast<FreivaldsResult&>(out) = run(a, b, c, s, rounds, seed, &out.frames, supplied);
  return out;
}
} // namespace freivalds_detail

bool freivalds_verify(
    const FreivaldsMatrix& a,
    const FreivaldsMatrix& b,
    const FreivaldsMatrix& c,
    int                    rounds,
    std::uint64_t          seed) {
  return freivalds_verify_work(a, b, c, rounds, seed).accepted;
}
FreivaldsResult freivalds_verify_work(
    const FreivaldsMatrix& a,
    const FreivaldsMatrix& b,
    const FreivaldsMatrix& c,
    int                    rounds,
    std::uint64_t          seed) {
  const auto s = freivalds_detail::prepare(a, b, c);
  return freivalds_detail::run(a, b, c, s, rounds, seed);
}
std::size_t freivalds_trace_entries(
    const FreivaldsMatrix& a,
    const FreivaldsMatrix& b,
    const FreivaldsMatrix& c,
    int                    rounds) {
  return freivalds_detail::entries(freivalds_detail::prepare(a, b, c), rounds);
}
FreivaldsTrace freivalds_trace(
    const FreivaldsMatrix& a,
    const FreivaldsMatrix& b,
    const FreivaldsMatrix& c,
    int                    rounds,
    std::uint64_t          seed,
    std::size_t            max_entries) {
  return freivalds_detail::capture(
      a, b, c, freivalds_detail::prepare(a, b, c), rounds, seed, max_entries);
}
FreivaldsTrace freivalds_probe(
    const FreivaldsMatrix&           a,
    const FreivaldsMatrix&           b,
    const FreivaldsMatrix&           c,
    const std::vector<std::uint8_t>& challenge,
    std::size_t                      max_entries) {
  const auto s = freivalds_detail::prepare(a, b, c);
  if (challenge.size() != s.n)
    throw std::invalid_argument("Freivalds challenge length");
  for (const auto bit : challenge)
    if (bit > 1)
      throw std::invalid_argument("Freivalds challenge is not binary");
  return freivalds_detail::capture(a, b, c, s, 1, 0, max_entries, &challenge);
}
std::size_t freivalds_binary_vectors(std::size_t columns) {
  if (columns >= std::numeric_limits<std::size_t>::digits)
    throw std::length_error("Freivalds binary vector count overflow");
  return std::size_t(1) << columns;
}
FreivaldsBinaryAudit freivalds_binary_audit(
    const FreivaldsMatrix& a,
    const FreivaldsMatrix& b,
    const FreivaldsMatrix& c,
    std::size_t            max_vectors) {
  const auto           s = freivalds_detail::prepare(a, b, c);
  FreivaldsBinaryAudit out;
  out.compatible = s.compatible;
  if (!s.compatible)
    return out;
  out.vectors = freivalds_binary_vectors(s.n);
  freivalds_detail::bounds(s, 1, out.vectors);
  freivalds_detail::admit(out.vectors, max_vectors);
  std::vector<std::uint8_t> rv(s.n);
  for (std::size_t mask = 0; mask < out.vectors; ++mask) {
    for (std::size_t j = 0; j < s.n; ++j)
      rv[j] = static_cast<std::uint8_t>((mask >> j) & 1);
    const auto r = freivalds_detail::run(a, b, c, s, 1, 0, nullptr, &rv);
    if (r.accepted)
      ++out.accepted;
    else
      ++out.rejected;
    freivalds_detail::accumulate(out.work, r.work);
  }
  return out;
}
FreivaldsSeedSweep freivalds_seed_sweep(
    const FreivaldsMatrix&  a,
    const FreivaldsMatrix&  b,
    const FreivaldsMatrix&  c,
    const std::vector<int>& rounds,
    std::size_t             trials,
    std::uint64_t           first_seed) {
  const auto         s = freivalds_detail::prepare(a, b, c);
  FreivaldsSeedSweep out;
  out.compatible = s.compatible;
  if (trials && trials - 1 > UINT64_MAX - first_seed)
    throw std::length_error("Freivalds seed range wraps");
  if (rounds.size() > out.rows.max_size() || trials > out.seeds.max_size())
    throw std::length_error("Freivalds sweep capacity");
  int maximum = 0;
  for (const auto r : rounds)
    maximum = std::max(maximum, r);
  freivalds_detail::bounds(s, freivalds_detail::positive(maximum), trials);
  if (rounds.empty())
    return out;
  out.rows.reserve(rounds.size());
  out.seeds.reserve(trials);
  for (const auto r : rounds)
    out.rows.push_back({r, trials, 0});
  for (std::size_t i = 0; i < trials; ++i) {
    const auto seed = first_seed + static_cast<std::uint64_t>(i);
    const auto r    = freivalds_detail::run(a, b, c, s, maximum, seed);
    out.seeds.push_back({seed, r.accepted, r.failed_round, r.failed_row, r.work.rounds_completed});
    freivalds_detail::accumulate(out.work, r.work);
    for (auto& row : out.rows)
      if (s.compatible &&
          (row.rounds <= 0 || r.accepted || static_cast<std::size_t>(row.rounds) <= r.failed_round))
        ++row.accepted;
  }
  return out;
}
FreivaldsExampleMatrices freivalds_example_matrices() {
  FreivaldsExampleMatrices out;
  out.a.assign(40, std::vector<double>(40));
  out.b = out.a;
  std::mt19937                       rng(3);
  std::uniform_int_distribution<int> uniform(-9, 9);
  for (auto& row : out.a)
    for (auto& x : row)
      x = uniform(rng);
  for (auto& row : out.b)
    for (auto& x : row)
      x = uniform(rng);
  out.product = strassen_multiply(out.a, out.b);
  return out;
}
} // namespace datamunge::algorithms
