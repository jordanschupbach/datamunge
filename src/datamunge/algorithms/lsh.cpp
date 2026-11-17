/// @file lsh.cpp
/// @brief Checked owned cosine index and shared actual signature/query producers.
#include <algorithm>
#include <cmath>
#include <datamunge/algorithms/lsh.hpp>
#include <numbers>
#include <random>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Private complete-input, finite-range and actual-capture helpers.
namespace {
/// @brief Validate all finite row entries and exact dimension before indexing.
/// @param x Complete borrowed row.
/// @param dimension Required coordinate count.
void validate_row(const std::vector<double>& x, std::size_t dimension) {
  if (x.size() != dimension)
    throw std::invalid_argument("LSH: dimension mismatch");
  for (double v : x)
    if (!std::isfinite(v))
      throw std::invalid_argument("LSH: nonfinite coordinate");
}
/// @brief Check a complete allocation product before vector construction.
/// @param a First count.
/// @param b Second count.
/// @return Exact representable product.
std::size_t product(std::size_t a, std::size_t b) {
  if (a && b > std::numeric_limits<std::size_t>::max() / a)
    throw std::length_error("LSH: geometry overflow");
  return a * b;
}
/// @brief Admit actual logical records before their optional capture copies.
/// @param trace Optional actual output.
/// @param count Actual next record count.
/// @param quota Optional total cap, zero disables it.
void admit(LSHQueryTrace* trace, std::size_t count, std::size_t quota) {
  if (!trace)
    return;
  if (count > std::numeric_limits<std::size_t>::max() - trace->captured_entries)
    throw std::length_error("LSH: capture overflow");
  if (quota && count > quota - std::min(quota, trace->captured_entries))
    throw std::length_error("LSH: capture quota");
  trace->captured_entries += count;
}
/// @brief Compute an actual plane dot, scaling query positively only after nonfinite raw arithmetic.
/// @param x Complete finite checked query.
/// @param h Actual finite stored Gaussian plane.
/// @param scaled Output indicating positive query scaling fallback.
/// @return Actual finite deciding sum.
double plane_dot(const std::vector<double>& x, const std::vector<double>& h, bool& scaled) {
  double dot = 0;
  for (std::size_t i = 0; i < x.size(); ++i)
    dot += h[i] * x[i];
  scaled = false;
  if (std::isfinite(dot))
    return dot;
  scaled       = true;
  double scale = 0;
  for (double v : x)
    scale = std::max(scale, std::abs(v));
  dot = 0;
  if (scale)
    for (std::size_t i = 0; i < x.size(); ++i)
      dot += h[i] * (x[i] / scale);
  if (!std::isfinite(dot))
    throw std::overflow_error("LSH: nonfinite scaled plane sum");
  return dot;
}
} // namespace
CosineLSH::CosineLSH(
    const std::vector<std::vector<double>>& data,
    std::size_t                             num_tables,
    std::size_t                             bits_per_table,
    std::uint64_t                           seed)
    : dim_(data.empty() ? 0 : data.front().size()), tables_(num_tables), bits_(bits_per_table) {
  if (bits_ > 64 || (!data.empty() && !dim_))
    throw std::invalid_argument("LSH: nonempty rows and width at most64 required");
  for (const auto& row : data)
    validate_row(row, dim_);
  product(data.size(), dim_);
  product(product(tables_, bits_), dim_);
  product(tables_, data.size());
  data_ = data;
  std::mt19937_64                  rng(seed);
  std::normal_distribution<double> gauss(0, 1);
  hyperplanes_.assign(tables_, {});
  for (std::size_t t = 0; t < tables_; ++t) {
    hyperplanes_[t].assign(bits_, std::vector<double>(dim_));
    for (auto& h : hyperplanes_[t])
      for (double& v : h) {
        v = gauss(rng);
        if (!std::isfinite(v))
          throw std::overflow_error("LSH: nonfinite generated plane");
      }
  }
  buckets_.assign(tables_, {});
  for (std::size_t i = 0; i < data_.size(); ++i)
    for (std::size_t t = 0; t < tables_; ++t)
      buckets_[t][hash(data_[i], t, nullptr, nullptr, 0)].push_back(i);
}
double CosineLSH::cosine(const std::vector<double>& a, const std::vector<double>& b) {
  validate_row(a, b.size());
  validate_row(b, a.size());
  double dot = 0, na = 0, nb = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  double denom = std::sqrt(na) * std::sqrt(nb);
  if (std::isfinite(dot) && std::isfinite(denom) && denom > 0)
    return dot / denom;
  double sa = 0, sb = 0;
  for (double v : a)
    sa = std::max(sa, std::abs(v));
  for (double v : b)
    sb = std::max(sb, std::abs(v));
  if (!sa || !sb)
    return 0;
  dot = na = nb = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double x = a[i] / sa, y = b[i] / sb;
    dot += x * y;
    na += x * x;
    nb += y * y;
  }
  const double result = dot / (std::sqrt(na) * std::sqrt(nb));
  if (!std::isfinite(result))
    throw std::overflow_error("LSH: nonfinite scaled cosine");
  return std::clamp(result, -1., 1.);
}
std::uint64_t CosineLSH::hash(
    const std::vector<double>& x,
    std::size_t                t,
    LSHQueryAnalysis*          work,
    LSHQueryTrace*             trace,
    std::size_t                quota) const {
  std::uint64_t sig = 0;
  for (std::size_t bit = 0; bit < bits_; ++bit) {
    bool   scaled;
    double dot = plane_dot(x, hyperplanes_[t][bit], scaled);
    if (dot >= 0)
      sig |= std::uint64_t{1} << bit;
    if (work)
      ++work->projections;
    if (trace) {
      admit(trace, 1, quota);
      trace->projections.push_back({t, bit, dot, dot >= 0, scaled});
    }
  }
  return sig;
}
std::uint64_t CosineLSH::signature(const std::vector<double>& x, std::size_t t) const {
  validate_row(x, dim_);
  if (t >= tables_)
    throw std::out_of_range("LSH: table outside index");
  return hash(x, t, nullptr, nullptr, 0);
}
LSHQueryAnalysis
CosineLSH::produce(const std::vector<double>& q, LSHQueryTrace* trace, std::size_t quota) const {
  validate_row(q, dim_);
  if (trace) {
    admit(trace, q.size(), quota);
    trace->query = q;
  }
  std::vector<char> seen(data_.size(), 0);
  LSHQueryAnalysis  out;
  out.seen_entries = seen.size();
  for (std::size_t t = 0; t < tables_; ++t) {
    const auto key = hash(q, t, &out, trace, quota);
    auto       it  = buckets_[t].find(key);
    if (trace) {
      admit(trace, 1, quota);
      if (it != buckets_[t].end())
        admit(trace, it->second.size(), quota);
      trace->buckets.push_back(
          {t, key, it == buckets_[t].end() ? std::vector<std::size_t>{} : it->second});
    }
    if (it == buckets_[t].end())
      continue;
    for (auto id : it->second) {
      ++out.bucket_visits;
      bool   duplicate  = seen[id] != 0;
      double similarity = 0;
      bool   record     = false;
      if (!duplicate) {
        seen[id] = 1;
        ++out.examined;
        similarity = cosine(q, data_[id]);
        record     = similarity > out.similarity;
        if (record) {
          out.similarity = similarity;
          out.best       = id;
        }
      }
      if (trace) {
        admit(trace, 1, quota);
        trace->candidates.push_back({t, id, duplicate, similarity, record});
      }
    }
  }
  return out;
}
std::size_t CosineLSH::query(const std::vector<double>& q, std::size_t* out_examined) const {
  auto out = produce(q, nullptr, 0);
  if (out_examined)
    *out_examined = out.examined;
  return out.best;
}
LSHQueryAnalysis CosineLSH::query_analysis(const std::vector<double>& q) const {
  return produce(q, nullptr, 0);
}
LSHQueryTrace CosineLSH::query_trace(const std::vector<double>& q, std::size_t max_entries) const {
  LSHQueryTrace out;
  out.analysis = produce(q, &out, max_entries);
  return out;
}
double lsh_collision_probability(double theta) {
  if (!std::isfinite(theta) || theta < 0 || theta > std::numbers::pi)
    throw std::invalid_argument("LSH: angle must be finite in[0,pi]");
  return 1 - theta / std::numbers::pi;
}
} // namespace datamunge::algorithms
