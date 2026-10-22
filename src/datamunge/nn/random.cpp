#include "datamunge/nn/random.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

#include "detail.hpp"

namespace datamunge::nn {

using detail::dispatch_float;

namespace {

std::uint64_t splitmix64(std::uint64_t& x) {
  std::uint64_t z = (x += 0x9e3779b97f4a7c15ULL);
  z               = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z               = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}

std::uint64_t rotl(std::uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }

/// Fills a new float tensor with draws from `draw()`.
template <typename Draw>
Tensor fill_float(const Shape& shape, DType dtype, Draw&& draw) {
  Tensor out = empty(shape, dtype);
  dispatch_float(dtype, "random", [&](auto tag) {
    using T             = typename decltype(tag)::type;
    T*                p = out.data_ptr<T>();
    const std::int64_t n = out.numel();
    for (std::int64_t i = 0; i < n; ++i) p[i] = static_cast<T>(draw());
  });
  return out;
}

}  // namespace

Generator::Generator(std::uint64_t s) { seed(s); }

void Generator::seed(std::uint64_t s) {
  for (auto& w : state_) w = splitmix64(s);
  has_spare_ = false;
}

std::uint64_t Generator::next_u64() {
  const std::uint64_t result = rotl(state_[1] * 5, 7) * 9;
  const std::uint64_t t      = state_[1] << 17;
  state_[2] ^= state_[0];
  state_[3] ^= state_[1];
  state_[1] ^= state_[2];
  state_[0] ^= state_[3];
  state_[2] ^= t;
  state_[3] = rotl(state_[3], 45);
  return result;
}

double Generator::uniform01() { return static_cast<double>(next_u64() >> 11) * 0x1.0p-53; }

double Generator::normal() {
  if (has_spare_) {
    has_spare_ = false;
    return spare_;
  }
  double u1 = uniform01();
  while (u1 <= 0.0) u1 = uniform01();
  const double u2 = uniform01();
  const double r  = std::sqrt(-2.0 * std::log(u1));
  const double th = 2.0 * 3.14159265358979323846 * u2;
  spare_          = r * std::sin(th);
  has_spare_      = true;
  return r * std::cos(th);
}

std::uint64_t Generator::below(std::uint64_t n) {
  if (n == 0) throw std::invalid_argument("Generator::below: n must be positive");
  // Rejection sampling to avoid modulo bias.
  const std::uint64_t limit = UINT64_MAX - UINT64_MAX % n;
  std::uint64_t       x;
  do {
    x = next_u64();
  } while (x >= limit);
  return x % n;
}

Generator& default_generator() {
  static Generator gen(0x5eed);
  return gen;
}

void manual_seed(std::uint64_t seed) { default_generator().seed(seed); }

Tensor uniform(const Shape& shape, double low, double high, Generator& gen, DType dtype) {
  return fill_float(shape, dtype, [&] { return low + (high - low) * gen.uniform01(); });
}
Tensor uniform(const Shape& shape, double low, double high, DType dtype) {
  return uniform(shape, low, high, default_generator(), dtype);
}

Tensor normal(const Shape& shape, double mean, double stddev, Generator& gen, DType dtype) {
  return fill_float(shape, dtype, [&] { return mean + stddev * gen.normal(); });
}
Tensor normal(const Shape& shape, double mean, double stddev, DType dtype) {
  return normal(shape, mean, stddev, default_generator(), dtype);
}

Tensor truncated_normal(const Shape& shape, double mean, double stddev, Generator& gen, DType dtype) {
  return fill_float(shape, dtype, [&] {
    double z;
    do {
      z = gen.normal();
    } while (std::abs(z) > 2.0);
    return mean + stddev * z;
  });
}
Tensor truncated_normal(const Shape& shape, double mean, double stddev, DType dtype) {
  return truncated_normal(shape, mean, stddev, default_generator(), dtype);
}

Tensor randint(const Shape& shape, std::int64_t low, std::int64_t high, Generator& gen) {
  if (high <= low) throw std::invalid_argument("randint: need low < high");
  Tensor        out = empty(shape, DType::Int64);
  std::int64_t* p   = out.data_ptr<std::int64_t>();
  const auto    span = static_cast<std::uint64_t>(high - low);
  for (std::int64_t i = 0; i < out.numel(); ++i) p[i] = low + static_cast<std::int64_t>(gen.below(span));
  return out;
}
Tensor randint(const Shape& shape, std::int64_t low, std::int64_t high) {
  return randint(shape, low, high, default_generator());
}

Tensor bernoulli(const Tensor& p, Generator& gen) {
  const Tensor        probs = cast(p.detach(), to_floating(p.dtype()));
  const auto          v     = probs.to_vector<double>();
  std::vector<double> draws(v.size());
  for (std::size_t i = 0; i < v.size(); ++i) draws[i] = gen.uniform01() < v[i] ? 1.0 : 0.0;
  return cast(from_vector(draws, probs.shape().empty() ? Shape{1} : probs.shape()).reshape(probs.shape()),
              probs.dtype());
}
Tensor bernoulli(const Tensor& p) { return bernoulli(p, default_generator()); }

Tensor bernoulli(const Shape& shape, double p, Generator& gen, DType dtype) {
  return fill_float(shape, dtype, [&] { return gen.uniform01() < p ? 1.0 : 0.0; });
}
Tensor bernoulli(const Shape& shape, double p, DType dtype) { return bernoulli(shape, p, default_generator(), dtype); }

namespace {

/// Draws from unnormalised non-negative weights by inverse CDF.
std::int64_t draw_index(const std::vector<double>& weights, double total, Generator& gen) {
  double u = gen.uniform01() * total;
  for (std::size_t i = 0; i < weights.size(); ++i) {
    u -= weights[i];
    if (u < 0) return static_cast<std::int64_t>(i);
  }
  // Rounding: return the last class with positive weight.
  for (std::size_t i = weights.size(); i-- > 0;)
    if (weights[i] > 0) return static_cast<std::int64_t>(i);
  return 0;
}

}  // namespace

Tensor categorical(const Tensor& logits, std::int64_t num_samples, Generator& gen) {
  if (logits.dim() != 2) throw std::invalid_argument("categorical: logits must be [batch, classes]");
  const std::int64_t batch = logits.shape()[0], classes = logits.shape()[1];
  const auto         v     = logits.detach().to_vector<double>();
  Tensor             out   = empty({batch, num_samples}, DType::Int64);
  std::int64_t*      p     = out.data_ptr<std::int64_t>();
  std::vector<double> w(static_cast<std::size_t>(classes));
  for (std::int64_t b = 0; b < batch; ++b) {
    double m = -INFINITY;
    for (std::int64_t c = 0; c < classes; ++c) m = std::max(m, v[static_cast<std::size_t>(b * classes + c)]);
    double total = 0;
    for (std::int64_t c = 0; c < classes; ++c) {
      w[static_cast<std::size_t>(c)] = std::exp(v[static_cast<std::size_t>(b * classes + c)] - m);
      total += w[static_cast<std::size_t>(c)];
    }
    for (std::int64_t s = 0; s < num_samples; ++s) p[b * num_samples + s] = draw_index(w, total, gen);
  }
  return out;
}
Tensor categorical(const Tensor& logits, std::int64_t num_samples) {
  return categorical(logits, num_samples, default_generator());
}

Tensor multinomial(const Tensor& probs, std::int64_t num_samples, bool replacement, Generator& gen) {
  if (probs.dim() != 1 && probs.dim() != 2) throw std::invalid_argument("multinomial: probs must be 1-D or 2-D");
  const bool         one_d   = probs.dim() == 1;
  const std::int64_t batch   = one_d ? 1 : probs.shape()[0];
  const std::int64_t classes = probs.shape().back();
  if (!replacement && num_samples > classes)
    throw std::invalid_argument("multinomial: cannot draw more samples than classes without replacement");
  const auto    v   = probs.detach().to_vector<double>();
  Tensor        out = empty(one_d ? Shape{num_samples} : Shape{batch, num_samples}, DType::Int64);
  std::int64_t* p   = out.data_ptr<std::int64_t>();
  for (std::int64_t b = 0; b < batch; ++b) {
    std::vector<double> w(v.begin() + b * classes, v.begin() + (b + 1) * classes);
    for (double x : w)
      if (x < 0 || !std::isfinite(x)) throw std::invalid_argument("multinomial: weights must be finite and >= 0");
    for (std::int64_t s = 0; s < num_samples; ++s) {
      const double total = std::accumulate(w.begin(), w.end(), 0.0);
      if (total <= 0) throw std::invalid_argument("multinomial: weights sum to zero");
      const std::int64_t i     = draw_index(w, total, gen);
      p[b * num_samples + s]   = i;
      if (!replacement) w[static_cast<std::size_t>(i)] = 0;
    }
  }
  return out;
}
Tensor multinomial(const Tensor& probs, std::int64_t num_samples, bool replacement) {
  return multinomial(probs, num_samples, replacement, default_generator());
}

Tensor permutation(std::int64_t n, Generator& gen) {
  std::vector<std::int64_t> v(static_cast<std::size_t>(n));
  std::iota(v.begin(), v.end(), 0);
  for (std::int64_t i = n - 1; i > 0; --i)
    std::swap(v[static_cast<std::size_t>(i)], v[gen.below(static_cast<std::uint64_t>(i + 1))]);
  return from_vector(v);
}
Tensor permutation(std::int64_t n) { return permutation(n, default_generator()); }

Tensor shuffle(const Tensor& x, Generator& gen) {
  if (x.dim() < 1) throw std::invalid_argument("shuffle: need rank >= 1");
  return gather(x, permutation(x.shape()[0], gen), 0);
}

}  // namespace datamunge::nn
