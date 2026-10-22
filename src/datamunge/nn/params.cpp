#include "datamunge/nn/params.hpp"

#include <cmath>
#include <stdexcept>

#include "datamunge/linalg/qr.hpp"
#include "datamunge/nn/autograd.hpp"
#include "datamunge/nn/io.hpp"
#include "datamunge/nn/ops.hpp"

namespace datamunge::nn {

// ---- initializers ---------------------------------------------------------------------

namespace init {

std::pair<double, double> fans(const Shape& shape) {
  if (shape.empty()) return {1.0, 1.0};
  if (shape.size() == 1) return {static_cast<double>(shape[0]), static_cast<double>(shape[0])};
  double receptive = 1;
  for (std::size_t i = 0; i + 2 < shape.size(); ++i) receptive *= static_cast<double>(shape[i]);
  return {receptive * static_cast<double>(shape[shape.size() - 2]), receptive * static_cast<double>(shape.back())};
}

Initializer zeros() {
  return [](const Shape& s, DType d, Generator&) { return nn::zeros(s, d); };
}
Initializer ones() {
  return [](const Shape& s, DType d, Generator&) { return nn::ones(s, d); };
}
Initializer constant(double value) {
  return [value](const Shape& s, DType d, Generator&) { return nn::full(s, value, d); };
}
Initializer uniform(double low, double high) {
  return [low, high](const Shape& s, DType d, Generator& g) { return nn::uniform(s, low, high, g, d); };
}
Initializer uniform(double limit) { return uniform(-limit, limit); }
Initializer normal(double stddev, double mean) {
  return [stddev, mean](const Shape& s, DType d, Generator& g) { return nn::normal(s, mean, stddev, g, d); };
}
Initializer truncated_normal(double stddev, double mean) {
  return [stddev, mean](const Shape& s, DType d, Generator& g) { return nn::truncated_normal(s, mean, stddev, g, d); };
}

namespace {
/// An initializer whose scale depends on the fans.
template <typename Scale>
Initializer scaled_uniform(Scale scale) {
  return [scale](const Shape& s, DType d, Generator& g) {
    const auto   f     = fans(s);
    const double limit = scale(f.first, f.second);
    return nn::uniform(s, -limit, limit, g, d);
  };
}
template <typename Scale>
Initializer scaled_normal(Scale scale) {
  return [scale](const Shape& s, DType d, Generator& g) {
    const auto f = fans(s);
    return nn::normal(s, 0.0, scale(f.first, f.second), g, d);
  };
}
}  // namespace

Initializer glorot_uniform() {
  return scaled_uniform([](double in, double out) { return std::sqrt(6.0 / (in + out)); });
}
Initializer glorot_normal() {
  return scaled_normal([](double in, double out) { return std::sqrt(2.0 / (in + out)); });
}
Initializer he_uniform() {
  return scaled_uniform([](double in, double) { return std::sqrt(6.0 / in); });
}
Initializer he_normal() {
  return scaled_normal([](double in, double) { return std::sqrt(2.0 / in); });
}
Initializer lecun_uniform() {
  return scaled_uniform([](double in, double) { return std::sqrt(3.0 / in); });
}
Initializer lecun_normal() {
  return scaled_normal([](double in, double) { return std::sqrt(1.0 / in); });
}

Initializer orthogonal(double gain) {
  return [gain](const Shape& s, DType d, Generator& g) {
    if (s.size() < 2) throw std::invalid_argument("orthogonal initializer needs rank >= 2, got " + shape_str(s));
    const std::int64_t cols = s.back();
    const std::int64_t rows = shape_numel(s) / std::max<std::int64_t>(cols, 1);
    // QR of the taller orientation, then keep the leading columns.
    const bool         tall = rows >= cols;
    const std::int64_t m = tall ? rows : cols, n = tall ? cols : rows;
    linalg::DenseMatrix<double> a(static_cast<std::size_t>(m), static_cast<std::size_t>(n));
    for (std::int64_t i = 0; i < m; ++i)
      for (std::int64_t j = 0; j < n; ++j) a(static_cast<std::size_t>(i), static_cast<std::size_t>(j)) = g.normal();
    const auto          qr = linalg::qr(a);
    std::vector<double> q(static_cast<std::size_t>(rows * cols));
    for (std::int64_t j = 0; j < n; ++j) {
      // Sign fix makes the result uniformly distributed (Mezzadri 2007).
      const double sign = qr.R(static_cast<std::size_t>(j), static_cast<std::size_t>(j)) < 0 ? -1.0 : 1.0;
      for (std::int64_t i = 0; i < m; ++i) {
        const double v = gain * sign * qr.Q(static_cast<std::size_t>(i), static_cast<std::size_t>(j));
        if (tall)
          q[static_cast<std::size_t>(i * cols + j)] = v;
        else
          q[static_cast<std::size_t>(j * cols + i)] = v;
      }
    }
    return cast(from_vector(q, s), d);
  };
}

}  // namespace init

// ---- ParameterStore ----------------------------------------------------------------------

ParameterStore::ParameterStore(std::uint64_t seed, DType dtype) : gen_(seed * 1000 + 1), dtype_(dtype) {
  if (!is_floating(dtype)) throw std::invalid_argument("ParameterStore: parameters must be float");
}

Tensor ParameterStore::get(const std::string& name, const Shape& shape, const Initializer& initializer) {
  auto it = index_.find(name);
  if (it != index_.end()) {
    const Tensor& existing = values_[it->second];
    if (existing.shape() != shape)
      throw std::invalid_argument("ParameterStore: parameter '" + name + "' has shape " + shape_str(existing.shape()) +
                                  ", requested " + shape_str(shape));
    return existing;
  }
  Tensor value;
  {
    NoGradGuard guard;
    value = cast(initializer(shape, dtype_, gen_), dtype_).detach().contiguous().clone();
  }
  if (value.shape() != shape)
    throw std::logic_error("ParameterStore: initializer for '" + name + "' returned shape " +
                           shape_str(value.shape()) + ", expected " + shape_str(shape));
  value.requires_grad_(true);
  index_[name] = values_.size();
  names_.push_back(name);
  values_.push_back(value);
  return value;
}

Tensor ParameterStore::at(const std::string& name) const {
  auto it = index_.find(name);
  if (it == index_.end()) throw std::out_of_range("ParameterStore: no parameter named '" + name + "'");
  return values_[it->second];
}

bool ParameterStore::contains(const std::string& name) const { return index_.count(name) > 0; }

void ParameterStore::set(const std::string& name, const Tensor& value) {
  auto it = index_.find(name);
  if (it == index_.end()) {
    get(name, value.shape(), [value](const Shape&, DType d, Generator&) { return cast(value.detach(), d); });
    return;
  }
  Tensor& existing = values_[it->second];
  if (existing.shape() != value.shape())
    throw std::invalid_argument("ParameterStore::set: '" + name + "' has shape " + shape_str(existing.shape()) +
                                ", value has " + shape_str(value.shape()));
  NoGradGuard guard;
  existing.copy_(value.detach());
}

std::vector<Tensor> ParameterStore::parameters() const { return values_; }

std::int64_t ParameterStore::count() const {
  std::int64_t n = 0;
  for (const auto& v : values_) n += v.numel();
  return n;
}

void ParameterStore::copy_from(const ParameterStore& src, double tau) {
  if (!(tau >= 0 && tau <= 1)) throw std::invalid_argument("copy_from: tau must be in [0, 1]");
  NoGradGuard guard;
  for (std::size_t i = 0; i < src.names_.size(); ++i) {
    const std::string& name = src.names_[i];
    auto               it   = index_.find(name);
    if (it == index_.end()) throw std::invalid_argument("copy_from: parameter '" + name + "' is missing here");
    Tensor&       mine   = values_[it->second];
    const Tensor& theirs = src.values_[i];
    if (mine.shape() != theirs.shape())
      throw std::invalid_argument("copy_from: parameter '" + name + "' shapes differ");
    if (tau == 1.0)
      mine.copy_(theirs.detach());
    else
      mine.copy_(add(mul(theirs.detach(), tau), mul(mine.detach(), 1.0 - tau)));
  }
}

ParameterStore ParameterStore::clone() const {
  ParameterStore out(0, dtype_);
  out.gen_ = gen_;
  for (std::size_t i = 0; i < names_.size(); ++i) out.set(names_[i], values_[i]);
  return out;
}

std::vector<double> ParameterStore::flatten() const {
  std::vector<double> out;
  out.reserve(static_cast<std::size_t>(count()));
  for (const auto& v : values_) {
    const auto x = v.to_vector<double>();
    out.insert(out.end(), x.begin(), x.end());
  }
  return out;
}

void ParameterStore::unflatten(const std::vector<double>& values) {
  if (static_cast<std::int64_t>(values.size()) != count())
    throw std::invalid_argument("unflatten: got " + std::to_string(values.size()) + " values for " +
                                std::to_string(count()) + " parameters");
  NoGradGuard guard;
  std::size_t at = 0;
  for (auto& v : values_) {
    const auto         n = static_cast<std::size_t>(v.numel());
    std::vector<double> part(values.begin() + static_cast<std::ptrdiff_t>(at),
                             values.begin() + static_cast<std::ptrdiff_t>(at + n));
    v.copy_(from_vector(part, v.shape().empty() ? Shape{1} : v.shape()).reshape(v.shape()));
    at += n;
  }
}

void ParameterStore::zero_grad() {
  for (auto& v : values_) v.zero_grad();
}

void ParameterStore::save(const std::string& path) const {
  NamedTensors arrays;
  for (std::size_t i = 0; i < names_.size(); ++i) arrays.emplace_back(names_[i], values_[i].detach());
  save_npz(path, arrays);
}

void ParameterStore::load(const std::string& path, bool strict) {
  const NamedTensors arrays = load_npz(path);
  if (strict) {
    if (arrays.size() != names_.size())
      throw std::invalid_argument("ParameterStore::load: '" + path + "' has " + std::to_string(arrays.size()) +
                                  " arrays, the store has " + std::to_string(names_.size()) + " parameters");
    for (const auto& [name, value] : arrays)
      if (!contains(name)) throw std::invalid_argument("ParameterStore::load: unexpected array '" + name + "'");
  }
  for (const auto& [name, value] : arrays) set(name, value);
}

}  // namespace datamunge::nn
