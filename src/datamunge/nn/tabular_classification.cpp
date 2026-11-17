/// @file tabular_classification.cpp
/// @brief Native original tabular fixtures and tensor-only preprocessing/inference figures.
#include <datamunge/datasets/datasets.hpp>
#include <datamunge/nn/autograd.hpp>
#include <datamunge/nn/ops.hpp>
#include <datamunge/nn/tabular_classification.hpp>
#include <numbers>
#include <numeric>
#include <random>
#include <stdexcept>
namespace datamunge::nn::study {
/// @brief Private tensor admission and raw fixture construction.
namespace {
/// @brief Require finite floating nonempty matrices without changing tensor state.
/// @param x Caller tensor.
/// @throws std::invalid_argument If shape/dtype/values are invalid.
void matrix(const Tensor& x) {
  if (!x.defined() || x.dim() != 2 || x.size(0) < 1 || x.size(1) < 1 || !is_floating(x.dtype()) ||
      !all(is_finite(x)).item<bool>())
    throw std::invalid_argument("tabular: finite floating nonempty matrix required");
}
/// @brief Package finite generated host observations into owned native tensors.
/// @param x Flat interleaved two-feature coordinates.
/// @param y Class IDs in row order.
/// @param classes Ordered class display names.
/// @return Owned Float32 input and Int64 labels.
TabularClassificationData pack(
    const std::vector<double>&       x,
    const std::vector<std::int64_t>& y,
    std::vector<std::string>         classes) {
  return {
      from_vector(x, {std::int64_t(y.size()), 2}).to(DType::Float32),
      from_vector(y),
      std::move(classes),
      {"x", "y"}};
}
} // namespace
TabularClassificationData xor_classification_data(std::size_t per) {
  if (!per || per > 250000)
    throw std::invalid_argument("XOR: per corner must be1..250000");
  std::mt19937_64                  rng(1);
  std::normal_distribution<double> noise(0., .12);
  std::vector<double>              x;
  std::vector<std::int64_t>        y;
  for (auto c : std::vector<std::pair<int, int>>{{0, 0}, {0, 1}, {1, 0}, {1, 1}})
    for (std::size_t i = 0; i < per; ++i) {
      x.push_back(c.first + noise(rng));
      x.push_back(c.second + noise(rng));
      y.push_back(c.first ^ c.second);
    }
  return pack(x, y, {"A", "B"});
}
TabularClassificationData moon_classification_data(std::size_t per) {
  if (per < 2 || per > 250000)
    throw std::invalid_argument("moons: per moon must be2..250000");
  std::mt19937_64                  rng(3);
  std::normal_distribution<double> noise(0., .12);
  std::vector<double>              x;
  std::vector<std::int64_t>        y;
  for (std::size_t i = 0; i < per; ++i) {
    double t = std::numbers::pi * double(i) / double(per - 1);
    x.push_back(std::cos(t) + noise(rng));
    x.push_back(std::sin(t) + noise(rng));
    y.push_back(0);
    x.push_back(1. - std::cos(t) + noise(rng));
    x.push_back(.5 - std::sin(t) + noise(rng));
    y.push_back(1);
  }
  return pack(x, y, {"upper", "lower"});
}
TabularClassificationData iris_petal_classification_data() {
  auto                      d = datasets::iris();
  std::vector<double>       x;
  std::vector<std::int64_t> y;
  std::vector<std::string>  classes{"setosa", "versicolor", "virginica"};
  for (std::size_t i = 0; i < d.nrows(); ++i) {
    x.push_back(d.double_at("Petal.Length", i));
    x.push_back(d.double_at("Petal.Width", i));
    auto s  = d.string_at("Species", i);
    auto it = std::find(classes.begin(), classes.end(), s);
    if (it == classes.end())
      throw std::invalid_argument("Iris: unknown class");
    y.push_back(it - classes.begin());
  }
  auto out       = pack(x, y, classes);
  out.predictors = {"Petal.Length", "Petal.Width"};
  return out;
}
TabularClassificationSplit
tabular_classification_split(std::size_t rows, double f, std::uint64_t seed) {
  if (rows < 2 || !std::isfinite(f) || f <= 0 || f >= 1)
    throw std::invalid_argument("tabular split: invalid rows/fraction");
  if (rows > std::size_t(std::numeric_limits<std::int64_t>::max()) ||
      rows > std::vector<std::int64_t>{}.max_size())
    throw std::length_error("tabular split: row geometry");
  auto n = std::size_t(f * double(rows));
  if (!n || n >= rows)
    throw std::invalid_argument("tabular split: empty subset");
  std::vector<std::int64_t> ids(rows);
  std::iota(ids.begin(), ids.end(), 0);
  std::mt19937_64 rng(seed);
  std::shuffle(ids.begin(), ids.end(), rng);
  return {{ids.begin(), ids.begin() + n}, {ids.begin() + n, ids.end()}};
}
TabularStandardization fit_tabular_standardization(const Tensor& training) {
  matrix(training);
  NoGradGuard guard;
  auto        mu = mean(training, {0}, true), v = variance(training, {0}, true);
  auto        sd = sqrt(v);
  sd             = where(greater(sd, 0), sd, ones_like(sd));
  if (!all(is_finite(mu)).item<bool>() || !all(is_finite(sd)).item<bool>())
    throw std::overflow_error("tabular: nonfinite training statistics");
  return {mu.detach(), sd.detach()};
}
Tensor transform_tabular_features(const Tensor& input, const TabularStandardization& s) {
  matrix(input);
  matrix(s.mean);
  matrix(s.scale);
  if (s.mean.shape() != Shape{1, input.size(1)} || s.scale.shape() != s.mean.shape() ||
      s.mean.dtype() != input.dtype() || s.scale.dtype() != input.dtype() ||
      !all(greater(s.scale, 0)).item<bool>())
    throw std::invalid_argument("tabular: incompatible fitted statistics");
  auto out = (input - s.mean) / s.scale;
  if (!all(is_finite(out)).item<bool>())
    throw std::overflow_error("tabular: transformed values overflow");
  return out;
}
graphics::RPlot tabular_decision_regions(
    const Tensor&                               x,
    const Tensor&                               labels,
    const std::vector<std::string>&             classes,
    const std::function<Tensor(const Tensor&)>& logits,
    int                                         g) {
  matrix(x);
  if (x.size(1) != 2 || !labels.defined() || labels.shape() != Shape{x.size(0)} ||
      (labels.dtype() != DType::Int32 && labels.dtype() != DType::Int64) || classes.size() < 2 ||
      classes.size() > 3 || !logits || g < 2 || g > 201 || min(labels).item() < 0 ||
      max(labels).item() >= double(classes.size()) ||
      std::any_of(classes.begin(), classes.end(), [](const auto& s) { return s.empty(); }))
    throw std::invalid_argument("tabular plot: invalid complete inputs");
  NoGradGuard         guard;
  auto                lo = min(x, {0}).to_vector<double>(), hi = max(x, {0}).to_vector<double>();
  std::vector<double> axes[2];
  for (int k = 0; k < 2; ++k) {
    double pad = .1 * std::max(hi[k] - lo[k], 1.);
    double a = lo[k] - pad, b = hi[k] + pad;
    if (!std::isfinite(a) || !std::isfinite(b) || !std::isfinite(b - a))
      throw std::overflow_error("tabular plot: bounds overflow");
    axes[k] = linspace(a, b, g, DType::Float64).to_vector<double>();
  }
  std::vector<double> coords;
  for (double yy : axes[1])
    for (double xx : axes[0]) {
      coords.push_back(xx);
      coords.push_back(yy);
    }
  auto query = from_vector(coords, {g * g, 2}).to(x.dtype());
  auto z     = logits(query);
  if (!z.defined() || z.shape() != Shape{g * g, std::int64_t(classes.size())} ||
      z.dtype() != x.dtype() || !all(is_finite(z)).item<bool>())
    throw std::invalid_argument("tabular plot: invalid logits");
  auto                             ids = argmax(z, 1).to_vector<std::int64_t>();
  std::vector<std::vector<double>> cells(g, std::vector<double>(g));
  for (int i = 0; i < g; ++i)
    for (int j = 0; j < g; ++j)
      cells[j][i] = ids[i * g + j];
  std::vector<graphics::RGBA> colors{{37, 99, 235}, {220, 38, 38}, {22, 163, 74}},
      pale{{213, 226, 249}, {249, 218, 218}, {213, 237, 220}};
  pale.resize(classes.size());
  auto p = graphics::RPlot::create();
  p.image(axes[0], axes[1], cells, pale, -.5, double(classes.size()) - .5);
  auto raw   = x.to_vector<double>();
  auto truth = labels.to_vector<std::int64_t>();
  for (std::size_t c = 0; c < classes.size(); ++c) {
    std::vector<double> px, py;
    for (std::size_t i = 0; i < truth.size(); ++i)
      if (truth[i] == std::int64_t(c)) {
        px.push_back(raw[2 * i]);
        py.push_back(raw[2 * i + 1]);
      }
    p.points(px, py, classes[c], colors[c], 3);
  }
  return p;
}
} // namespace datamunge::nn::study
