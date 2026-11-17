/// @file kalman_analysis.cpp
/// @brief Original full-family simulation through native filters and actual boundary-state capture.
#include <algorithm>
#include <cmath>
#include <datamunge/filter/kalman_analysis.hpp>
#include <limits>
#include <random>
#include <stdexcept>
namespace datamunge::filter {
/// @brief Fixed study model callbacks, complete validation and native-output observation helpers.
namespace study_detail {
/// @brief Owned rectangular linear callback, retaining the original report's matvec operation order.
class LinearFunction final : public VectorFunction {
  std::vector<std::vector<double>> matrix_; ///< Owned fixed callback coefficients.
 public:
  /// @brief Own a known complete rectangular fixed study matrix.
  /// @param matrix Actual fixed transition/observation coefficients.
  explicit LinearFunction(std::vector<std::vector<double>> matrix) : matrix_(std::move(matrix)) {}
  /// @brief Evaluate the actual original matrix-vector model callback.
  /// @param x Compatible native state vector, two coordinates in this study.
  /// @return Owned linear model output, with original row/column summation order.
  /// @throws std::invalid_argument On incompatible input shape.
  std::vector<double> evaluate(const std::vector<double>& x) override {
    std::vector<double> out(matrix_.size(), 0);
    for (std::size_t i = 0; i < matrix_.size(); ++i) {
      if (matrix_[i].size() != x.size())
        throw std::invalid_argument("study callback shape");
      for (std::size_t j = 0; j < x.size(); ++j)
        out[i] += matrix_[i][j] * x[j];
    }
    return out;
  }
};
/// @brief Original monotonic nonlinear scalar sensor2*position+sin(position).
class MonotonicSensor final : public VectorFunction {
 public:
  /// @brief Evaluate the original sensor using position coordinate0.
  /// @param x Compatible two-coordinate native state.
  /// @return Owned single sensor observation.
  /// @throws std::invalid_argument On wrong state dimension.
  std::vector<double> evaluate(const std::vector<double>& x) override {
    if (x.size() != 2)
      throw std::invalid_argument("study sensor shape");
    return {2 * x[0] + std::sin(x[0])};
  }
};
/// @brief Validate a complete nonempty matched finite scalar observation/truth sequence.
/// @param z Borrowed nested scalar observations.
/// @param truth Borrowed matched true positions.
/// @throws std::invalid_argument On shape/finite violations.
void validate(const std::vector<std::vector<double>>& z, const std::vector<double>& truth) {
  if (z.empty() || z.size() != truth.size())
    throw std::invalid_argument("study sequence shape");
  for (std::size_t i = 0; i < z.size(); ++i)
    if (z[i].size() != 1 || !std::isfinite(z[i][0]) || !std::isfinite(truth[i]))
      throw std::invalid_argument("study finite scalar sequence");
}
/// @brief Validate the actual native sigma scale using the same two-state floating formula.
/// @param alpha Actual UKF alpha.
/// @throws std::invalid_argument On nonpositive/nonfinite alpha or nonpositive/nonfinite2+lambda.
void sigma(double alpha) {
  const double lambda = alpha * alpha * 2 - 2, scale = 2 + lambda;
  if (!std::isfinite(alpha) || alpha <= 0 || !std::isfinite(scale) || scale <= 0)
    throw std::invalid_argument("study representable sigma scale");
}
/// @brief Observe all actual captured states/covariances for finite output, without rerunning a filter.
/// @param states Borrowed producing trajectory.
/// @throws std::overflow_error On any nonfinite actual state/covariance coordinate.
void finite(const std::vector<KalmanState>& states) {
  for (const auto& s : states) {
    for (auto x : s.x)
      if (!std::isfinite(x))
        throw std::overflow_error("study nonfinite state");
    for (const auto& row : s.P)
      for (auto x : row)
        if (!std::isfinite(x))
          throw std::overflow_error("study nonfinite covariance");
  }
}
} // namespace study_detail
KalmanStage kalman_capture_step(KalmanFilter& k, const std::vector<double>& z) {
  KalmanStage out;
  out.prior = {k.state(), k.covariance()};
  k.predict();
  out.predicted = {k.state(), k.covariance()};
  k.update(z);
  out.filtered = {k.state(), k.covariance()};
  return out;
}
KalmanStage kalman_tiny_step() {
  KalmanFilter k({{1}}, {{1}}, {{1}}, {{3}}, {0}, {{4}});
  return kalman_capture_step(k, {2});
}
KalmanStudyData kalman_demo_data() {
  KalmanStudyData                  d;
  std::mt19937                     rng(42);
  std::normal_distribution<double> noise(0, 2);
  double                           position = 0;
  for (int i = 0; i < 50; ++i) {
    position += 2;
    d.linear_truth.push_back(position);
    d.linear_measurements.push_back({position + noise(rng)});
  }
  std::normal_distribution<double> small(0, .5);
  position = 0;
  for (int i = 0; i < 50; ++i) {
    position += 2;
    d.nonlinear_truth.push_back(position);
    d.nonlinear_measurements.push_back({2 * position + std::sin(position) + small(rng)});
  }
  double velocity = 0;
  position        = 0;
  for (int i = 0; i < 40; ++i) {
    position += velocity + .5 * .5;
    velocity += .5;
    d.acceleration_truth.push_back(position);
    d.acceleration_measurements.push_back(position);
  }
  return d;
}
std::vector<KalmanState> kalman_demo_ukf(const KalmanStudyData& d, double alpha, std::size_t cap) {
  study_detail::validate(d.nonlinear_measurements, d.nonlinear_truth);
  study_detail::sigma(alpha);
  if (cap && d.nonlinear_measurements.size() > cap)
    throw std::length_error("study complete UKF steps");
  study_detail::LinearFunction  f({{1, 1}, {0, 1}});
  study_detail::MonotonicSensor h;
  UnscentedKalmanFilter         ukf(
      f, h, {{.01, 0}, {0, .01}}, {{.25}}, {0, 0}, {{100, 0}, {0, 100}}, alpha);
  auto states = ukf.filter(d.nonlinear_measurements);
  study_detail::finite(states);
  return states;
}
KalmanFamilyStudy kalman_family_study(const KalmanStudyData& d, KalmanStudyOptions o) {
  study_detail::validate(d.linear_measurements, d.linear_truth);
  study_detail::validate(d.nonlinear_measurements, d.nonlinear_truth);
  if (d.acceleration_measurements.empty() ||
      d.acceleration_measurements.size() != d.acceleration_truth.size())
    throw std::invalid_argument("study acceleration shape");
  for (std::size_t i = 0; i < d.acceleration_truth.size(); ++i)
    if (!std::isfinite(d.acceleration_measurements[i]) || !std::isfinite(d.acceleration_truth[i]))
      throw std::invalid_argument("study acceleration finite");
  study_detail::sigma(o.ukf_alpha);
  if (o.ensemble_size < 2 || o.particle_count < 2)
    throw std::invalid_argument("study populations");
  const auto max = std::numeric_limits<std::size_t>::max();
  if (d.nonlinear_truth.size() > max - d.linear_truth.size() ||
      d.acceleration_truth.size() > max - d.linear_truth.size() - d.nonlinear_truth.size())
    throw std::length_error("study complete step overflow");
  const auto steps = d.linear_truth.size() + d.nonlinear_truth.size() + d.acceleration_truth.size();
  const auto population = std::size_t(o.ensemble_size) + std::size_t(o.particle_count);
  if ((o.max_input_steps && steps > o.max_input_steps) ||
      (o.max_population && population > o.max_population))
    throw std::length_error("study complete admission");
  KalmanFamilyStudy out;
  out.data                                 = d;
  const std::vector<std::vector<double>> F = {{1, 1}, {0, 1}}, H = {{1, 0}},
                                         Q = {{.01, 0}, {0, .01}}, R = {{4}}, Rn = {{.25}},
                                         P0 = {{100, 0}, {0, 100}};
  const std::vector<double> x0              = {0, 0};
  KalmanFilter              k(F, H, Q, R, x0, P0);
  out.linear = k.smooth(d.linear_measurements);
  study_detail::LinearFunction  f(F);
  study_detail::MonotonicSensor h;
  ExtendedKalmanFilter          ekf(f, h, Q, Rn, x0, P0);
  out.ekf = ekf.filter(d.nonlinear_measurements);
  out.ukf = kalman_demo_ukf(d, o.ukf_alpha, o.max_input_steps);
  InformationFilter information(F, H, Q, R, x0, P0);
  out.information = information.filter(d.linear_measurements);
  EnsembleKalmanFilter ensemble(f, h, Q, Rn, x0, P0, o.ensemble_size, 7);
  out.ensemble = ensemble.filter(d.nonlinear_measurements);
  ParticleFilter particle(f, h, Q, Rn, x0, P0, o.particle_count, 7);
  out.particle = particle.filter(d.nonlinear_measurements);
  std::vector<double> measurements;
  for (const auto& z : d.linear_measurements)
    measurements.push_back(z[0]);
  AlphaBetaFilter ab(.5, .3, 1);
  out.alpha_beta = ab.filter(measurements);
  AlphaBetaGammaFilter abg(.5, .1, .01, 1);
  out.alpha_beta_gamma = abg.filter(d.acceleration_measurements);
  for (const auto* states :
       {&out.linear.filtered,
        &out.linear.smoothed,
        &out.ekf,
        &out.ukf,
        &out.information,
        &out.ensemble,
        &out.particle})
    study_detail::finite(*states);
  for (const auto& s : out.alpha_beta)
    if (!std::isfinite(s.position) || !std::isfinite(s.velocity))
      throw std::overflow_error("study alpha-beta output");
  for (const auto& s : out.alpha_beta_gamma)
    if (!std::isfinite(s.position) || !std::isfinite(s.velocity) || !std::isfinite(s.acceleration))
      throw std::overflow_error("study alpha-beta-gamma output");
  return out;
}
std::vector<double> kalman_state_column(
    const std::vector<KalmanState>& states,
    std::size_t                     coordinate) {
  std::vector<double> out;
  out.reserve(states.size());
  for (const auto& s : states) {
    if (coordinate >= s.x.size() || !std::isfinite(s.x[coordinate]))
      throw std::invalid_argument("study state coordinate");
    out.push_back(s.x[coordinate]);
  }
  return out;
}
long double kalman_position_mse(
    const std::vector<KalmanState>& states,
    const std::vector<double>&      truth) {
  if (states.empty() || states.size() != truth.size())
    throw std::invalid_argument("study MSE shape");
  long double sum = 0;
  for (std::size_t i = 0; i < states.size(); ++i) {
    if (states[i].x.empty() || !std::isfinite(states[i].x[0]) || !std::isfinite(truth[i]))
      throw std::invalid_argument("study MSE finite");
    const long double e = static_cast<long double>(states[i].x[0]) - truth[i];
    sum += e * e;
  }
  const auto mse = sum / states.size();
  if (!std::isfinite(mse))
    throw std::overflow_error("study MSE overflow");
  return mse;
}
long double kalman_min_eigenvalue2(const std::vector<std::vector<double>>& m) {
  if (m.size() != 2 || m[0].size() != 2 || m[1].size() != 2)
    throw std::invalid_argument("study two-state covariance shape");
  for (const auto& r : m)
    for (auto x : r)
      if (!std::isfinite(x))
        throw std::invalid_argument("study covariance finite");
  const long double a = m[0][0], c = m[1][1], b = (static_cast<long double>(m[0][1]) + m[1][0]) / 2;
  const auto        result = (a + c - std::hypot(a - c, 2 * b)) / 2;
  if (!std::isfinite(result))
    throw std::overflow_error("study covariance diagnostic");
  return result;
}
} // namespace datamunge::filter
