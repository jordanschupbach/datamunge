#pragma once
/// @file classical_components.hpp
/// @brief Tensor RBF/Fourier features, one Lloyd update and one categorical HMM expectation/update.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <datamunge/nn/autograd.hpp>
#include <datamunge/nn/ops.hpp>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>
namespace datamunge::nn::ml {
namespace nn = datamunge::nn; ///< Native owning tensor handles and operations.

/// @brief Internal validation shared by the classical tensor components.
namespace classical_detail {
/// @brief Reject nonfinite, non-CPU or incompatible dense feature matrices.
/// @param x Positive floating CPU[N,D] feature matrix.
/// @throws std::invalid_argument Invalid shape, dtype, backend or finite values.
inline void Features(const nn::Tensor& x) {
  if (!x.defined() || x.device().backend != compute::Backend::CPU || x.dim() != 2 ||
      x.size(0) < 1 || x.size(1) < 1 ||
      (x.dtype() != nn::DType::Float32 && x.dtype() != nn::DType::Float64) ||
      !nn::all(nn::is_finite(x)).item<bool>())
    throw std::invalid_argument("finite floating CPU feature matrix required");
}
} // namespace classical_detail

/// @brief Pairwise direct squared Euclidean distances without cancellation from norm expansion.
///
/// Finite CPU Float32/64 matrices of equal width/dtype. Live derivatives, shared input handles; no
/// input writes. O(NKD) time and temporary space. Standard floating overflow remains possible.
///
/// @param x Queries[N,D], positive axes.
/// @param y References[K,D], positive axes.
/// @return Live matching-dtype[N,K] distances.
/// @throws std::invalid_argument Invalid geometry, dtype, backend, nonfinite inputs or scalar
/// domain.
/// @par Example
/// @code
/// auto d=dm::nn::ml::SquaredDistances(dm::nn::tensor({0,2},{2,1}),dm::nn::tensor({1},{1,1}));
/// @endcode
/// @par References
/// - Rahimi and Recht (2007), Random Features for Large-Scale Kernel Machines,
/// https://people.eecs.berkeley.edu/~brecht/papers/07.rah.rec.nips.pdf .
/// @see RbfKernel()
inline nn::Tensor SquaredDistances(const nn::Tensor& x, const nn::Tensor& y) {
  classical_detail::Features(x);
  classical_detail::Features(y);
  if (x.dtype() != y.dtype() || x.size(1) != y.size(1))
    throw std::invalid_argument("feature widths/dtypes differ");
  const auto delta = nn::expand_dims(x, 1) - nn::expand_dims(y, 0);
  return nn::sum(nn::square(delta), {2});
}

/// @brief Apply exp(-gamma squared_distance) to two dense feature matrices.
///
/// Same matrix contract as SquaredDistances; shared input storage and live derivatives. The caller
/// supplies gamma as a finite positive Float32 scalar.
///
/// @param x Queries[N,D].
/// @param y References[K,D], matching dtype/width.
/// @param gamma Positive finite inverse length scale.
/// @return Live floating[N,K] kernel.
/// @throws std::invalid_argument Invalid geometry, dtype, backend, nonfinite inputs or scalar
/// domain.
/// @par Example
/// @code
/// auto k=dm::nn::ml::RbfKernel(dm::nn::ones({2,1}),dm::nn::ones({3,1}),.5f);
/// @endcode
/// @par References
/// - Rahimi and Recht (2007), Random Features for Large-Scale Kernel Machines,
/// https://people.eecs.berkeley.edu/~brecht/papers/07.rah.rec.nips.pdf .
/// @see SquaredDistances()
inline nn::Tensor RbfKernel(const nn::Tensor& x, const nn::Tensor& y, float gamma) {
  if (!std::isfinite(gamma) || gamma <= 0)
    throw std::invalid_argument("RBF gamma must be positive");
  return nn::exp(double(-gamma) * SquaredDistances(x, y));
}

/// @brief Own fixed seeded random cosine features for an RBF kernel.
///
/// The constructor uses std::mt19937_64 and standard-library Float32 normal/uniform distributions.
/// Results reproduce the source on the same standard-library implementation; no cross-library RNG
/// equivalence is promised. Public tensor copies share storage; synchronize writes against
/// Transform calls.
///
/// @throws std::invalid_argument Invalid geometry, dtype, backend, nonfinite inputs or scalar
/// domain.
/// @par Example
/// @code
/// dm::nn::ml::RandomFourierFeatures f(2,8,.5f,7);auto z=f.Transform(dm::nn::zeros({3,2}));
/// @endcode
/// @par References
/// - Rahimi and Recht (2007), Random Features for Large-Scale Kernel Machines,
/// https://people.eecs.berkeley.edu/~brecht/papers/07.rah.rec.nips.pdf .
/// @see Transform()
struct RandomFourierFeatures {
  /// @brief Draw frozen Float32 frequencies and phases with the source RNG order.
  ///
  /// W entries are independent N(0,2 gamma), followed by phases uniform in[0,2 pi). Owns CPU
  /// tensors; does not train or mutate any caller data.
  ///
  /// @param input_width Positive D.
  /// @param output_width Positive F.
  /// @param gamma Positive finite kernel parameter.
  /// @param seed Local RNG seed; global state is unchanged.
  /// @throws std::invalid_argument Invalid geometry, dtype, backend, nonfinite inputs or scalar
  /// domain.
  /// @par Example
  /// @code
  /// dm::nn::ml::RandomFourierFeatures f(2,8,.5f,7);
  /// @endcode
  /// @par References
  /// - Rahimi and Recht (2007), Random Features for Large-Scale Kernel Machines,
  /// https://people.eecs.berkeley.edu/~brecht/papers/07.rah.rec.nips.pdf .
  /// @see Transform()
  RandomFourierFeatures(int input_width, int output_width, float gamma, std::uint64_t seed) {
    if (input_width <= 0 || output_width <= 0 || !std::isfinite(gamma) || gamma <= 0)
      throw std::invalid_argument("positive RFF widths/gamma required");
    std::mt19937_64                       rng(seed);
    std::normal_distribution<float>       normal(0, std::sqrt(2 * gamma));
    std::uniform_real_distribution<float> uniform(0, 2 * std::acos(-1.f));
    std::vector<float> w(std::size_t(input_width) * output_width), b(output_width);
    for (auto& v : w)
      v = normal(rng);
    for (auto& v : b)
      v = uniform(rng);
    weights = nn::from_vector(w, {input_width, output_width});
    phase   = nn::from_vector(b, {output_width});
  }
  /// @brief Map features to sqrt(2/F) cos(x W + phase).
  ///
  /// Live CPU Float32[B,F] operation. Does not mutate weights, phases or inputs. Validates public
  /// mutable tensor geometry on every call. Input and frequency gradients remain available.
  ///
  /// @param x Finite Float32 CPU[B,D], positive B.
  /// @return Live Float32[B,F] features.
  /// @throws std::invalid_argument Invalid geometry, dtype, backend, nonfinite inputs or scalar
  /// domain.
  /// @par Example
  /// @code
  /// dm::nn::ml::RandomFourierFeatures f(2,8,.5f,7);auto z=f.Transform(dm::nn::zeros({3,2}));
  /// @endcode
  /// @par References
  /// - Rahimi and Recht (2007), Random Features for Large-Scale Kernel Machines,
  /// https://people.eecs.berkeley.edu/~brecht/papers/07.rah.rec.nips.pdf .
  /// @see SquaredDistances()
  nn::Tensor Transform(const nn::Tensor& x) const {
    classical_detail::Features(x);
    classical_detail::Features(weights);
    if (x.dtype() != nn::DType::Float32 || weights.dtype() != x.dtype() ||
        x.size(1) != weights.size(0) || phase.dtype() != x.dtype() || phase.dim() != 1 ||
        phase.numel() != weights.size(1) || phase.device().backend != compute::Backend::CPU ||
        !nn::all(nn::is_finite(phase)).item<bool>())
      throw std::invalid_argument("Fourier feature geometry/dtype mismatch");
    return double(std::sqrt(2.f / float(phase.numel()))) * nn::cos(nn::matmul(x, weights) + phase);
  }
  nn::Tensor weights; ///< Float32[D,F] shared mutable frequencies.
  nn::Tensor phase;   ///< Float32[F] shared mutable phase offsets.
};

/// @brief Retain one Lloyd update and diagnostics for the supplied centres.
///
/// Owning tensor handles; copies share storage. Assignments and inertia describe the old centres,
/// while centres contains the next means. No fitting loop or convergence rule is hidden in the
/// result.
///
/// @throws std::invalid_argument Invalid geometry, dtype, backend, nonfinite inputs or scalar
/// domain.
/// @par Example
/// @code
/// dm::nn::ml::KMeans m(1);auto s=m.Step(dm::nn::tensor({0,2},{2,1}),dm::nn::tensor({1},{1,1}));
/// @endcode
/// @par References
/// - Lloyd (1982), Least squares quantization in PCM, https://doi.org/10.1109/TIT.1982.1056489 .
/// @see KMeans::Step()
struct KMeansStep {
  nn::Tensor centers; ///< Live Float32[K,D] updated means; empty clusters retain previous centres.
  nn::Tensor labels;  ///< Int64[N] first-minimum assignments to the supplied centres.
  nn::Tensor squared_distances; ///< Float32[N] distances before the update.
  nn::Tensor counts;            ///< Float32[K] assignment counts.
  double     inertia = 0;       ///< Detached sum of pre-update squared distances.
};

/// @brief Provide seeded centre selection and a single tensor Lloyd update.
///
/// Finite CPU Float32[N,D] only. The immutable object stores a positive width; methods borrow
/// inputs and return owning handles. Step has live derivatives through sums and means within fixed
/// assignments. The caller owns the iteration count, stopping rule and history.
///
/// @throws std::invalid_argument Invalid geometry, dtype, backend, nonfinite inputs or scalar
/// domain.
/// @par Example
/// @code
/// dm::nn::ml::KMeans m(1);auto s=m.Step(dm::nn::tensor({0,2},{2,1}),dm::nn::tensor({1},{1,1}));
/// @endcode
/// @par References
/// - Lloyd (1982), Least squares quantization in PCM, https://doi.org/10.1109/TIT.1982.1056489 .
/// @see Step(), Initialize()
class KMeans {
 public:
  /// @brief Fix the positive feature width for a Lloyd component.
  ///
  /// No data or RNG state is retained. Independent calls may run concurrently if callers do not
  /// mutate shared input storage.
  ///
  /// @param dimensions Positive feature count D.
  /// @throws std::invalid_argument Invalid geometry, dtype, backend, nonfinite inputs or scalar
  /// domain.
  /// @par Example
  /// @code
  /// dm::nn::ml::KMeans m(2);
  /// @endcode
  /// @par References
  /// - Lloyd (1982), Least squares quantization in PCM, https://doi.org/10.1109/TIT.1982.1056489 .
  /// @see Step()
  explicit KMeans(int dimensions) : dimensions_(dimensions) {
    if (dimensions <= 0)
      throw std::invalid_argument("positive feature width required");
  }

  /// @brief Assign each row to the first nearest centre and compute next means.
  ///
  /// Finite CPU Float32 inputs. Empty clusters retain their old centres; ties select the lowest
  /// index. No input writes, optimizer or iterative fitting. Diagnostics describe distances before
  /// this update; centres retains live derivatives through selected assignments.
  ///
  /// @param x Data[N,D], positive axes and configured width.
  /// @param centers Prior centres[K,D], positive K; K may exceed N.
  /// @return Updated means, assignments, counts and pre-update inertia.
  /// @throws std::invalid_argument Invalid geometry, dtype, backend, nonfinite inputs or scalar
  /// domain.
  /// @par Example
  /// @code
  /// dm::nn::ml::KMeans m(1);auto s=m.Step(dm::nn::tensor({0,2},{2,1}),dm::nn::tensor({1},{1,1}));
  /// @endcode
  /// @par References
  /// - Lloyd (1982), Least squares quantization in PCM, https://doi.org/10.1109/TIT.1982.1056489 .
  /// @see Initialize()
  KMeansStep Step(const nn::Tensor& x, const nn::Tensor& centers) const {
    Validate(x);
    if (centers.dtype() != nn::DType::Float32 || centers.dim() != 2 ||
        centers.size(1) != dimensions_ || centers.size(0) < 1)
      throw std::invalid_argument("centers must be nonempty float32 [K,D]");
    classical_detail::Features(centers);
    const std::int64_t k         = centers.size(0);
    const auto         distances = SquaredDistances(x, centers);
    KMeansStep         out;
    out.labels            = nn::argmin(distances, 1);
    out.squared_distances = nn::min(distances, {1});
    out.inertia           = nn::sum(out.squared_distances, {0}).item<float>();
    const auto sums       = nn::unsorted_segment_sum(x, out.labels, k);
    out.counts = nn::unsorted_segment_sum(nn::ones_like(out.squared_distances), out.labels, k);
    const auto column = nn::expand_dims(out.counts, 1);
    out.centers       = nn::where(
        nn::greater(column, 0.0),
        sums / nn::maximum(column, nn::scalar_like(1.0, column)),
        centers);
    return out;
  }

  /// @brief Choose distinct source rows by random selection or k-means++ sampling.
  ///
  /// Local source-order std::mt19937_64 stream. All-zero residual distances choose an unused
  /// shuffled row. Returns independent Float32 CPU storage with no fitting loop; implementation may
  /// compute one assignment pass per selected centre.
  ///
  /// @param x Finite Float32 CPU[N,D].
  /// @param k Number of centres, 1..N.
  /// @param seed Local RNG seed.
  /// @param plus_plus Use distance-proportional seeding rather than shuffled rows.
  /// @return Owned Float32[K,D] centres.
  /// @throws std::invalid_argument Invalid geometry, dtype, backend, nonfinite inputs or scalar
  /// domain.
  /// @par Example
  /// @code
  /// dm::nn::ml::KMeans m(1);auto c=m.Initialize(dm::nn::tensor({0,2,4},{3,1}),2,7,true);
  /// @endcode
  /// @par References
  /// - Lloyd (1982), Least squares quantization in PCM, https://doi.org/10.1109/TIT.1982.1056489 .
  /// @see Step()
  nn::Tensor Initialize(const nn::Tensor& x, int k, uint64_t seed, bool plus_plus) const {
    Validate(x);
    const int64_t n = x.size(0);
    if (k <= 0 || k > n)
      throw std::invalid_argument("k must be in [1, rows]");
    std::mt19937_64     rng(seed);
    std::vector<size_t> candidates(n);
    std::iota(candidates.begin(), candidates.end(), 0);
    std::shuffle(candidates.begin(), candidates.end(), rng);
    std::vector<size_t> selected;
    const auto          values = x.to_vector<float>();
    nn::Tensor          centers;
    for (int i = 0; i < k; ++i) {
      size_t next = candidates[i];
      if (plus_plus && i > 0) {
        const auto          step = Step(x, centers);
        const auto          d2   = step.squared_distances.to_vector<float>();
        std::vector<double> weights(d2.begin(), d2.end());
        for (size_t row : selected)
          weights[row] = 0;
        if (std::accumulate(weights.begin(), weights.end(), 0.) > 0)
          next = std::discrete_distribution<size_t>(weights.begin(), weights.end())(rng);
        else
          for (size_t row : candidates)
            if (std::find(selected.begin(), selected.end(), row) == selected.end()) {
              next = row;
              break;
            }
      }
      selected.push_back(next);
      std::vector<float> c;
      for (size_t row : selected)
        c.insert(
            c.end(), values.begin() + row * dimensions_, values.begin() + (row + 1) * dimensions_);
      centers = nn::from_vector(c, {i + 1, dimensions_});
    }
    return centers;
  }

 private:
  /// @brief Check finite Float32 CPU data at the configured width.
  /// @param x Borrowed data matrix.
  /// @throws std::invalid_argument Invalid geometry, dtype, backend or values.
  void Validate(const nn::Tensor& x) const {
    classical_detail::Features(x);
    if (x.dtype() != nn::DType::Float32 || x.dim() != 2 || x.size(1) != dimensions_ ||
        x.size(0) < 1)
      throw std::invalid_argument("data must be nonempty float32 [N,D]");
    if (!nn::all(nn::is_finite(x)).item<bool>())
      throw std::invalid_argument("k-means requires finite observations");
  }
  int dimensions_; ///< Required positive feature width.
};

/// initial [K], transition [K,K], emission [K,V]; all float64 probability rows.
/// @brief Hold categorical HMM probability tables.
///
/// Float64 CPU probabilities in[0,1], each initial/transition/emission row sums to one within1e-10.
/// Tensor copies share storage; Run validates borrowed tables and never writes them.
///
/// @throws std::invalid_argument Invalid geometry, dtype, backend, nonfinite inputs or scalar
/// domain.
/// @par Example
/// @code
/// namespace n=dm::nn;n::ml::HmmState
/// s{n::full({2},.5,n::DType::Float64),n::full({2,2},.5,n::DType::Float64),n::full({2,2},.5,n::DType::Float64)};
/// @endcode
/// @par References
/// - Rabiner (1989), A tutorial on hidden Markov models and selected applications in speech
/// recognition, https://doi.org/10.1109/5.18626 .
/// @see HiddenMarkovModel::Run()
struct HmmState {
  nn::Tensor initial;    ///< Float64[K] normalized initial probabilities.
  nn::Tensor transition; ///< Float64[K,K] normalized from/to probability rows.
  nn::Tensor emission;   ///< Float64[K,V] normalized symbol probability rows.
};
/// @brief Own detached smoothing, best-path and one-update HMM results.
///
/// Copies share tensor storage. No hidden training schedule; next contains exactly one Baum-Welch
/// update. Path uses first-index ties. Input arrays remain unchanged.
///
/// @throws std::invalid_argument Invalid geometry, dtype, backend, nonfinite inputs or scalar
/// domain.
/// @par Example
/// @code
/// dm::nn::ml::HmmResult result; // assign the result of HiddenMarkovModel::Run
/// @endcode
/// @par References
/// - Rabiner (1989), A tutorial on hidden Markov models and selected applications in speech
/// recognition, https://doi.org/10.1109/5.18626 .
/// @see HmmState, HiddenMarkovModel
struct HmmResult {
  HmmState   next;           ///< Detached single Baum-Welch update; zero-count rows preserved.
  nn::Tensor log_likelihood; ///< Float64[B] scaled forward log evidence.
  nn::Tensor posterior;      ///< Float64[B,T,K] normalized smoothed state probabilities.
  nn::Tensor viterbi_score;  ///< Float64[B] best log path probabilities.
  nn::Tensor path;           ///< Int32[B,T] best paths; first-index ties.
};

/// @brief Run scaled categorical HMM inference and one expectation/maximization update.
///
/// Fixed positive K,V,T; accepts nonempty Int32 CPU[B,T] symbols and validated Float64 probability
/// tables. Runs without autograd, preserving source equations and first-index ties. Callers own EM
/// iterations, convergence and checkpoints. Shared input writes require synchronization.
///
/// @throws std::invalid_argument Invalid geometry, dtype, backend, nonfinite inputs or scalar
/// domain.
/// @par Example
/// @code
/// dm::nn::ml::HiddenMarkovModel model(2,2,3);
/// @endcode
/// @par References
/// - Rabiner (1989), A tutorial on hidden Markov models and selected applications in speech
/// recognition, https://doi.org/10.1109/5.18626 .
/// @see Run(), hmm_log_likelihood()
class HiddenMarkovModel {
 public:
  /// @brief Fix categorical state, symbol and sequence dimensions.
  ///
  /// Stores dimensions only; owns no learned table and does not fit anything.
  ///
  /// @param states Positive K.
  /// @param symbols Positive V.
  /// @param length Positive T.
  /// @throws std::invalid_argument Invalid geometry, dtype, backend, nonfinite inputs or scalar
  /// domain.
  /// @par Example
  /// @code
  /// dm::nn::ml::HiddenMarkovModel model(2,2,3);
  /// @endcode
  /// @par References
  /// - Rabiner (1989), A tutorial on hidden Markov models and selected applications in speech
  /// recognition, https://doi.org/10.1109/5.18626 .
  /// @see Run()
  HiddenMarkovModel(int states, int symbols, int length) : k_(states), v_(symbols), t_(length) {
    if (k_ <= 0 || v_ <= 0 || t_ <= 0)
      throw std::invalid_argument("positive HMM dimensions required");
  }

  /// observations: int32 [B,T] symbols in [0,V).
  /// @brief Compute scaled smoothing, Viterbi paths and exactly one Baum-Welch update.
  ///
  /// Detached Float64 CPU outputs. Zero-count transition/emission rows retain prior probabilities;
  /// T=1 retains all transition rows. No pseudocounts or fitting loop. O(B T K squared + B T K V)
  /// work; owns output storage. Numerically impossible sequences throw rather than returning
  /// invalid posteriors.
  ///
  /// @param observations Int32 CPU[B,T], B>0, symbols in[0,V).
  /// @param state Borrowed normalized Float64 CPU[K],[K,K],[K,V] tables.
  /// @return Detached likelihoods, smoothed state probabilities, best paths and next tables.
  /// @throws std::invalid_argument Invalid dimensions, symbol range, backend, dtype or normalized
  /// probability tables.
  /// @throws std::runtime_error A sequence has zero or numerically invalid probability.
  /// @par Example
  /// @code
  /// namespace n=dm::nn;n::ml::HiddenMarkovModel m(2,2,3);n::ml::HmmState
  /// s{n::full({2},.5,n::DType::Float64),n::full({2,2},.5,n::DType::Float64),n::full({2,2},.5,n::DType::Float64)};auto
  /// r=m.Run(n::zeros({1,3},n::DType::Int32),s);
  /// @endcode
  /// @par References
  /// - Rabiner (1989), A tutorial on hidden Markov models and selected applications in speech
  /// recognition, https://doi.org/10.1109/5.18626 .
  /// @see hmm_log_likelihood()
  HmmResult Run(const nn::Tensor& observations, const HmmState& state) const {
    Validate(observations, state);
    nn::NoGradGuard guard;
    const auto&     transition = state.transition;
    const auto emission_probs  = nn::gather(nn::transpose(state.emission), observations); // [B,T,K]
    const auto log_transition  = nn::log(transition);
    std::vector<nn::Tensor> evidence, alpha, beta(t_), scale, log_scale, posterior, transitions,
        pointers;
    nn::Tensor delta;
    for (int t = 0; t < t_; ++t) {
      evidence.push_back(nn::select(emission_probs, 1, t)); // [B,K]
      const auto prediction = t == 0 ? state.initial : nn::matmul(alpha.back(), transition);
      const auto joint      = prediction * evidence.back();
      scale.push_back(nn::sum(joint, {1}, true));
      alpha.push_back(joint / scale.back());
      log_scale.push_back(nn::log(scale.back()));
      if (t == 0) {
        delta = nn::log(state.initial) + nn::log(evidence.back());
        pointers.push_back(nn::zeros(delta.shape(), nn::DType::Int64));
      } else {
        const auto candidates = nn::expand_dims(delta, 2) + log_transition; // [B,from,to]
        pointers.push_back(nn::argmax(candidates, 1));
        delta = nn::max(candidates, {1}) + nn::log(evidence.back());
      }
    }
    HmmResult out;
    out.log_likelihood     = nn::reshape(nn::sum(nn::concat(log_scale, 1), {1}), {-1});
    const auto final_state = nn::argmax(delta, 1);
    out.viterbi_score      = nn::max(delta, {1});
    const auto ptrs        = nn::stack(pointers, 1); // [B,T,K]
    beta.back()            = nn::ones_like(alpha.back());
    for (int t = t_ - 2; t >= 0; --t)
      beta[t] = nn::matmul(evidence[t + 1] * beta[t + 1], transition, false, true) / scale[t + 1];
    for (int t = 0; t < t_; ++t) {
      const auto joint = alpha[t] * beta[t];
      posterior.push_back(joint / nn::sum(joint, {1}, true));
      if (t + 1 < t_) {
        const auto pair = nn::expand_dims(alpha[t], 2) * transition *
                          nn::expand_dims(evidence[t + 1] * beta[t + 1], 1);
        transitions.push_back(nn::sum(pair / nn::sum(pair, {1, 2}, true), {0}));
      }
    }
    out.posterior    = nn::stack(posterior, 1);
    out.next.initial = nn::mean(posterior.front(), {0});
    const auto keep  = [](const nn::Tensor& counts, const nn::Tensor& old) {
      const auto total = nn::sum(counts, {1}, true);
      return nn::where(
          nn::greater(total, 0.0),
          counts / nn::maximum(total, nn::scalar_like(1e-300, total)),
          old);
    };
    if (t_ > 1) {
      nn::Tensor sum = transitions[0];
      for (size_t i = 1; i < transitions.size(); ++i)
        sum = sum + transitions[i];
      out.next.transition = keep(sum, transition);
    } else {
      out.next.transition = transition.clone();
    }
    const auto one_hot_symbols =
        nn::one_hot(nn::reshape(observations, {-1}), v_, 1.0, 0.0, nn::DType::Float64);
    const auto emission_counts =
        nn::matmul(nn::reshape(out.posterior, {-1, k_}), one_hot_symbols, true, false);
    out.next.emission = keep(emission_counts, state.emission);

    const auto batch = observations.size(0);
    const auto ll    = out.log_likelihood.to_vector<double>(),
               vs    = out.viterbi_score.to_vector<double>();
    const auto last = final_state.to_vector<std::int64_t>(), back = ptrs.to_vector<std::int64_t>();
    std::vector<std::int32_t> path(static_cast<size_t>(batch * t_));
    for (std::int64_t b = 0; b < batch; ++b) {
      if (!std::isfinite(ll[b]) || !std::isfinite(vs[b]))
        throw std::runtime_error("observation sequence has zero or invalid HMM probability");
      std::int64_t s = last[b];
      for (int t = t_ - 1; t >= 0; --t) {
        path[b * t_ + t] = static_cast<std::int32_t>(s);
        if (t)
          s = back[(b * t_ + t) * k_ + s];
      }
    }
    out.path = nn::from_vector(path, {batch, t_});
    return out;
  }

 private:
  /// @brief Validate all borrowed HMM arrays before native inference.
  /// @param observations Int32 CPU[B,T] symbol IDs.
  /// @param state Float64 CPU normalized probability tables.
  /// @throws std::invalid_argument Invalid geometry, symbols or probabilities.
  void Validate(const nn::Tensor& observations, const HmmState& state) const {
    const auto f64 = nn::DType::Float64;
    for (const auto* x : {&observations, &state.initial, &state.transition, &state.emission})
      if (!x->defined() || x->device().backend != compute::Backend::CPU)
        throw std::invalid_argument("CPU HMM tensors required");
    if (observations.dtype() != nn::DType::Int32 || observations.dim() != 2 ||
        observations.size(0) < 1 || observations.size(1) != t_ || state.initial.dtype() != f64 ||
        state.initial.shape() != nn::Shape{k_} || state.transition.dtype() != f64 ||
        state.transition.shape() != nn::Shape{k_, k_} || state.emission.dtype() != f64 ||
        state.emission.shape() != nn::Shape{k_, v_})
      throw std::invalid_argument("HMM observation/state shape or dtype mismatch");
    for (auto symbol : observations.to_vector<std::int32_t>())
      if (symbol < 0 || symbol >= v_)
        throw std::invalid_argument("observation symbol out of range");
    const auto validate_rows = [](const nn::Tensor& p, int width) {
      const auto values = p.to_vector<double>();
      for (size_t start = 0; start < values.size(); start += width) {
        double sum = 0;
        for (int j = 0; j < width; ++j) {
          const double probability = values[start + j];
          if (!std::isfinite(probability) || probability < 0 || probability > 1)
            throw std::invalid_argument("invalid HMM probability");
          sum += probability;
        }
        if (std::abs(sum - 1) > 1e-10)
          throw std::invalid_argument("HMM probability row must sum to one");
      }
    };
    validate_rows(state.initial, k_);
    validate_rows(state.transition, k_);
    validate_rows(state.emission, v_);
  }

  int k_; ///< Positive state count.
  int v_; ///< Positive symbol count.
  int t_; ///< Positive fixed sequence length.
};
} // namespace datamunge::nn::ml
