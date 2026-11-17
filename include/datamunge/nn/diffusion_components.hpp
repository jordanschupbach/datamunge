#pragma once
/// @file diffusion_components.hpp
/// @brief Explicit compact diffusion predictors, live time features and DDIM transitions.
#include <datamunge/nn/diffusion.hpp>
#include <random>
namespace datamunge::nn::models {
/// @brief Initialize the source compact two-SiLU-layer predictor leaves.
/// Creates hidden1/W,b, hidden2/W,b, output/W,b in that source order: widths input->32->32->output,
/// normal(.1) kernels and zero biases. Existing valid leaves are reused; never resets leaves,
/// creates an optimizer or runs training. Shared mutable stores require caller serialization;
/// creation failures may leave earlier missing leaves present.
/// @param parameters Caller-owned floating parameter store; dtype Float32/64.
/// @param input_width Positive encoded feature width, at most1024.
/// @param output_width Positive prediction width, at most1024.
/// @throws std::invalid_argument Invalid finite CPU Float32/64 geometry, dtype, index or
/// coefficient.
/// @par Example
/// @code
/// namespace n=dm::nn; n::ParameterStore p(11003); n::models::InitializeDiffusionMlp(p,72,64);
/// @endcode
/// @par References
/// - Original compact diffusion examples; Song, Meng and Ermon, Denoising Diffusion Implicit
/// Models(2020), https://arxiv.org/abs/2010.02502; Song and Ermon, Generative Modeling by
/// Estimating Gradients of the Data Distribution(2019), https://arxiv.org/abs/1907.05600. These
/// functions expose single tensor operations, not fitting/sampling orchestration.
/// @see CosineDiffusionSchedule(), DiffusionNoised(), DdpmReverse()
inline void InitializeDiffusionMlp(
    ParameterStore& parameters,
    std::int64_t    input_width,
    std::int64_t    output_width) {
  if (input_width <= 0 || output_width <= 0 || input_width > 1024 || output_width > 1024 ||
      (parameters.dtype() != DType::Float32 && parameters.dtype() != DType::Float64))
    throw std::invalid_argument("diffusion MLP widths/dtype");
  for (const std::string name : {"hidden1", "hidden2", "output"}) {
    auto in = name == "hidden1" ? input_width : 32, out = name == "output" ? output_width : 32;
    parameters.get(name + "/W", {in, out}, init::normal(.1));
    parameters.get(name + "/b", {out}, init::zeros());
  }
}
/// @brief Own live compact-MLP prediction and hidden endpoints.
/// Fields retain graphs/storage after this record and the original store are collected; no cached
/// model or optimizer.
/// @throws std::invalid_argument Invalid finite CPU Float32/64 geometry, dtype, index or
/// coefficient.
/// @par Example
/// @code
/// namespace n=dm::nn; n::ParameterStore p(11003); n::models::InitializeDiffusionMlp(p,72,64);auto
/// out=n::models::DiffusionMlp(p,n::zeros({2,72}));
/// @endcode
/// @par References
/// - Original compact diffusion examples; Song, Meng and Ermon, Denoising Diffusion Implicit
/// Models(2020), https://arxiv.org/abs/2010.02502; Song and Ermon, Generative Modeling by
/// Estimating Gradients of the Data Distribution(2019), https://arxiv.org/abs/1907.05600. These
/// functions expose single tensor operations, not fitting/sampling orchestration.
/// @see CosineDiffusionSchedule(), DiffusionNoised(), DdpmReverse()
struct DiffusionMlpOutput {
  Tensor value;   ///< Live CPU[N,output] prediction.
  Tensor hidden1; ///< Live CPU[N,32] first SiLU endpoint.
  Tensor hidden2; ///< Live CPU[N,32] second SiLU endpoint.
};
/// @brief Evaluate explicitly initialized compact predictor leaves without mutation.
/// No missing-leaf creation, RNG draws, fitting, optimizer or hidden no-grad. Computes
/// bias_add(matmul(x,W),b), with SiLU x*sigmoid(x), preserving source operation grouping.
/// O(N*parameter_values) work and first/higher input/leaf derivatives; arbitrary strides accepted.
/// @param parameters Read-only store initialized by InitializeDiffusionMlp, matching encoded dtype.
/// @param encoded Finite Float32/64 CPU[N,input_width], N>0.
/// @return Owning live prediction and both hidden endpoints.
/// @throws std::invalid_argument Invalid finite CPU Float32/64 geometry, dtype, index or
/// coefficient.
/// @throws std::out_of_range A required explicitly initialized predictor leaf is missing.
/// @par Example
/// @code
/// namespace n=dm::nn; n::ParameterStore p(11003); n::models::InitializeDiffusionMlp(p,72,64);auto
/// out=n::models::DiffusionMlp(p,n::zeros({2,72}));
/// @endcode
/// @par References
/// - Original compact diffusion examples; Song, Meng and Ermon, Denoising Diffusion Implicit
/// Models(2020), https://arxiv.org/abs/2010.02502; Song and Ermon, Generative Modeling by
/// Estimating Gradients of the Data Distribution(2019), https://arxiv.org/abs/1907.05600. These
/// functions expose single tensor operations, not fitting/sampling orchestration.
/// @see CosineDiffusionSchedule(), DiffusionNoised(), DdpmReverse()
inline DiffusionMlpOutput DiffusionMlp(const ParameterStore& parameters, const Tensor& encoded) {
  diffusion_detail::Floating(encoded);
  if (encoded.dim() != 2 || encoded.dtype() != parameters.dtype())
    throw std::invalid_argument("diffusion encoded matrix/store dtype");
  Tensor                h = encoded;
  std::array<Tensor, 3> endpoints;
  int                   index = 0;
  for (const std::string name : {"hidden1", "hidden2", "output"}) {
    const auto w = parameters.at(name + "/W"), bias = parameters.at(name + "/b");
    diffusion_detail::Floating(w);
    gan_detail::Finite(bias);
    if (w.dim() != 2 || bias.dim() != 1 || w.size(0) != h.size(1) || w.size(1) != bias.size(0) ||
        w.dtype() != h.dtype() || bias.dtype() != h.dtype() ||
        (name != "output" && w.size(1) != 32))
      throw std::invalid_argument("diffusion MLP leaf shape/dtype");
    auto z = bias_add(matmul(h, w), bias);
    gan_detail::Finite(z);
    h = name == "output" ? z : z * sigmoid(z);
    gan_detail::Finite(h);
    endpoints[index++] = h;
  }
  return {endpoints[2], endpoints[0], endpoints[1]};
}
/// @brief Embed continuous or structural time with the source-rounded sinusoidal phases.
/// Cast time to Float32, divide by maximum_time, then multiply the four Float32 pi,2pi,4pi,8pi
/// constants before sin/cos. No hidden no-grad, RNG or mutation. Floating-time derivatives follow
/// the primitive cast/phase semantics, including Float32 rounding; this is not a smooth exact-real
/// embedding. Work/storage O(N).
/// @param time CPU[N] Int32/64 structural IDs or live Float32/64 nonnegative times.
/// @param maximum_time Finite positive Float32 normalization, default31.
/// @param dtype Output Float32(default) or Float64; source phase math remains Float32.
/// @return Owning [N,8] features, live for floating time and structural for integer time.
/// @throws std::invalid_argument Invalid finite CPU Float32/64 geometry, dtype, index or
/// coefficient.
/// @par Example
/// @code
/// namespace n=dm::nn;auto t=n::from_vector(std::vector<float>{0,1},{2});auto
/// e=n::models::DiffusionContinuousTimeEmbedding(t,1);
/// @endcode
/// @par References
/// - Original compact diffusion examples; Song, Meng and Ermon, Denoising Diffusion Implicit
/// Models(2020), https://arxiv.org/abs/2010.02502; Song and Ermon, Generative Modeling by
/// Estimating Gradients of the Data Distribution(2019), https://arxiv.org/abs/1907.05600. These
/// functions expose single tensor operations, not fitting/sampling orchestration.
/// @see CosineDiffusionSchedule(), DiffusionNoised(), DdpmReverse()
inline Tensor DiffusionContinuousTimeEmbedding(
    const Tensor& time,
    float         maximum_time = 31.f,
    DType         dtype        = DType::Float32) {
  if (!time.defined() || time.device().backend != compute::Backend::CPU || time.dim() != 1 ||
      time.size(0) <= 0 || !std::isfinite(maximum_time) || maximum_time <= 0 ||
      (dtype != DType::Float32 && dtype != DType::Float64) ||
      (time.dtype() != DType::Float32 && time.dtype() != DType::Float64 &&
       time.dtype() != DType::Int32 && time.dtype() != DType::Int64))
    throw std::invalid_argument("diffusion time embedding geometry/options");
  for (double value : cast(time, DType::Float64).to_vector<double>())
    if (!std::isfinite(value) || value < 0)
      throw std::invalid_argument("diffusion nonnegative finite time");
  auto   normalized = reshape(cast(time, DType::Float32), {-1, 1}) / double(maximum_time);
  double pi         = std::acos(-1.);
  auto   phase =
      normalized *
      from_vector(
          std::vector<float>{float(pi), float(2 * pi), float(4 * pi), float(8 * pi)}, {1, 4});
  gan_detail::Finite(phase);
  auto output = cast(concat({sin(phase), cos(phase)}, 1), dtype);
  gan_detail::Finite(output);
  return output;
}
/// @brief Own one live clipping-consistent DDIM reverse transition.
/// State paths follow native arithmetic promotion with eta; extracted schedule coefficients retain
/// source Float32 rounding. No model, random sampler, full trajectory or optimizer is retained.
/// @throws std::invalid_argument Invalid finite CPU Float32/64 geometry, dtype, index or
/// coefficient.
/// @par Example
/// @code
/// namespace n=dm::nn; n::models::DdimReverseOutput out;
/// @endcode
/// @par References
/// - Original compact diffusion examples; Song, Meng and Ermon, Denoising Diffusion Implicit
/// Models(2020), https://arxiv.org/abs/2010.02502; Song and Ermon, Generative Modeling by
/// Estimating Gradients of the Data Distribution(2019), https://arxiv.org/abs/1907.05600. These
/// functions expose single tensor operations, not fitting/sampling orchestration.
/// @see CosineDiffusionSchedule(), DiffusionNoised(), DdpmReverse()
struct DdimReverseOutput {
  Tensor value;              ///< Live next sample, state shape.
  Tensor mean;               ///< Live deterministic transition mean.
  Tensor sigma;              ///< Broadcast standard deviation from source coefficient arithmetic.
  Tensor direction;          ///< Broadcast remaining noise-direction coefficient.
  Tensor x0_raw;             ///< Live unclipped inferred clean sample.
  Tensor x0_clipped;         ///< Live inferred clean sample clipped to[-1,1].
  Tensor epsilon_consistent; ///< Live epsilon recomputed from clipped x0.
};
/// @brief Compute one explicitly conditioned DDIM reverse step.
/// Uses source Float32 extraction and binary grouping. previous=-1 gives alpha_bar_previous1; eta0
/// ignores innovation; eta1 on adjacent levels matches the clipped-x0 DDPM posterior within
/// floating arithmetic. No hidden no-grad, sampling, fitting or mutation; first/higher applicable
/// paths survive record/store collection. Integer indices are structural. Clipping, maximum and
/// sqrt use native nonsmooth derivative conventions; no differentiability promise at kink
/// boundaries. O(sample elements+table length).
/// @param state Finite Float32/64 CPU[N,...] noisy sample, rank>=2.
/// @param time Int32/64 CPU[N] current schedule indices.
/// @param previous Int32/64 CPU[N], -1 for clean endpoint or0<=previous<time.
/// @param epsilon Finite matching predicted noise.
/// @param innovation Finite matching explicit standard-normal innovation; never sampled here.
/// @param eta Finite floating CPU scalar in[0,1], Float32/64.
/// @param schedule Explicit validated consumed alpha-bar/signal/noise tables.
/// @return Owning live next sample, means and clipping-consistent diagnostics.
/// @throws std::invalid_argument Invalid finite CPU Float32/64 geometry, dtype, index or
/// coefficient.
/// @par Example
/// @code
/// namespace n=dm::nn;auto
/// t=n::from_vector(std::vector<std::int32_t>{1},{1}),prev=n::from_vector(std::vector<std::int32_t>{0},{1});auto
/// x=n::zeros({1,2});auto
/// out=n::models::DdimReverse(x,t,prev,x,x,n::scalar(0),n::models::CosineDiffusionSchedule());
/// @endcode
/// @par References
/// - Original compact diffusion examples; Song, Meng and Ermon, Denoising Diffusion Implicit
/// Models(2020), https://arxiv.org/abs/2010.02502; Song and Ermon, Generative Modeling by
/// Estimating Gradients of the Data Distribution(2019), https://arxiv.org/abs/1907.05600. These
/// functions expose single tensor operations, not fitting/sampling orchestration.
/// @see CosineDiffusionSchedule(), DiffusionNoised(), DdpmReverse()
inline DdimReverseOutput DdimReverse(
    const Tensor&                  state,
    const Tensor&                  time,
    const Tensor&                  previous,
    const Tensor&                  epsilon,
    const Tensor&                  innovation,
    const Tensor&                  eta,
    const DiffusionScheduleOutput& schedule) {
  diffusion_detail::Same(state, epsilon);
  diffusion_detail::Same(state, innovation);
  diffusion_detail::Time(time, state.size(0));
  if (!previous.defined() || previous.dim() != 1 || previous.shape() != time.shape() ||
      previous.device().backend != compute::Backend::CPU ||
      (previous.dtype() != DType::Int32 && previous.dtype() != DType::Int64))
    throw std::invalid_argument("DDIM previous indices");
  diffusion_detail::Floating(eta, 0);
  if (eta.dim() != 0 || eta.item<double>() < 0 || eta.item<double>() > 1)
    throw std::invalid_argument("DDIM eta");
  auto current = cast(time, DType::Int64).to_vector<std::int64_t>(),
       prior   = cast(previous, DType::Int64).to_vector<std::int64_t>();
  std::vector<std::int32_t> clamped(prior.size());
  for (std::size_t i = 0; i < prior.size(); ++i) {
    if (prior[i] < -1 || prior[i] >= current[i])
      throw std::invalid_argument("DDIM ordered previous times");
    clamped[i] = std::max<std::int64_t>(0, prior[i]);
  }
  if (schedule.alpha_bar.shape() != schedule.sqrt_alpha_bar.shape() ||
      schedule.alpha_bar.shape() != schedule.sqrt_one_minus_alpha_bar.shape())
    throw std::invalid_argument("DDIM schedule table lengths");
  int  rank        = int(state.dim());
  auto bar         = DiffusionExtract(schedule.alpha_bar, time, rank),
       signal      = DiffusionExtract(schedule.sqrt_alpha_bar, time, rank),
       noise_scale = DiffusionExtract(schedule.sqrt_one_minus_alpha_bar, time, rank);
  diffusion_detail::Positive(signal, true);
  diffusion_detail::Positive(noise_scale, true);
  diffusion_detail::Positive(bar, true);
  for (double v : bar.to_vector<double>())
    if (v >= 1)
      throw std::invalid_argument("DDIM alpha bar must be below1");
  auto  raw     = div(sub(state, mul(noise_scale, epsilon)), signal);
  auto  clipped = minimum(maximum(raw, scalar_like(-1, raw)), scalar_like(1, raw));
  Shape broadcast(rank, 1);
  broadcast[0] = state.size(0);
  auto condition =
      reshape(greater_equal(cast(previous, DType::Int32), scalar(0, DType::Int32)), broadcast);
  auto prevbar = where(
      condition,
      DiffusionExtract(schedule.alpha_bar, from_vector(clamped, {state.size(0)}), rank),
      add(mul(bar, 0.), 1.));
  const auto pv = prevbar.to_vector<double>(), cv = bar.to_vector<double>();
  for (std::size_t i = 0; i < prior.size(); ++i) {
    double pb = pv[i], cb = cv[i];
    if (pb < cb || pb > 1)
      throw std::invalid_argument("DDIM decreasing alpha bar");
  }
  auto variance =
      mul(square(eta), mul(div(sub(1., prevbar), sub(1., bar)), sub(1., div(bar, prevbar))));
  auto sigma      = sqrt(maximum(variance, scalar_like(0, variance)));
  auto direction  = sqrt(maximum(sub(sub(1., prevbar), variance), scalar_like(0, variance)));
  auto consistent = div(sub(state, mul(signal, clipped)), noise_scale);
  auto mean       = add(mul(sqrt(prevbar), clipped), mul(direction, consistent));
  auto value      = add(mean, mul(sigma, innovation));
  for (auto x : {raw, clipped, sigma, direction, consistent, mean, value})
    gan_detail::Finite(x);
  return {value, mean, sigma, direction, raw, clipped, consistent};
}

/// @brief Internal CPU geometry and scalar-domain checks for compact diffusion components.
namespace diffusion_component_detail {
/// @brief Validate aligned finite point/time tensors and an inclusive physical-time interval.
/// @param x Finite CPU[N,2] floating points.
/// @param time Matching floating CPU[N] times.
/// @param lower Inclusive lower time.
/// @param upper Inclusive upper time.
/// @throws std::invalid_argument Invalid geometry, dtype or interval membership.
inline void Points(const Tensor& x, const Tensor& time, double lower, double upper) {
  diffusion_detail::Floating(x);
  diffusion_detail::Floating(time, 1);
  if (x.dim() != 2 || x.size(1) != 2 || time.dim() != 1 || time.size(0) != x.size(0) ||
      time.dtype() != x.dtype())
    throw std::invalid_argument("point/time geometry or dtype");
  for (double t : time.to_vector<double>())
    if (t < lower || t > upper)
      throw std::invalid_argument("point time range");
}
/// @brief Validate finite floating CPU[N,8,8,1] state geometry.
/// @param x Batched image-shaped state.
/// @throws std::invalid_argument Invalid dtype, device or geometry.
inline void Image(const Tensor& x) {
  diffusion_detail::Floating(x);
  if (x.dim() != 4 || x.size(1) != 8 || x.size(2) != 8 || x.size(3) != 1)
    throw std::invalid_argument("diffusion image requires [N,8,8,1]");
}
/// @brief Require a finite nonnegative floating CPU scalar.
/// @param h Explicit transition strength or step.
/// @return Unchanged owning tensor handle.
/// @throws std::invalid_argument Invalid geometry, dtype or sign.
inline Tensor Scale(const Tensor& h) {
  diffusion_detail::Floating(h, 0);
  if (h.dim() != 0 || h.item<double>() < 0)
    throw std::invalid_argument("nonnegative scalar step required");
  return h;
}
} // namespace diffusion_component_detail
/// @brief Owning live image predictor endpoints, independent of caller store lifetime.
/// @par Example
/// @code
/// dm::nn::models::DiffusionImageOutput endpoints;
/// @endcode
/// @see InitializeDiffusionMlp()
/// @par References
/// - Ho et al. (2020), Denoising Diffusion Probabilistic Models, https://arxiv.org/abs/2006.11239 ;
/// Song et al. (2021), Score-Based Generative Modeling through SDEs,
/// https://arxiv.org/abs/2011.13456 . Source-rounded fixture details are specified by each function
/// contract.
struct DiffusionImageOutput {
  Tensor epsilon;        ///< Live CPU[N,8,8,1] predicted noise.
  Tensor time_embedding; ///< Live CPU[N,8] source-rounded time features.
  Tensor hidden1;        ///< Live CPU[N,32] first SiLU.
  Tensor hidden2;        ///< Live CPU[N,32] second SiLU.
};
/// @brief Predict image epsilon using caller-owned compact MLP leaves.
/// No mutation, RNG or training; floating time and input/parameter derivative paths remain live.
/// Work O(N*5504).
/// @param p Read-only 72->32->32->64 initialized store.
/// @param state Finite floating CPU[N,8,8,1], same dtype as store.
/// @param time CPU[N] nonnegative integer or floating conditioning times.
/// @param maximum_time Finite positive source Float32 time normalization, default31.
/// @return Owning live epsilon, embedding and hidden endpoints.
/// @throws std::invalid_argument Invalid CPU geometry, dtype, finite value or domain.
/// @throws std::out_of_range A required explicitly initialized predictor leaf is missing.
/// @par Example
/// @code
/// namespace n=dm::nn; n::ParameterStore p(12003); n::models::InitializeDiffusionMlp(p,72,64);auto
/// v=n::models::DiffusionImagePrediction(p,n::zeros({1,8,8,1}),n::zeros({1}));
/// @endcode
/// @par References
/// - Ho et al. (2020), Denoising Diffusion Probabilistic Models, https://arxiv.org/abs/2006.11239 ;
/// Song et al. (2021), Score-Based Generative Modeling through SDEs,
/// https://arxiv.org/abs/2011.13456 . Source-rounded fixture details are specified by each function
/// contract.
inline DiffusionImageOutput DiffusionImagePrediction(
    const ParameterStore& p,
    const Tensor&         state,
    const Tensor&         time,
    float                 maximum_time = 31.f) {
  diffusion_component_detail::Image(state);
  if (!time.defined() || time.dim() != 1 || time.size(0) != state.size(0))
    throw std::invalid_argument("image/time batch");
  auto embedding = DiffusionContinuousTimeEmbedding(time, maximum_time, state.dtype());
  auto path      = DiffusionMlp(p, concat({reshape(state, {-1, 64}), embedding}, 1));
  return {reshape(path.value, {-1, 8, 8, 1}), embedding, path.hidden1, path.hidden2};
}
/// @brief Extract the original sixteen geometric noise scales.
/// Computes Double pow(.05,i/15), then source Float32 extraction. Structural indices have no
/// gradient. O(N); no mutable cache or RNG.
/// @param level Int32/64 CPU[N] indices in[0,15].
/// @return Owning detached Float32 CPU[N,1,1,1] scales.
/// @throws std::invalid_argument Invalid CPU geometry, dtype, finite value or domain.
/// @par Example
/// @code
/// namespace n=dm::nn;auto sigma=n::models::NoiseConditionalSigma(n::zeros({2},n::DType::Int32));
/// @endcode
/// @par References
/// - Ho et al. (2020), Denoising Diffusion Probabilistic Models, https://arxiv.org/abs/2006.11239 ;
/// Song et al. (2021), Score-Based Generative Modeling through SDEs,
/// https://arxiv.org/abs/2011.13456 . Source-rounded fixture details are specified by each function
/// contract.
inline Tensor NoiseConditionalSigma(const Tensor& level) {
  diffusion_detail::Time(level, 0, 16);
  std::vector<float> out;
  for (auto index : cast(level, DType::Int64).to_vector<std::int64_t>())
    out.push_back(float(std::pow(.05, index / 15.)));
  return from_vector(out, {level.size(0), 1, 1, 1});
}
/// @brief Owning live noise-conditional score endpoints; no model lifetime dependency.
/// @par Example
/// @code
/// dm::nn::models::NoiseConditionalScoreOutput endpoints;
/// @endcode
/// @see InitializeDiffusionMlp()
/// @par References
/// - Ho et al. (2020), Denoising Diffusion Probabilistic Models, https://arxiv.org/abs/2006.11239 ;
/// Song et al. (2021), Score-Based Generative Modeling through SDEs,
/// https://arxiv.org/abs/2011.13456 . Source-rounded fixture details are specified by each function
/// contract.
struct NoiseConditionalScoreOutput {
  Tensor score;            ///< Negative epsilon divided by source Float32 sigma.
  Tensor normalized_score; ///< Negative epsilon.
  Tensor epsilon;          ///< Live predicted epsilon.
  Tensor time_embedding;   ///< Source-rounded [N,8] embedding.
  Tensor hidden1;          ///< First SiLU endpoint.
  Tensor hidden2;          ///< Second SiLU endpoint.
};
/// @brief Evaluate the sixteen-level noise-conditional image score.
/// Score=-epsilon/sigma; no fitting, RNG, mutation or hidden no-grad. Source Song and Ermon(2019),
/// https://arxiv.org/abs/1907.05600. Work O(N*5504).
/// @param p Read-only initialized 72->32->32->64 leaves.
/// @param state Finite floating CPU[N,8,8,1], matching store dtype.
/// @param level Int32/64 CPU[N] in[0,15].
/// @return Owning live score and predictor diagnostics.
/// @throws std::invalid_argument Invalid CPU geometry, dtype, finite value or domain.
/// @throws std::out_of_range A required explicitly initialized predictor leaf is missing.
/// @par Example
/// @code
/// namespace n=dm::nn;n::ParameterStore p(12003);n::models::InitializeDiffusionMlp(p,72,64);auto
/// y=n::models::NoiseConditionalScore(p,n::zeros({1,8,8,1}),n::zeros({1},n::DType::Int32));
/// @endcode
/// @par References
/// - Ho et al. (2020), Denoising Diffusion Probabilistic Models, https://arxiv.org/abs/2006.11239 ;
/// Song et al. (2021), Score-Based Generative Modeling through SDEs,
/// https://arxiv.org/abs/2011.13456 . Source-rounded fixture details are specified by each function
/// contract.
inline NoiseConditionalScoreOutput
NoiseConditionalScore(const ParameterStore& p, const Tensor& state, const Tensor& level) {
  auto sigma      = NoiseConditionalSigma(level);
  auto path       = DiffusionImagePrediction(p, state, level, 15.f);
  auto normalized = neg(path.epsilon);
  auto score      = div(normalized, sigma);
  gan_detail::Finite(score);
  return {score, normalized, path.epsilon, path.time_embedding, path.hidden1, path.hidden2};
}
/// @brief Owning live single Langevin transition, without a sampling loop.
/// @par Example
/// @code
/// dm::nn::models::AnnealedLangevinOutput endpoints;
/// @endcode
/// @see InitializeDiffusionMlp()
/// @par References
/// - Ho et al. (2020), Denoising Diffusion Probabilistic Models, https://arxiv.org/abs/2006.11239 ;
/// Song et al. (2021), Score-Based Generative Modeling through SDEs,
/// https://arxiv.org/abs/2011.13456 . Source-rounded fixture details are specified by each function
/// contract.
struct AnnealedLangevinOutput {
  Tensor value; ///< Next sample, same shape as state.
  Tensor mean;  ///< Deterministic drift update.
  Tensor alpha; ///< Float32 CPU[N,1,1,1] step coefficient.
};
/// @brief Apply one annealed Langevin transition with explicit innovation.
/// Uses alpha=2e-5*(sigma/.05)^2 and x+.5*alpha*score+sqrt(alpha)*innovation. No RNG, mutation or
/// detachment; applicable tensor paths retain gradients. Work O(N*64).
/// @param state Finite floating CPU[N,8,8,1].
/// @param level Int32/64 CPU[N] in[0,15].
/// @param score Finite score matching state shape/dtype.
/// @param innovation Finite explicit innovation matching state shape/dtype.
/// @return Live next sample, mean and broadcast alpha.
/// @throws std::invalid_argument Invalid CPU geometry, dtype, finite value or domain.
/// @par Example
/// @code
/// namespace n=dm::nn;auto x=n::zeros({1,8,8,1});auto
/// y=n::models::AnnealedLangevin(x,n::zeros({1},n::DType::Int32),x,x);
/// @endcode
/// @par References
/// - Ho et al. (2020), Denoising Diffusion Probabilistic Models, https://arxiv.org/abs/2006.11239 ;
/// Song et al. (2021), Score-Based Generative Modeling through SDEs,
/// https://arxiv.org/abs/2011.13456 . Source-rounded fixture details are specified by each function
/// contract.
inline AnnealedLangevinOutput AnnealedLangevin(
    const Tensor& state,
    const Tensor& level,
    const Tensor& score,
    const Tensor& innovation) {
  diffusion_component_detail::Image(state);
  diffusion_detail::Same(state, score);
  diffusion_detail::Same(state, innovation);
  diffusion_detail::Time(level, state.size(0), 16);
  if (level.size(0) != state.size(0))
    throw std::invalid_argument("Langevin batch");
  auto alpha = mul(square(div(NoiseConditionalSigma(level), .05)), 2e-5);
  auto mean  = add(state, mul(mul(alpha, .5), score));
  auto value = add(mean, mul(sqrt(alpha), innovation));
  gan_detail::Finite(value);
  return {value, mean, alpha};
}
/// @brief Owning live consistency-map endpoints, with no retained model or optimizer.
/// @par Example
/// @code
/// dm::nn::models::ConsistencyMapOutput endpoints;
/// @endcode
/// @see InitializeDiffusionMlp()
/// @par References
/// - Ho et al. (2020), Denoising Diffusion Probabilistic Models, https://arxiv.org/abs/2006.11239 ;
/// Song et al. (2021), Score-Based Generative Modeling through SDEs,
/// https://arxiv.org/abs/2011.13456 . Source-rounded fixture details are specified by each function
/// contract.
struct ConsistencyMapOutput {
  Tensor value;          ///< Live CPU[N,2] consistency value.
  Tensor raw;            ///< Live CPU[N,2] raw compact predictor output.
  Tensor skip;           ///< Live CPU[N,1] input skip coefficient.
  Tensor out;            ///< Live CPU[N,1] predictor coefficient.
  Tensor time_embedding; ///< Live CPU[N,8] log-time features.
};

/// @brief Evaluate the boundary-preserving compact consistency map.
///
/// Log-time source Float32 phase constants; f(x,.01f)=x. No optimizer, target EMA, fitting, RNG or
/// detachment. O(N*1474).
///
/// @param p Read-only initialized 10->32->32->2 store matching x dtype.
/// @param x Finite CPU[N,2] Float32/64 noisy points.
/// @param time Finite matching floating CPU[N] in[Float32(.01),2].
/// @return Owning live value, raw MLP output, skip/output coefficients and time features.
/// @throws std::invalid_argument Invalid finite CPU geometry, dtype or domain.
/// @throws std::out_of_range A required explicitly initialized predictor leaf is missing.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore
/// p(19003);m::InitializeDiffusionMlp(p,10,2);auto
/// y=m::ConsistencyMap(p,n::zeros({2,2}),n::full({2},.01f));
/// @endcode
/// @par References
/// - Song, Dhariwal, Chen and Sutskever (2023), Consistency Models,
/// https://arxiv.org/abs/2303.01469. This compact Gaussian-teacher fixture follows the original
/// repository example.
/// @see InitializeDiffusionMlp(), DiffusionMlp()
inline ConsistencyMapOutput
ConsistencyMap(const ParameterStore& p, const Tensor& x, const Tensor& time) {
  diffusion_component_detail::Points(x, time, double(.01f), 2.);
  const Tensor t = reshape(time, {-1, 1});
  const Tensor phase =
      mul(div(log(div(t, double(.01f))), double(float(std::log(200.)))),
          from_vector(
              std::vector<float>{
                  float(std::acos(-1.)),
                  float(2 * std::acos(-1.)),
                  float(4 * std::acos(-1.)),
                  float(8 * std::acos(-1.))},
              {1, 4}));
  const Tensor embedding = concat({sin(phase), cos(phase)}, 1);
  const Tensor scale     = sqrt(add(square(t), double(.25f)));
  const Tensor input     = concat({div(x, scale), embedding}, 1);
  const Tensor raw       = DiffusionMlp(p, input).value;
  const Tensor shift     = sub(t, double(.01f));
  const Tensor skip      = div(.25, add(square(shift), .25));
  const Tensor out       = div(mul(shift, .5), scale);
  return {add(mul(skip, x), mul(out, raw)), raw, skip, out, embedding};
}

/// @brief Evaluate the original Gaussian VE teacher drift.
///
/// No RNG, fitting, mutation or detachment; O(N), applicable input/time derivatives retained.
///
/// @param x Finite floating CPU[N,2].
/// @param time Same-dtype finite CPU[N] in[Float32(.01),2].
/// @return Live t*(x-mu)/(variance+t²) with source Float32 mu=(.3,-.2), variance=(.49,.25).
/// @throws std::invalid_argument Invalid finite CPU geometry, dtype or domain.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;auto
/// v=m::ConsistencyGaussianDrift(n::zeros({2,2}),n::ones({2}));
/// @endcode
/// @par References
/// - Song, Dhariwal, Chen and Sutskever (2023), Consistency Models,
/// https://arxiv.org/abs/2303.01469. This compact Gaussian-teacher fixture follows the original
/// repository example.
/// @see InitializeDiffusionMlp(), DiffusionMlp()
inline Tensor ConsistencyGaussianDrift(const Tensor& x, const Tensor& time) {
  diffusion_component_detail::Points(x, time, double(.01f), 2.);
  const Tensor t = reshape(time, {-1, 1});
  return div(mul(t, sub(x, tensor({{.3f, -.2f}}))), add(tensor({{.49f, .25f}}), square(t)));
}

/// @brief Apply one descending-time Gaussian Euler teacher step.
///
/// No target model or stop-gradient is hidden here. Caller owns distillation objective, target
/// detachment and post-optimizer EMA. O(N).
///
/// @param x Finite floating CPU[N,2].
/// @param time Same-dtype CPU[N] in[Float32(.01),2].
/// @param previous Same-dtype CPU[N], Float32(.01)<=previous<=time.
/// @return Live single Euler teacher state x+(previous-time)*drift.
/// @throws std::invalid_argument Invalid finite CPU geometry, dtype or domain.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;auto
/// v=m::ConsistencyEulerTeacher(n::zeros({2,2}),n::ones({2}),n::full({2},.5));
/// @endcode
/// @par References
/// - Song, Dhariwal, Chen and Sutskever (2023), Consistency Models,
/// https://arxiv.org/abs/2303.01469. This compact Gaussian-teacher fixture follows the original
/// repository example.
/// @see InitializeDiffusionMlp(), DiffusionMlp()
inline Tensor ConsistencyEulerTeacher(const Tensor& x, const Tensor& time, const Tensor& previous) {
  diffusion_component_detail::Points(x, previous, double(.01f), 2.);
  auto a = time.to_vector<double>(), b = previous.to_vector<double>();
  for (std::size_t i = 0; i < a.size(); ++i)
    if (b[i] > a[i])
      throw std::invalid_argument("consistency teacher time order");
  return add(x, mul(reshape(sub(previous, time), {-1, 1}), ConsistencyGaussianDrift(x, time)));
}

/// @brief Evaluate the compact velocity field or analytic ring oracle.
///
/// Oracle computes stable softmax weights of eight radius-two, deviation-.15 components. Component
/// means are source Float32. No fitting, RNG, mutation or detachment. Work O(N*1250) learned,
/// O(N*8) oracle.
///
/// @param p Read-only initialized 3->32->32->2 store matching x, or nullptr for the fixed ring
/// oracle.
/// @param x Finite floating CPU[N,2].
/// @param time Same-dtype finite floating CPU[N]; fitting/sampling convention is[0,1], diagnostic
/// extrapolation is allowed.
/// @return Owning live CPU[N,2] learned or analytic mixture velocity.
/// @throws std::invalid_argument Invalid finite CPU geometry, dtype or domain.
/// @throws std::out_of_range A required explicitly initialized predictor leaf is missing.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;auto
/// v=m::FlowMatchingVelocity(nullptr,n::zeros({2,2}),n::full({2},.5));
/// @endcode
/// @par References
/// - Lipman, Chen, Ben-Hamu, Nickel and Le (2022), Flow Matching for Generative Modeling,
/// https://arxiv.org/abs/2210.02747. The original fixture uses an eight-Gaussian ring and a compact
/// velocity MLP.
/// @see InitializeDiffusionMlp(), DiffusionMlp()
inline Tensor FlowMatchingVelocity(const ParameterStore* p, const Tensor& x, const Tensor& time) {
  diffusion_component_detail::Points(
      x, time, -std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity());
  const Tensor t = reshape(time, {-1, 1});
  if (!p) {
    const Tensor        one_minus = 1.0 - t;
    const Tensor        variance  = add(square(one_minus), mul(square(t), .0225));
    const Tensor        gain      = div(sub(mul(t, .0225), one_minus), variance);
    std::vector<Tensor> logits, vs;
    for (int k = 0; k < 8; ++k) {
      const Tensor mu = from_vector(
          std::vector<float>{
              float(2 * std::cos(2 * std::acos(-1.) * k / 8)),
              float(2 * std::sin(2 * std::acos(-1.) * k / 8))},
          {1, 2});
      const Tensor delta = sub(x, mul(t, mu));
      logits.push_back(div(mul(sum(square(delta), {1}, true), -.5), variance));
      vs.push_back(expand_dims(add(mu, mul(gain, delta)), 1));
    }
    const Tensor weights = expand_dims(softmax(concat(logits, 1), 1), 2);
    return sum(mul(weights, concat(vs, 1)), {1});
  }
  const Tensor input = concat({x, t}, 1);
  return DiffusionMlp(*p, input).value;
}

/// @brief Owning live Euler or RK4 stage endpoints, with no full trajectory.
/// @par Example
/// @code
/// dm::nn::models::FlowMatchingStepOutput endpoints;
/// @endcode
/// @see InitializeDiffusionMlp()
/// @par References
/// - Ho et al. (2020), Denoising Diffusion Probabilistic Models, https://arxiv.org/abs/2006.11239 ;
/// Song et al. (2021), Score-Based Generative Modeling through SDEs,
/// https://arxiv.org/abs/2011.13456 . Source-rounded fixture details are specified by each function
/// contract.
struct FlowMatchingStepOutput {
  Tensor              value;      ///< Live CPU[N,2] integrated point state.
  std::vector<Tensor> states;     ///< One or four live stage states.
  std::vector<Tensor> velocities; ///< One or four live stage velocities.
};

/// @brief Apply one explicit Euler or RK4 flow-matching step.
///
/// Single explicit forward-time step; no full trajectory, fitting or RNG. Applicable
/// state/time/leaf/h paths remain live. Work one or four velocity evaluations.
///
/// @param p Read-only initialized store or nullptr for analytic ring oracle.
/// @param x Finite floating CPU[N,2].
/// @param time Matching finite floating CPU[N]; extrapolation is allowed.
/// @param h Finite nonnegative floating CPU scalar; no upper endpoint restriction for diagnostic
/// probes.
/// @param rk4 True for classical RK4, false for Euler.
/// @return Owning live next state, stage states and velocities.
/// @throws std::invalid_argument Invalid finite CPU geometry, dtype or domain.
/// @throws std::out_of_range A required explicitly initialized predictor leaf is missing.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;auto
/// y=m::FlowMatchingStep(nullptr,n::zeros({2,2}),n::full({2},.5),n::scalar(.1),true);
/// @endcode
/// @par References
/// - Lipman, Chen, Ben-Hamu, Nickel and Le (2022), Flow Matching for Generative Modeling,
/// https://arxiv.org/abs/2210.02747. The original fixture uses an eight-Gaussian ring and a compact
/// velocity MLP.
/// @see InitializeDiffusionMlp(), DiffusionMlp()
inline FlowMatchingStepOutput FlowMatchingStep(
    const ParameterStore* p,
    const Tensor&         x,
    const Tensor&         time,
    const Tensor&         h,
    bool                  rk4) {
  diffusion_component_detail::Scale(h);
  diffusion_component_detail::Points(
      x, time, -std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity());
  const Tensor a = FlowMatchingVelocity(p, x, time);
  if (!rk4)
    return {add(x, mul(h, a)), {x}, {a}};
  const Tensor half  = mul(h, .5);
  const Tensor xb    = add(x, mul(half, a));
  const Tensor b     = FlowMatchingVelocity(p, xb, add(time, half));
  const Tensor xc    = add(x, mul(half, b));
  const Tensor c     = FlowMatchingVelocity(p, xc, add(time, half));
  const Tensor xd    = add(x, mul(h, c));
  const Tensor d     = FlowMatchingVelocity(p, xd, add(time, h));
  const Tensor value = add(x, mul(div(h, 6.0), add(add(a, d), mul(add(b, c), 2.0))));
  return {value, {x, xb, xc, xd}, {a, b, c, d}};
}

/// @brief Owning live variance-exploding score-field endpoints.
/// @par Example
/// @code
/// dm::nn::models::ScoreSdeFieldOutput endpoints;
/// @endcode
/// @see InitializeDiffusionMlp()
/// @par References
/// - Ho et al. (2020), Denoising Diffusion Probabilistic Models, https://arxiv.org/abs/2006.11239 ;
/// Song et al. (2021), Score-Based Generative Modeling through SDEs,
/// https://arxiv.org/abs/2011.13456 . Source-rounded fixture details are specified by each function
/// contract.
struct ScoreSdeFieldOutput {
  Tensor score;             ///< Live image-shaped score.
  Tensor epsilon;           ///< Live image-shaped noise prediction.
  Tensor sigma;             ///< Live CPU[N,1,1,1] VE noise scale.
  Tensor g_squared;         ///< Live CPU[N,1,1,1] diffusion coefficient squared.
  Tensor reverse_drift;     ///< Live image-shaped reverse drift.
  Tensor probability_drift; ///< Live image-shaped half reverse drift.
};

/// @brief Evaluate the compact variance-exploding reverse field.
///
/// Uses sigma=.05*exp(Float32(log20)*t), g²=Float32(2log20)*sigma². Learned conditioning
/// is15*(1-t); oracle mean.3, variance.49. No fitting, RNG or detachment; O(N*5504) learned.
///
/// @param p Read-only initialized 72->32->32->64 store, or nullptr for scalar Gaussian oracle.
/// @param state Finite floating CPU[N,8,8,1].
/// @param physical_time Matching floating CPU[N] in[0,1], endpoint tolerance1e-7.
/// @return Owning live score, epsilon, sigma, g² and reverse/probability drifts.
/// @throws std::invalid_argument Invalid finite CPU geometry, dtype or domain.
/// @throws std::out_of_range A required explicitly initialized predictor leaf is missing.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;auto
/// y=m::ScoreSdeField(nullptr,n::zeros({1,8,8,1}),n::full({1},.5));
/// @endcode
/// @par References
/// - Song et al. (2020), Score-Based Generative Modeling through Stochastic Differential Equations,
/// https://arxiv.org/abs/2011.13456. The compact original fixture uses a VE field and fixed-step
/// EM/RK4.
/// @see InitializeDiffusionMlp(), DiffusionMlp()
inline ScoreSdeFieldOutput
ScoreSdeField(const ParameterStore* p, const Tensor& state, const Tensor& physical_time) {
  diffusion_component_detail::Image(state);
  diffusion_detail::Floating(physical_time, 1);
  if (physical_time.dim() != 1 || physical_time.size(0) != state.size(0) ||
      physical_time.dtype() != state.dtype())
    throw std::invalid_argument("score SDE time geometry/dtype");
  for (double t : physical_time.to_vector<double>())
    if (t < 0 || t > 1. + 1e-7)
      throw std::invalid_argument("score SDE time range");
  const Tensor time  = reshape(physical_time, {-1, 1, 1, 1});
  const Tensor sigma = mul(exp(mul(time, double(float(std::log(20.))))), .05);
  const Tensor g2    = mul(square(sigma), double(float(2 * std::log(20.))));
  Tensor       score, epsilon;
  if (!p) {
    score   = div(sub(scalar_like(.3, state), state), add(square(sigma), .49));
    epsilon = neg(mul(sigma, score));
  } else {
    const Tensor level = mul(1.0 - physical_time, 15.0);
    epsilon            = DiffusionImagePrediction(*p, state, level, 15.f).epsilon;
    score              = neg(div(epsilon, sigma));
  }
  const Tensor reverse = mul(g2, score);
  return {score, epsilon, sigma, g2, reverse, mul(reverse, .5)};
}

/// @brief Owning live reverse-SDE and probability-flow single-step endpoints.
/// @par Example
/// @code
/// dm::nn::models::ScoreSdeStepOutput endpoints;
/// @endcode
/// @see InitializeDiffusionMlp()
/// @par References
/// - Ho et al. (2020), Denoising Diffusion Probabilistic Models, https://arxiv.org/abs/2006.11239 ;
/// Song et al. (2021), Score-Based Generative Modeling through SDEs,
/// https://arxiv.org/abs/2011.13456 . Source-rounded fixture details are specified by each function
/// contract.
struct ScoreSdeStepOutput {
  ScoreSdeFieldOutput field;            ///< Owning initial live field endpoints.
  Tensor              em_mean;          ///< Live image-shaped EM mean.
  Tensor              em_value;         ///< Live image-shaped EM sample.
  Tensor              ode_value;        ///< Live image-shaped RK4 probability-flow endpoint.
  std::vector<Tensor> ode_stage_states; ///< Four live RK4 image-shaped stage states.
  std::vector<Tensor> ode_stage_drifts; ///< Four live RK4 image-shaped probability drifts.
};

/// @brief Apply one reverse EM and probability-flow RK4 step.
///
/// Reverse physical time decreases. No sampling loop, RNG, fitting, mutation or no-grad guard; four
/// field evaluations. Applicable paths stay live.
///
/// @param p Read-only initialized store or nullptr for Gaussian oracle.
/// @param state Finite floating CPU[N,8,8,1].
/// @param time Matching finite floating CPU[N]; extrapolation is allowed.
/// @param h Finite nonnegative floating CPU scalar, <=time+1e-7 for every row.
/// @param innovation Finite explicit innovation matching state shape/dtype.
/// @return Owning live EM mean/sample and RK4 probability-flow state with all stages.
/// @throws std::invalid_argument Invalid finite CPU geometry, dtype or domain.
/// @throws std::out_of_range A required explicitly initialized predictor leaf is missing.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;auto x=n::zeros({1,8,8,1});auto
/// y=m::ScoreSdeStep(nullptr,x,n::full({1},.5),n::scalar(.1),x);
/// @endcode
/// @par References
/// - Song et al. (2020), Score-Based Generative Modeling through Stochastic Differential Equations,
/// https://arxiv.org/abs/2011.13456. The compact original fixture uses a VE field and fixed-step
/// EM/RK4.
/// @see InitializeDiffusionMlp(), DiffusionMlp()
inline ScoreSdeStepOutput ScoreSdeStep(
    const ParameterStore* p,
    const Tensor&         state,
    const Tensor&         time,
    const Tensor&         h,
    const Tensor&         innovation) {
  diffusion_component_detail::Scale(h);
  diffusion_detail::Same(state, innovation);
  diffusion_detail::Floating(time, 1);
  if (time.dim() != 1 || time.size(0) != state.size(0) || time.dtype() != state.dtype())
    throw std::invalid_argument("score step time geometry/dtype");
  for (double t : time.to_vector<double>())
    if (h.item<double>() > t + 1e-7)
      throw std::invalid_argument("reverse step past endpoint");
  const auto   a    = ScoreSdeField(p, state, time);
  const Tensor mean = add(state, mul(h, a.reverse_drift));
  const Tensor em   = add(mean, mul(sqrt(mul(h, a.g_squared)), innovation));
  const Tensor half = mul(h, .5);
  const Tensor xb   = add(state, mul(half, a.probability_drift));
  const auto   b    = ScoreSdeField(p, xb, sub(time, half));
  const Tensor xc   = add(state, mul(half, b.probability_drift));
  const auto   c    = ScoreSdeField(p, xc, sub(time, half));
  const Tensor xd   = add(state, mul(h, c.probability_drift));
  const auto   d    = ScoreSdeField(p, xd, sub(time, h));
  const Tensor ode =
      add(state,
          mul(div(h, 6.0),
              add(add(a.probability_drift, d.probability_drift),
                  mul(add(b.probability_drift, c.probability_drift), 2.0))));
  return {
      a,
      mean,
      em,
      ode,
      {state, xb, xc, xd},
      {a.probability_drift, b.probability_drift, c.probability_drift, d.probability_drift}};
}

/// @brief Owning live class-conditioned predictor endpoints.
/// @par Example
/// @code
/// dm::nn::models::DiffusionGuidanceOutput endpoints;
/// @endcode
/// @see InitializeDiffusionMlp()
/// @par References
/// - Ho et al. (2020), Denoising Diffusion Probabilistic Models, https://arxiv.org/abs/2006.11239 ;
/// Song et al. (2021), Score-Based Generative Modeling through SDEs,
/// https://arxiv.org/abs/2011.13456 . Source-rounded fixture details are specified by each function
/// contract.
struct DiffusionGuidanceOutput {
  Tensor prediction;     ///< Live logits[N,10] or epsilon[N,8,8,1].
  Tensor time_embedding; ///< Source-rounded CPU[N,8] time features.
  Tensor hidden1;        ///< First live CPU[N,32] SiLU endpoint.
  Tensor hidden2;        ///< Second live CPU[N,32] SiLU endpoint.
  Tensor class_onehot;   ///< Structural CPU[N,11] class/NULL indicators.
};

/// @brief Initialize source classifier or conditional-predictor leaves.
///
/// Explicit source-order initialization through InitializeDiffusionMlp; existing valid leaves
/// reused, no optimizer or fitting. Caller serializes mutable stores.
///
/// @param p Mutable caller-owned Float32/64 parameter store.
/// @param classifier True:72->32->32->10 logits; false:83->32->32->64 epsilon.
/// @throws std::invalid_argument Invalid finite CPU geometry, dtype or domain.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore p(14007);m::CreateGuidance(p,false);
/// @endcode
/// @par References
/// - Dhariwal and Nichol (2021), Diffusion Models Beat GANs on Image Synthesis,
/// https://arxiv.org/abs/2105.05233; Ho and Salimans (2022), Classifier-Free Diffusion Guidance,
/// https://arxiv.org/abs/2207.12598. These are compact original-example MLPs.
/// @see InitializeDiffusionMlp(), DiffusionMlp()
inline void CreateGuidance(ParameterStore& p, bool classifier) {
  InitializeDiffusionMlp(p, classifier ? 72 : 83, classifier ? 10 : 64);
}

/// @brief Evaluate the class-conditioned predictor or noisy-image classifier.
///
/// Integer conditions are structural; source Float32 phase features. No sampling, fitting, RNG,
/// mutation or detachment; work O(N*parameter count).
///
/// @param p Read-only explicitly initialized guidance store matching state dtype.
/// @param state Finite floating CPU[N,8,8,1].
/// @param time Int32/64 CPU[N] in[0,31].
/// @param label Int32/64 CPU[N], classifier0..9, conditional0..10(NULL).
/// @param classifier True for10 logits; false for class-conditioned epsilon.
/// @return Owning live prediction, time/one-hot features and hidden endpoints.
/// @throws std::invalid_argument Invalid finite CPU geometry, dtype or domain.
/// @throws std::out_of_range A required explicitly initialized predictor leaf is missing.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;n::ParameterStore
/// p(14007);m::CreateGuidance(p,false);auto
/// y=m::GuidancePredictor(p,n::zeros({1,8,8,1}),n::zeros({1},n::DType::Int32),n::zeros({1},n::DType::Int32),false);
/// @endcode
/// @par References
/// - Dhariwal and Nichol (2021), Diffusion Models Beat GANs on Image Synthesis,
/// https://arxiv.org/abs/2105.05233; Ho and Salimans (2022), Classifier-Free Diffusion Guidance,
/// https://arxiv.org/abs/2207.12598. These are compact original-example MLPs.
/// @see InitializeDiffusionMlp(), DiffusionMlp()
inline DiffusionGuidanceOutput GuidancePredictor(
    const ParameterStore& p,
    const Tensor&         state,
    const Tensor&         time,
    const Tensor&         label,
    bool                  classifier) {
  diffusion_component_detail::Image(state);
  diffusion_detail::Time(time, state.size(0), 32);
  diffusion_detail::Time(label, state.size(0), classifier ? 10 : 11);
  const Tensor embedding    = DiffusionContinuousTimeEmbedding(time, 31.f, state.dtype());
  const Tensor class_onehot = cast(one_hot(cast(label, DType::Int64), 11), state.dtype());
  const Tensor flat         = reshape(state, {-1, 64});
  const Tensor encoded =
      classifier ? concat({flat, embedding}, 1) : concat({flat, embedding, class_onehot}, 1);
  const auto path = DiffusionMlp(p, encoded);
  const auto h1 = path.hidden1, h2 = path.hidden2;
  Tensor     prediction = path.value;
  if (!classifier)
    prediction = reshape(prediction, {-1, 8, 8, 1});
  return {prediction, embedding, h1, h2, class_onehot};
}

/// @brief Select stable log probabilities for integer class labels.
///
/// Uses native stable log_softmax then one-hot selection. No classification fitting or mutation;
/// O(N*10).
///
/// @param logits Finite floating CPU[N,10].
/// @param label Int32/64 CPU[N] labels0..9.
/// @return Live selected-class log probabilities CPU[N].
/// @throws std::invalid_argument Invalid finite CPU geometry, dtype or domain.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;auto
/// y=m::ClassifierLogProbability(n::zeros({2,10}),n::zeros({2},n::DType::Int32));
/// @endcode
/// @par References
/// - Dhariwal and Nichol (2021), Diffusion Models Beat GANs on Image Synthesis,
/// https://arxiv.org/abs/2105.05233; Ho and Salimans (2022), Classifier-Free Diffusion Guidance,
/// https://arxiv.org/abs/2207.12598. These are compact original-example MLPs.
/// @see InitializeDiffusionMlp(), DiffusionMlp()
inline Tensor ClassifierLogProbability(const Tensor& logits, const Tensor& label) {
  diffusion_detail::Floating(logits);
  if (logits.dim() != 2 || logits.size(1) != 10)
    throw std::invalid_argument("classifier logits [N,10]");
  diffusion_detail::Time(label, logits.size(0), 10);
  return sum(
      mul(log_softmax(logits, -1), cast(one_hot(cast(label, DType::Int64), 10), logits.dtype())),
      {1});
}

/// @brief Mix an explicit classifier gradient into predicted noise.
///
/// Caller computes classifier gradient, including higher-gradient options if needed. No hidden
/// grad, fitting or sampling; O(elements+table length).
///
/// @param epsilon Finite floating CPU[N,...], rank>=2.
/// @param time Int32/64 CPU[N] schedule indices.
/// @param gradient Finite selected log-class input gradient matching epsilon shape/dtype.
/// @param scale Finite nonnegative floating CPU scalar; caller chooses strength.
/// @param schedule Explicit schedule supplying sqrt_one_minus_alpha_bar.
/// @return Live epsilon-scale*source-noise-coefficient*gradient.
/// @throws std::invalid_argument Invalid finite CPU geometry, dtype or domain.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;auto x=n::zeros({1,2});auto
/// y=m::ClassifierGuidedEpsilon(x,n::zeros({1},n::DType::Int32),x,n::scalar(1),m::CosineDiffusionSchedule());
/// @endcode
/// @par References
/// - Dhariwal and Nichol (2021), Diffusion Models Beat GANs on Image Synthesis,
/// https://arxiv.org/abs/2105.05233; Ho and Salimans (2022), Classifier-Free Diffusion Guidance,
/// https://arxiv.org/abs/2207.12598. These are compact original-example MLPs.
/// @see InitializeDiffusionMlp(), DiffusionMlp()
inline Tensor ClassifierGuidedEpsilon(
    const Tensor&                  epsilon,
    const Tensor&                  time,
    const Tensor&                  gradient,
    const Tensor&                  scale,
    const DiffusionScheduleOutput& schedule) {
  diffusion_detail::Same(epsilon, gradient);
  diffusion_component_detail::Scale(scale);
  diffusion_detail::Time(time, epsilon.size(0));
  return sub(
      epsilon,
      mul(mul(scale, DiffusionExtract(schedule.sqrt_one_minus_alpha_bar, time, int(epsilon.dim()))),
          gradient));
}

/// @brief Mix conditional and NULL predictions with explicit strength.
///
/// Scale0 returns null and scale1 returns conditional within floating arithmetic; no classifier,
/// RNG, fitting, detachment or mutation. O(elements).
///
/// @param conditional Finite floating CPU[N,...], rank>=2.
/// @param null_prediction Matching finite floating unconditional prediction.
/// @param scale Finite nonnegative floating CPU scalar.
/// @return Live null+scale*(conditional-null).
/// @throws std::invalid_argument Invalid finite CPU geometry, dtype or domain.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;auto x=n::zeros({1,2});auto
/// y=m::ClassifierFreeEpsilon(x,x,n::scalar(1));
/// @endcode
/// @par References
/// - Dhariwal and Nichol (2021), Diffusion Models Beat GANs on Image Synthesis,
/// https://arxiv.org/abs/2105.05233; Ho and Salimans (2022), Classifier-Free Diffusion Guidance,
/// https://arxiv.org/abs/2207.12598. These are compact original-example MLPs.
/// @see InitializeDiffusionMlp(), DiffusionMlp()
inline Tensor ClassifierFreeEpsilon(
    const Tensor& conditional,
    const Tensor& null_prediction,
    const Tensor& scale) {
  diffusion_detail::Same(conditional, null_prediction);
  diffusion_component_detail::Scale(scale);
  return add(null_prediction, mul(scale, sub(conditional, null_prediction)));
}

/// @brief Live unclipped latent posterior endpoints, without a sampler or model.
/// @par Example
/// @code
/// dm::nn::models::LatentDiffusionReverseOutput endpoints;
/// @endcode
/// @see InitializeDiffusionMlp()
/// @par References
/// - Ho et al. (2020), Denoising Diffusion Probabilistic Models, https://arxiv.org/abs/2006.11239 ;
/// Song et al. (2021), Score-Based Generative Modeling through SDEs,
/// https://arxiv.org/abs/2011.13456 . Source-rounded fixture details are specified by each function
/// contract.
struct LatentDiffusionReverseOutput {
  Tensor value;    ///< Next latent state, same shape/dtype as input.
  Tensor mean;     ///< Live posterior mean.
  Tensor variance; ///< Source Float32 broadcast posterior variance.
  Tensor x0;       ///< Unclipped clean-latent estimate.
};
/// @brief Predict noise on two-dimensional latent codes with source time features.
/// @param parameters Explicitly initialized 10->32->32->2 floating store.
/// @param state Finite CPU[N,2] Float32/64 matching the store dtype.
/// @param time Int32/64 CPU[N] in[0,31].
/// @return Owning live CPU[N,2] noise prediction; no RNG or mutation.
/// @throws std::invalid_argument Invalid geometry, indices, dtype or finite values.
/// @throws std::out_of_range A required explicitly initialized predictor leaf is missing.
/// @par References
/// - Ho et al. (2020), Denoising Diffusion Probabilistic Models, https://arxiv.org/abs/2006.11239 ;
/// Song et al. (2021), Score-Based Generative Modeling through SDEs,
/// https://arxiv.org/abs/2011.13456 . Source-rounded fixture details are specified by each function
/// contract.
inline Tensor
LatentDiffusionEpsilon(const ParameterStore& parameters, const Tensor& state, const Tensor& time) {
  diffusion_detail::Floating(state);
  if (state.dim() != 2 || state.size(1) != 2)
    throw std::invalid_argument("latent state [N,2]");
  diffusion_detail::Time(time, state.size(0), 32);
  auto embedding = DiffusionContinuousTimeEmbedding(time, 31.f, state.dtype());
  return DiffusionMlp(parameters, concat({state, embedding}, 1)).value;
}
/// @brief Compute one unclipped latent DDPM posterior transition.
/// @param state Finite CPU[N,2] floating latent state.
/// @param time Int32/64 CPU[N] schedule indices.
/// @param epsilon Finite prediction matching state shape/dtype.
/// @param innovation Explicit finite noise matching state shape/dtype.
/// @param schedule Caller-owned source-rounded posterior and noising tables.
/// @return Owning live posterior sample/mean/variance and unclipped clean estimate.
/// @throws std::invalid_argument Invalid geometry, dtype, index or coefficients.
/// No fitting, RNG, clipping, mutation or hidden no-grad. Work O(N+table length).
/// @par References
/// - Ho et al. (2020), Denoising Diffusion Probabilistic Models, https://arxiv.org/abs/2006.11239 ;
/// Song et al. (2021), Score-Based Generative Modeling through SDEs,
/// https://arxiv.org/abs/2011.13456 . Source-rounded fixture details are specified by each function
/// contract.
inline LatentDiffusionReverseOutput LatentDiffusionReverse(
    const Tensor&                  state,
    const Tensor&                  time,
    const Tensor&                  epsilon,
    const Tensor&                  innovation,
    const DiffusionScheduleOutput& schedule) {
  diffusion_detail::Same(state, epsilon);
  diffusion_detail::Same(state, innovation);
  if (state.dim() != 2 || state.size(1) != 2)
    throw std::invalid_argument("latent state [N,2]");
  diffusion_detail::Time(time, state.size(0));
  auto signal = DiffusionExtract(schedule.sqrt_alpha_bar, time, 2),
       sigma  = DiffusionExtract(schedule.sqrt_one_minus_alpha_bar, time, 2);
  diffusion_detail::Positive(signal, true);
  diffusion_detail::Positive(sigma);
  auto x0       = div(sub(state, mul(sigma, epsilon)), signal);
  auto c1       = DiffusionExtract(schedule.posterior_coefficient1, time, 2),
       c2       = DiffusionExtract(schedule.posterior_coefficient2, time, 2);
  auto variance = DiffusionExtract(schedule.posterior_variance, time, 2);
  diffusion_detail::Positive(variance);
  auto mean  = add(mul(c1, x0), mul(c2, state));
  auto value = add(mean, mul(sqrt(variance), innovation));
  gan_detail::Finite(value);
  return {value, mean, variance, x0};
}

/// @brief Owning detached periodic-waveform fixture with exact source provenance.
/// @par Example
/// @code
/// dm::nn::models::DiffusionSignalData endpoints;
/// @endcode
/// @see InitializeDiffusionMlp()
/// @par References
/// - Ho et al. (2020), Denoising Diffusion Probabilistic Models, https://arxiv.org/abs/2006.11239 ;
/// Song et al. (2021), Score-Based Generative Modeling through SDEs,
/// https://arxiv.org/abs/2011.13456 . Source-rounded fixture details are specified by each function
/// contract.
struct DiffusionSignalData {
  Tensor signals;    ///< Float32 CPU[N,8,8,1], ordered 64-point waveforms (not spatial images).
  Tensor parameters; ///< Double CPU[N,3]: frequency, amplitude, phase in radians.
  Tensor source_ids; ///< Int32 CPU[N] source IDs, seed*1000+row with uint32 wrapping.
};
/// @brief Generate the original three-word-per-row periodic diffusion fixture.
/// Uses a local MT19937 seeded by seed; consumes three midpoint uniforms per row.
/// Frequencies are1..4, amplitudes [.6,1), phases(0,2pi). Double harmonic
/// evaluation rounds each observation once to Float32. No global RNG, fitting,
/// mutable cache or gradient path; results outlive all locals. O(64*rows) work.
/// @param rows Positive row count, at most1000000.
/// @param seed UInt32 local MT seed; default18001, matching the fitting pool.
/// @return Owning signals, Double generating parameters and source IDs.
/// @throws std::invalid_argument Row count outside[1,1000000].
/// @par Example
/// @code
/// auto data=dm::nn::models::PeriodicDiffusionSignals(3,18001);
/// @endcode
/// @see DiffusionNoised(), DdpmEpsilon()
/// @par References
/// - Ho et al. (2020), Denoising Diffusion Probabilistic Models, https://arxiv.org/abs/2006.11239 ;
/// Song et al. (2021), Score-Based Generative Modeling through SDEs,
/// https://arxiv.org/abs/2011.13456 . Source-rounded fixture details are specified by each function
/// contract.
inline DiffusionSignalData PeriodicDiffusionSignals(std::int64_t rows, std::uint32_t seed = 18001) {
  if (rows < 1 || rows > 1000000)
    throw std::invalid_argument("periodic signal row count");
  std::vector<double>       parameters(std::size_t(rows) * 3);
  std::vector<float>        signals(std::size_t(rows) * 64);
  std::vector<std::int32_t> ids(static_cast<std::size_t>(rows));
  std::mt19937              rng(seed);
  const double              pi = std::acos(-1.);
  for (std::int64_t b = 0; b < rows; ++b) {
    const double u = (double(rng()) + .5) / 4294967296., v = (double(rng()) + .5) / 4294967296.,
                 w         = (double(rng()) + .5) / 4294967296.;
    const int    frequency = 1 + int(4 * u);
    const double amplitude = .6 + .4 * v, phase = 2 * pi * w;
    parameters[b * 3]     = frequency;
    parameters[b * 3 + 1] = amplitude;
    parameters[b * 3 + 2] = phase;
    ids[b]                = std::int32_t(seed * 1000 + std::uint32_t(b));
    for (int j = 0; j < 64; ++j)
      signals[b * 64 + j] = float(
          .9 * amplitude / 1.25 *
          (std::sin(2 * pi * frequency * j / 64 + phase) +
           .25 * std::sin(2 * pi * 3 * frequency * j / 64 + 2 * phase)));
  }
  return {
      from_vector(signals, {rows, 8, 8, 1}),
      from_vector(parameters, {rows, 3}),
      from_vector(ids, {rows})};
}

/// @brief Render the original periodic-waveform RGB tile grid without writing files.
/// Each tile is256x112 pixels with background250, axis210 and blue(30,90,190)
/// source-rounded polyline pixels. Display clips values to[-1.5,1.5] only;
/// input observations are unchanged. Output is detached, owning Float32 RGB
/// byte-scale[H,W,3], independent of all input/model lifetimes. No RNG/fitting.
/// Work/storage O(rows*256*112). Caller supplies serialization for shared output files.
/// @param images Finite Float32/64 CPU[N,8,8,1], 1<=N<=1024, ordered waveform samples.
/// @param columns Grid columns in[1,64], default8; final row padded to full width.
/// @return Detached RGB byte-scale Float32 CPU[ceil(N/columns)*112,columns*256,3].
/// @throws std::invalid_argument Invalid geometry, nonfinite values or row/column limits.
/// @par Example
/// @code
/// auto
/// grid=dm::nn::models::PeriodicSignalGrid(dm::nn::models::PeriodicDiffusionSignals(8).signals);
/// @endcode
/// @see PeriodicDiffusionSignals(), dm::nn::save_png()
/// @par References
/// - Ho et al. (2020), Denoising Diffusion Probabilistic Models, https://arxiv.org/abs/2006.11239 ;
/// Song et al. (2021), Score-Based Generative Modeling through SDEs,
/// https://arxiv.org/abs/2011.13456 . Source-rounded fixture details are specified by each function
/// contract.
inline Tensor PeriodicSignalGrid(const Tensor& images, int columns = 8) {
  diffusion_component_detail::Image(images);
  if (images.size(0) > 1024 || columns < 1 || columns > 64)
    throw std::invalid_argument("signal grid rows/columns");
  NoGradGuard guard;
  const int   n = int(images.size(0)), H = ((n + columns - 1) / columns) * 112, W = columns * 256;
  const auto  v = images.contiguous().to_vector<float>();
  std::vector<float> out(std::size_t(H) * W * 3, 250.f);
  auto               set = [&](int y, int x, float r, float g, float b) {
    const std::size_t k = (std::size_t(y) * W + x) * 3;
    out[k] = r, out[k + 1] = g, out[k + 2] = b;
  };
  for (int b = 0; b < n; ++b) {
    const int oy = b / columns * 112, ox = b % columns * 256;
    for (int x = 0; x < 256; ++x)
      set(oy + 56, ox + x, 210, 210, 210);
    for (int segment = 0; segment < 63; ++segment) {
      const int y0 = int(
                    std::floor(56 - 32 * std::clamp(double(v[b * 64 + segment]), -1.5, 1.5) + .5)),
                y1    = int(std::floor(
                    56 - 32 * std::clamp(double(v[b * 64 + segment + 1]), -1.5, 1.5) + .5));
      const int steps = std::max(4, std::abs(y1 - y0));
      for (int k = 0; k <= steps; ++k) {
        const int x = int(std::floor(4 * segment + 4 * double(k) / steps + .5)),
                  y = int(std::floor(y0 + (y1 - y0) * double(k) / steps + .5));
        set(oy + y, ox + x, 30, 90, 190);
      }
    }
  }
  return from_vector(out, {H, W, 3});
}

} // namespace datamunge::nn::models
