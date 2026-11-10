#include <gtest/gtest.h>

#include <datamunge/nn/autograd.hpp>
#include <datamunge/nn/training.hpp>
#include <datamunge/nn/gradcheck.hpp>
#include <datamunge/nn/optim.hpp>
#include <datamunge/nn/posterior_collapse.hpp>
namespace n = datamunge::nn;
namespace m = n::models;
namespace {
n::Tensor Sentences() {
  return n::from_vector(std::vector<int>{3, 5, 8, 11, 13, 2, 0, 4, 7, 10, 12, 16, 2, 0}, {2, 7});
}
} // namespace
TEST(NnSentenceVae, EighteenOwningEndpointsSourceCensusAndLoss) {
  for (auto type : {n::DType::Float32, n::DType::Float64}) {
    n::ParameterStore p(3109, type);
    auto out = m::SequenceVariationalAutoencoder(p, Sentences(), n::zeros({2, 4}, type));
    EXPECT_EQ(p.size(), 22u);
    EXPECT_EQ(p.count(), 3243u);
    EXPECT_EQ(out.encoder_states.shape(), (n::Shape{2, 7, 16}));
    EXPECT_EQ(out.decoder_states.shape(), (n::Shape{2, 7, 16}));
    EXPECT_EQ(out.logits.shape(), (n::Shape{2, 7, 19}));
    EXPECT_EQ(out.predicted_ids.dtype(), n::DType::Int32);
    EXPECT_TRUE(n::equal_bits(out.z, out.mu));
    EXPECT_TRUE(n::equal_bits(out.loss, n::mean(out.nll + out.kl, {0})));
    EXPECT_EQ(out.valid_tokens.to_vector<double>(), (std::vector<double>{6, 6}));
    EXPECT_EQ(n::select(out.token_nll, 1, 6).to_vector<double>(), (std::vector<double>{0, 0}));
    auto fixed = m::SequenceTeacher(p, out.z, out.decoder_inputs);
    EXPECT_TRUE(n::equal_bits(fixed.logits, out.logits));
    EXPECT_TRUE(n::equal_bits(fixed.probability, out.probability));
    for (const auto& t :
         {out.encoder_state,
          out.encoder_states,
          out.mu,
          out.log_variance,
          out.z,
          out.decoder_initial,
          out.decoder_states,
          out.logits,
          out.probability,
          out.mask,
          out.token_nll,
          out.nll,
          out.kl,
          out.correct,
          out.valid_tokens,
          out.loss}) {
      EXPECT_EQ(t.dtype(), type);
      for (auto v : t.to_vector<double>())
        EXPECT_TRUE(std::isfinite(v));
    }
    const auto prefixes = m::SequenceDecoderInputs(n::cast(Sentences(), n::DType::Int64));
    EXPECT_EQ(
        prefixes.to_vector<int>(),
        (std::vector<int>{1, 3, 5, 8, 11, 13, 2, 1, 4, 7, 10, 12, 16, 2}));
    auto strided = n::transpose(n::transpose(Sentences(), {1, 0}).contiguous(), {1, 0});
    EXPECT_TRUE(
        n::equal_bits(
            m::SequenceVariationalAutoencoder(p, strided, n::zeros({2, 4}, type)).probability,
            out.probability));
  }
}
TEST(NnSentenceVae, All3243LeafDerivativesAndNoiseSecondDerivatives) {
  const auto        type = n::DType::Float64;
  n::ParameterStore p(3109, type);
  const auto        x = Sentences(), noise = n::full({2, 4}, .3, type);
  const auto        value = [&] {
    return m::AnnealedSequenceObjective(m::SequenceVariationalAutoencoder(p, x, noise), .37);
  };
  const auto original_loss = value();
  const auto gradients     = n::grad(original_loss, p.parameters());
  ASSERT_EQ(gradients.size(), 22u);
  std::vector<double> analytical;
  for (const auto& g : gradients) {
    auto v = g.to_vector<double>();
    analytical.insert(analytical.end(), v.begin(), v.end());
  }
  const auto original = p.flatten();
  ASSERT_EQ(original.size(), 3243u);
  for (std::size_t j = 0; j < original.size(); ++j) {
    auto a = original;
    a[j] += 1e-6;
    p.unflatten(a);
    const auto plus = value().item();
    a[j] -= 2e-6;
    p.unflatten(a);
    const auto minus = value().item();
    EXPECT_NEAR((plus - minus) / 2e-6, analytical[j], 2e-8) << "scalar " << j;
  }
  p.unflatten(original);
  auto              eps = n::full({2, 4}, .3, type).requires_grad_();
  n::TensorFunction f   = [&](const auto& a) {
    return m::SequenceVariationalAutoencoder(p, x, a[0]).loss;
  };
  EXPECT_TRUE(n::gradcheck(f, {eps}, {.atol = 1e-8, .rtol = 1e-5}).ok);
  EXPECT_TRUE(n::gradgradcheck(f, {eps}, {.atol = 1e-8, .rtol = 1e-5}).ok);
}
TEST(NnSentenceVae, ExactPadInvarianceAndUnusedEmbeddingRows) {
  n::ParameterStore p(3109, n::DType::Float64);
  auto       out = m::SequenceVariationalAutoencoder(p, Sentences(), n::zeros({2, 4}, p.dtype()));
  const auto padded = n::concat({Sentences(), n::zeros({2, 1}, n::DType::Int32)}, 1);
  EXPECT_TRUE(n::equal_bits(m::SequenceEncoder(p, padded).state, out.encoder_state));
  auto zero = m::SequenceEncoder(p, n::zeros({2, 8}, n::DType::Int32));
  for (auto v : zero.states.to_vector<double>())
    EXPECT_EQ(v, 0);
  const auto gradients = n::grad(out.loss, p.parameters());
  const auto names     = p.names();
  for (std::size_t j = 0; j < names.size(); ++j) {
    if (names[j] == "encoder_embedding" || names[j] == "decoder_embedding") {
      for (int row : {0, names[j] == "encoder_embedding" ? 1 : 2})
        for (auto v : n::select(gradients[j], 0, row).to_vector<double>())
          EXPECT_EQ(v, 0);
    }
  }
  const auto allpad = m::SequenceVariationalAutoencoder(
      p, n::zeros({2, 7}, n::DType::Int32), n::zeros({2, 4}, p.dtype()));
  EXPECT_EQ(allpad.nll.to_vector<double>(), (std::vector<double>{0, 0}));
}
TEST(NnSentenceVae, FixedCodeCausalityAndLiveDecoderPaths) {
  n::ParameterStore p(3109, n::DType::Float64);
  auto              z        = n::full({1, 4}, .2, p.dtype()).requires_grad_();
  auto              prefixes = m::SequenceDecoderInputs(n::take_rows(Sentences(), {0}));
  auto              altered  = prefixes.to_vector<int>();
  altered[4]                 = 7;
  const auto before          = m::SequenceTeacher(p, z, prefixes);
  const auto after           = m::SequenceTeacher(p, z, n::from_vector(altered, {1, 7}));
  EXPECT_EQ(p.size(), 11u);
  EXPECT_EQ(p.count(), 1755u);
  auto   a = before.logits.to_vector<double>(), b = after.logits.to_vector<double>();
  double later = 0;
  for (int t = 0; t < 7; ++t)
    for (int w = 0; w < 19; ++w) {
      const auto delta = std::abs(a[t * 19 + w] - b[t * 19 + w]);
      if (t < 4)
        EXPECT_EQ(delta, 0);
      else
        later = std::max(later, delta);
    }
  EXPECT_GT(later, 1e-6);
  n::TensorFunction f = [&](const auto& v) {
    return n::sum(n::square(m::SequenceTeacher(p, v[0], prefixes).logits));
  };
  EXPECT_TRUE(n::gradcheck(f, {z}, {.atol = 1e-8, .rtol = 1e-5}).ok);
  EXPECT_TRUE(n::gradgradcheck(f, {z}, {.atol = 1e-8, .rtol = 1e-5}).ok);
}
TEST(NnSentenceVae, GreedyEosSuffixAndRawScoresRetainGradients) {
  n::ParameterStore p(3109, n::DType::Float64);
  auto              z = n::full({2, 4}, .2, p.dtype()).requires_grad_();
  m::SequenceGenerate(p, z);
  std::vector<double> bias(19, -20);
  bias[2] = 20;
  p.set("token_output/b", n::from_vector(bias));
  const auto result = m::SequenceGenerate(p, z);
  EXPECT_EQ(p.size(), 11u);
  EXPECT_EQ(p.count(), 1755u);
  EXPECT_EQ(
      result.ids.to_vector<int>(), (std::vector<int>{2, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0}));
  EXPECT_EQ(result.ended.to_vector<int>(), (std::vector<int>{1, 1}));
  EXPECT_GT(n::max(n::abs(n::grad(n::sum(result.logits), {z})[0])).item(), 0);
  EXPECT_FALSE(result.ids.requires_grad());
  EXPECT_FALSE(result.ended.requires_grad());
  n::TensorFunction f = [&](const auto& v) {
    return n::sum(n::square(m::SequenceGenerate(p, v[0]).logits));
  };
  EXPECT_TRUE(n::gradcheck(f, {z}, {.atol = 1e-6, .rtol = 1e-5}).ok);
}
TEST(NnSentenceVae, ExplicitCategoricalCdfTiesAndNoHiddenDraws) {
  n::ParameterStore p(3109);
  const auto        z = n::zeros({1, 4});
  m::SequenceGenerate(p, z);
  p.set("token_output/W", n::zeros({16, 19}));
  p.set("token_output/b", n::zeros({19}));
  for (auto [u, id] : std::vector<std::pair<double, int>>{
           {1. / 19, 1}, {2. / 19, 2}, {.5, 9}, {std::nextafter(1., 0.), 18}}) {
    const auto uniform = n::full({1, 7}, u, n::DType::Float64);
    auto       result  = m::SequenceGenerate(p, z, true, uniform);
    EXPECT_EQ(result.ids.to_vector<int>()[0], id);
    EXPECT_EQ(result.ended.to_vector<int>()[0], id == 2);
    EXPECT_TRUE(n::equal_bits(result.ids, m::SequenceGenerate(p, z, true, uniform).ids));
    if (id == 2)
      EXPECT_EQ(result.ids.to_vector<int>(), (std::vector<int>{2, 0, 0, 0, 0, 0, 0}));
  }
}
TEST(NnSentenceVae, AnnealedObjectiveAndSourceRoundedSchedule) {
  m::SequenceVaeOutput out;
  out.nll = n::ones({2}, n::DType::Float64).requires_grad_();
  out.kl  = n::full({2}, 2., n::DType::Float64).requires_grad_();
  EXPECT_EQ(m::AnnealedSequenceObjective(out, .5).item(), 2.);
  auto g = n::grad(m::AnnealedSequenceObjective(out, 0), {out.nll, out.kl});
  EXPECT_EQ(g[0].to_vector<double>(), (std::vector<double>{.5, .5}));
  EXPECT_EQ(g[1].to_vector<double>(), (std::vector<double>{0, 0}));
  for (int step = 0; step < 27; ++step) {
    EXPECT_EQ(m::CollapseBeta(false, step), 1.f);
    EXPECT_EQ(m::CollapseBeta(true, step), std::min(1.f, float(step) / 17));
  }
  EXPECT_THROW(m::CollapseBeta(true, -1), std::invalid_argument);
  EXPECT_THROW(m::CollapseBeta(false, 0, 1), std::invalid_argument);
  EXPECT_THROW(m::AnnealedSequenceObjective(out, 1.1), std::invalid_argument);
  EXPECT_THROW(m::AnnealedSequenceObjective(out, -.1), std::invalid_argument);
  EXPECT_THROW(
      m::AnnealedSequenceObjective(out, std::numeric_limits<double>::quiet_NaN()),
      std::invalid_argument);
}
TEST(NnSentenceVae, CollapsedSeparatedAndSignedMixtureControls) {
  const auto type      = n::DType::Float64;
  const auto zero      = n::zeros({8, 4}, type);
  const auto collapsed = m::CollapseStatistics(zero, zero, {n::ones({8, 4}, type)});
  EXPECT_EQ(collapsed.analytic_kl, 0);
  EXPECT_EQ(collapsed.active_units, 0);
  EXPECT_EQ(collapsed.mixture_information, 0);
  EXPECT_EQ(collapsed.sampled_conditional_kl, 0);
  EXPECT_EQ(collapsed.aggregate_kl, 0);
  std::vector<double> v(32, 0);
  for (int b = 0; b < 8; ++b)
    v[b * 4] = b % 2 ? 2 : -2;
  const auto shifted = n::from_vector(v, {8, 4});
  const auto active  = m::CollapseStatistics(shifted, zero, {shifted});
  EXPECT_EQ(active.active_units, 1);
  EXPECT_EQ(active.analytic_kl, 2);
  EXPECT_EQ(active.mu_variance[0], 4);
  EXPECT_GT(active.mixture_information, .1);
  EXPECT_NEAR(
      active.sampled_conditional_kl, active.mixture_information + active.aggregate_kl, 1e-14);
  const auto negative = m::CollapseStatistics(zero, n::full({8, 4}, std::log(2.), type), {zero});
  EXPECT_GT(negative.analytic_kl, 0);
  EXPECT_LT(negative.aggregate_kl, 0);
  EXPECT_LT(negative.sampled_conditional_kl, 0);
  EXPECT_EQ(negative.mixture_information, 0);
  const auto halves = n::from_vector(std::vector<double>{-.5, .5}, {2, 1});
  EXPECT_EQ(m::CollapseStatistics(halves, n::zeros({2, 1}, type), {halves}, .25).active_units, 0);
  EXPECT_EQ(
      m::CollapseStatistics(halves, n::zeros({2, 1}, type), {halves}, std::nextafter(.25, 0.))
          .active_units,
      1);
  auto strided = n::transpose(n::transpose(shifted, {1, 0}).contiguous(), {1, 0});
  EXPECT_EQ(m::CollapseStatistics(strided, zero, {strided}).unit_kl, active.unit_kl);
}
TEST(NnSentenceVae, InvalidInputsFailBeforeLeavesAndExtremeDiagnosticsAreExplicit) {
  n::ParameterStore p(3109);
  const auto        noise = n::zeros({2, 4});
  for (const auto& tokens :
       {n::zeros({2, 7}),
        n::zeros({2, 6}, n::DType::Int32),
        n::full({2, 7}, 19, n::DType::Int32),
        n::full({2, 7}, -1, n::DType::Int64),
        n::zeros({0, 7}, n::DType::Int32)})
    EXPECT_THROW(m::SequenceVariationalAutoencoder(p, tokens, noise), std::invalid_argument);
  EXPECT_THROW(
      m::SequenceVariationalAutoencoder(p, Sentences(), n::zeros({1, 4})), std::invalid_argument);
  EXPECT_THROW(
      m::SequenceVariationalAutoencoder(p, Sentences(), n::zeros({2, 4}, n::DType::Float64)),
      std::invalid_argument);
  EXPECT_THROW(
      m::SequenceVariationalAutoencoder(
          p, Sentences(), n::full({2, 4}, std::numeric_limits<double>::infinity())),
      std::invalid_argument);
  EXPECT_THROW(
      m::SequenceTeacher(p, noise, n::ones({1, 7}, n::DType::Int32)), std::invalid_argument);
  for (double uniform : {0., 1., std::numeric_limits<double>::infinity()})
    EXPECT_THROW(
        m::SequenceGenerate(p, noise, true, n::full({2, 7}, uniform, n::DType::Float64)),
        std::invalid_argument);
  EXPECT_THROW(m::SequenceGenerate(p, noise, true, n::full({2, 7}, .5)), std::invalid_argument);
  EXPECT_EQ(p.size(), 0u);
  EXPECT_THROW(
      m::sequence_vae_detail::Geometry(std::numeric_limits<std::int64_t>::max(), 7),
      std::invalid_argument);
  const auto z = n::zeros({2, 4});
  EXPECT_THROW(m::CollapseStatistics(z, z, {}), std::invalid_argument);
  EXPECT_THROW(
      m::CollapseStatistics(n::zeros({1, 4}), n::zeros({1, 4}), {n::zeros({1, 4})}),
      std::invalid_argument);
  EXPECT_THROW(m::CollapseStatistics(z, z, {n::zeros({2, 3})}), std::invalid_argument);
  EXPECT_THROW(m::CollapseStatistics(z, z, {z}, -.1), std::invalid_argument);
  EXPECT_THROW(m::CollapseStatistics(z, n::full({2, 4}, 1000.), {z}), std::overflow_error);
}
TEST(NnSentenceVae, OwnedFieldsExplicitAdamAndFreshModelRestore) {
  auto out = []() {
    n::ParameterStore p(3109);
    return m::SequenceVariationalAutoencoder(p, Sentences(), n::zeros({2, 4}));
  }();
  EXPECT_EQ(out.probability.numel(), 266);
  EXPECT_TRUE(std::isfinite(out.loss.item()));
  n::ParameterStore p(3109);
  out                       = m::SequenceVariationalAutoencoder(p, Sentences(), n::zeros({2, 4}));
  const auto         before = p.flatten();
  n::OptimizerConfig config;
  config.kind = n::OptimizerKind::Adam;
  n::Optimizer optimizer(p, .003, config);
  optimizer.step(n::grad(out.loss, p.parameters()));
  EXPECT_NE(p.flatten(), before);
  EXPECT_EQ(optimizer.iterations(), 1);
  n::ParameterStore restored(9);
  m::SequenceVariationalAutoencoder(restored, Sentences(), n::zeros({2, 4}));
  restored.unflatten(p.flatten());
  EXPECT_TRUE(
      n::equal_bits(
          m::SequenceVariationalAutoencoder(p, Sentences(), n::zeros({2, 4})).probability,
          m::SequenceVariationalAutoencoder(restored, Sentences(), n::zeros({2, 4})).probability));
}
