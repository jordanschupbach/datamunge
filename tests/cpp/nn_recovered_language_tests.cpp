#include <gtest/gtest.h>

#include <dm/dm.hpp>
namespace {
namespace n = dm::nn;
namespace m = n::models;
TEST(RecoveredLanguage, CachedChunkMatchesFullContextAndDoesNotAddLeaves) {
  m::DecoderConfig cfg;
  cfg.context        = 8;
  cfg.input_classes  = 5;
  cfg.output_classes = 5;
  cfg.dimensions     = 4;
  cfg.heads          = 2;
  cfg.feed_forward   = 8;
  m::CachedTransformerLM model(cfg);
  auto                   ids    = n::from_vector(std::vector<int>{0, 1, 2, 3, 4, 3, 2, 1}, {1, 8});
  auto                   full   = model.Predict(ids);
  auto                   state  = model.Empty(1);
  auto                   first  = model.Append(n::slice_axis(ids, 1, 0, 3), state);
  auto                   second = model.Append(n::slice_axis(ids, 1, 3, 8), state);
  auto                   actual = n::concat({first.at("logits"), second.at("logits")}, 1);
  EXPECT_EQ(model.store.size(), 18);
  EXPECT_LT(n::max(n::abs(actual - full)).item(), 2e-6);
  EXPECT_EQ(state.keys.size(2), 8);
}
TEST(RecoveredLanguage, CacheFailurePreservesEndpointsAndRebuildsAbsolutePositions) {
  m::DecoderConfig cfg;
  cfg.context        = 4;
  cfg.input_classes  = 5;
  cfg.output_classes = 5;
  cfg.dimensions     = 4;
  cfg.heads          = 2;
  cfg.feed_forward   = 8;
  m::CachedTransformerLM model(cfg);
  auto                   state = model.Empty(1);
  model.Append(n::from_vector(std::vector<int>{1}, {1, 1}), state);
  auto before = state.keys.to_vector<float>();
  EXPECT_THROW(
      model.Append(n::from_vector(std::vector<int>{9}, {1, 1}), state), std::invalid_argument);
  EXPECT_EQ(state.keys.to_vector<float>(), before);
  bool rebuilt = false;
  auto output  = model.Advance({0, 1, 2, 3, 4}, state, &rebuilt);
  EXPECT_TRUE(rebuilt);
  EXPECT_EQ(state.keys.size(2), 4);
  EXPECT_THROW(model.Empty(0), std::invalid_argument);
}
TEST(RecoveredLanguage, NucleusRankingAndCallerUniformBoundaries) {
  auto d = m::FilterTokens({std::log(.4), std::log(.3), std::log(.2), std::log(.1)}, 1, 0, .6);
  EXPECT_EQ(d.retained, (std::vector<int>{0, 1}));
  EXPECT_NEAR(d.probability[0], 4. / 7, 1e-15);
  EXPECT_NEAR(d.retained_mass, .7, 1e-15);
  EXPECT_EQ(m::DrawToken(d.probability, 0), 0);
  EXPECT_EQ(m::DrawToken(d.probability, std::nextafter(1., 0.)), 1);
  EXPECT_EQ(m::FilterTokens({0, 0, 0}, 1, 2).retained, (std::vector<int>{0, 1}));
  EXPECT_THROW(m::FilterTokens({NAN}), std::invalid_argument);
  EXPECT_THROW(m::DrawToken({1}, 1), std::invalid_argument);
}
TEST(RecoveredLanguage, CharacterSequenceEqualsStreamingAndHasNoFutureDerivative) {
  n::ParameterStore p(8903);
  m::CharacterRnnParameters(p);
  auto                   x     = n::ones({1, 64, 65}).requires_grad_();
  auto                   full  = m::CharacterRnnSequence(p, x);
  auto                   state = n::zeros({1, 16});
  std::vector<n::Tensor> scores;
  for (int t = 0; t < 64; ++t) {
    auto out = m::CharacterRnnStep(p, n::squeeze(n::slice_axis(x, 1, t, t + 1), {1}), state);
    state    = out.first;
    scores.push_back(n::expand_dims(out.second, 1));
  }
  EXPECT_EQ(n::concat(scores, 1).to_vector<float>(), full.first.to_vector<float>());
  EXPECT_EQ(p.count(), 2025);
  auto g = n::grad(n::sum(n::slice_axis(full.first, 1, 0, 1)), {x})[0];
  EXPECT_EQ(n::max(n::abs(n::slice_axis(g, 1, 1, 64))).item(), 0);
}
TEST(RecoveredLanguage, WordRecurrenceRetainsFullVocabularyAndSourceLeafCount) {
  n::ParameterStore p(1);
  auto              input = n::zeros({1, 16, 10000});
  auto              path  = m::word_lm::Sequence(p, input);
  auto              first = m::word_lm::Step(p, n::zeros({1, 10000}), n::zeros({1, 8}));
  EXPECT_EQ(p.count(), 170136);
  EXPECT_EQ(
      n::squeeze(n::slice_axis(path.logits, 1, 0, 1), {1}).to_vector<float>(),
      first.logits.to_vector<float>());
  EXPECT_EQ(path.logits.shape(), (n::Shape{1, 16, 10000}));
}
TEST(RecoveredLanguage, ImdbPaddingHasZeroInputDerivativeAndZeroLengthFreezesState) {
  n::ParameterStore p(0);
  auto              x    = n::ones({2, 48, 256}).requires_grad_();
  auto              path = m::imdb::ImdbLstm(
      p, x, n::from_vector(std::vector<int>{2, 0}), n::zeros({2, 8}), n::zeros({2, 8}));
  auto g = n::grad(n::sum(path.logits), {x})[0];
  EXPECT_EQ(n::max(n::abs(n::slice_axis(g, 1, 2, 48))).item(), 0);
  EXPECT_EQ(n::max(n::abs(n::slice_axis(g, 0, 1, 2))).item(), 0);
  EXPECT_EQ(n::max(n::abs(n::slice_axis(path.final_state, 0, 1, 2))).item(), 0);
  EXPECT_EQ(path.logits.shape(), (n::Shape{2, 2}));
}
TEST(RecoveredLanguage, BidirectionalAlignmentIsInvariantToPaddingChanges) {
  n::ParameterStore p(0);
  auto              x       = n::zeros({1, 24, 512});
  auto              altered = x.detach().clone();
  auto*             data    = altered.data_ptr<float>();
  for (int i = 3 * 512; i < 24 * 512; ++i)
    data[i] = 1;
  auto                   lengths = n::from_vector(std::vector<int>{3});
  std::vector<n::Tensor> initial(4, n::zeros({1, 8}));
  auto                   a = m::bilstm_tagging::BiLstmFeatures(p, x, lengths, initial);
  auto                   b = m::bilstm_tagging::BiLstmFeatures(p, altered, lengths, initial);
  EXPECT_EQ(a.features.to_vector<float>(), b.features.to_vector<float>());
  EXPECT_EQ(a.forward_states.shape(), (n::Shape{1, 25, 8}));
  EXPECT_EQ(a.backward_aligned.shape(), (n::Shape{1, 24, 8}));
}
TEST(RecoveredLanguage, CrfEncoderAndTransitionMarginalsRetainSequenceLength) {
  n::ParameterStore p(0);
  auto              lengths = n::from_vector(std::vector<int>{2});
  auto              path    = m::bilstm_crf::Encode(p, n::zeros({1, 24}, n::DType::Int32), lengths);
  auto              crf     = m::bilstm_crf::CrfParams(p);
  auto normalizer = n::crf_log_norm(path.unaries, lengths, crf.transition, crf.start, crf.end);
  auto g          = n::grad(n::sum(normalizer), {crf.transition})[0];
  EXPECT_NEAR(n::sum(g).item(), 1, 2e-6);
  EXPECT_EQ(n::max(n::abs(n::slice_axis(path.unaries, 1, 2, 24))).item(), 0);
}
} // namespace
