#include <gtest/gtest.h>
#include <datamunge/nn/nn.hpp>

namespace n = datamunge::nn;

TEST(NnClassicalInference, ForwardEvidenceHasCorrectMassAndPosteriorGradient) {
  const auto type = n::DType::Float64;
  const auto initial = n::log(n::tensor({.6, .4}, {2}, type));
  const auto transition = n::log(n::tensor({.8, .2, .3, .7}, {2, 2}, type));
  auto evidence = n::log(n::tensor({.65, .05, .3, .25, .05, .7, .3, .25},
                                  {1, 4, 2}, type)).detach().requires_grad_();
  const auto likelihood = n::ml::hmm_log_likelihood(initial, transition, evidence);
  const auto posterior = n::grad(n::sum(likelihood), {evidence})[0];
  EXPECT_NEAR(n::exp(likelihood).item(), .00773258, 1e-12);
  EXPECT_LT(n::max(n::abs(n::sum(posterior, {2}) - 1)).item(), 1e-12);
  const auto check = n::gradcheck(
      [&](const std::vector<n::Tensor>& values) {
        return n::ml::hmm_log_likelihood(initial, transition, values[0]);
      }, {evidence});
  EXPECT_TRUE(check.ok) << check.message;
}

TEST(NnClassicalInference, ForwardHandlesBatchAndSingleTimeStep) {
  const auto type = n::DType::Float64;
  const auto initial = n::log(n::tensor({.5, .5}, {2}, type));
  const auto transition = n::log(n::full({2, 2}, .5, type));
  const auto uniform = n::log(n::full({2, 3, 2}, .25, type));
  const auto ll = n::ml::hmm_log_likelihood(initial, transition, uniform);
  EXPECT_EQ(ll.shape(), (n::Shape{2}));
  EXPECT_NEAR(ll.at({0}), std::log(1. / 64), 1e-12);
  EXPECT_DOUBLE_EQ(ll.at({0}), ll.at({1}));
  const auto single = n::log(n::tensor({.2, .8}, {1, 1, 2}, type));
  EXPECT_NEAR(n::ml::hmm_log_likelihood(initial, transition, single).item(), std::log(.5), 1e-12);
  EXPECT_THROW(n::ml::hmm_log_likelihood(initial, transition, n::zeros({1, 0, 2}, type)),
               std::invalid_argument);
  EXPECT_THROW(n::ml::hmm_log_likelihood(initial, transition, n::zeros({1, 2, 3}, type)),
               std::invalid_argument);
  EXPECT_THROW(n::ml::hmm_log_likelihood(initial, transition, n::zeros({1, 2, 2})),
               std::invalid_argument);
}

TEST(NnClassicalInference, WeightedNeighborsUseOnlyExactMatches) {
  const auto references = n::tensor({0, 0, 2}, {3, 1});
  const auto query = n::tensor({0}, {1, 1});
  const n::ml::TensorNeighbors neighbors(references, query, 3, true);
  EXPECT_EQ(neighbors.indices.to_vector<int64_t>(), (std::vector<int64_t>{0, 1, 2}));
  EXPECT_EQ(neighbors.weights.to_vector<float>(), (std::vector<float>{.5f, .5f, 0.f}));
  auto targets = n::tensor({2, 4, 20}).requires_grad_();
  const auto prediction = neighbors.regress(targets);
  EXPECT_DOUBLE_EQ(prediction.item(), 3.);
  EXPECT_EQ(n::grad(n::sum(prediction), {targets})[0].to_vector<float>(),
            (std::vector<float>{.5f, .5f, 0.f}));
}

TEST(NnClassicalInference, NeighborWeightsAndClassTiesMatchKnownValues) {
  const auto type = n::DType::Float64;
  const auto references = n::tensor({0, 2}, {2, 1}, type);
  const n::ml::TensorNeighbors neighbors(references, n::tensor({.5}, {1, 1}, type), 2, true);
  EXPECT_NEAR(neighbors.regress(n::tensor({0, 10}, {2}, type)).item(), 2.5, 1e-12);
  const n::ml::TensorNeighbors tie(references, n::tensor({1}, {1, 1}, type), 2);
  EXPECT_EQ(tie.classify(n::from_vector(std::vector<int64_t>{1, 0}), 2).item(), 0);
  EXPECT_THROW(n::ml::TensorNeighbors(references, n::zeros({1, 2}, type), 1),
               std::invalid_argument);
  EXPECT_THROW(n::ml::TensorNeighbors(references, n::zeros({1, 1}, type), 3),
               std::invalid_argument);
  EXPECT_THROW(tie.classify(n::from_vector(std::vector<int64_t>{1, 0}), 0),
               std::invalid_argument);
}
