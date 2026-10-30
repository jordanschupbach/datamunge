
#include <dm/dm.hpp>
#include <gtest/gtest.h>
namespace n = dm::nn;
namespace m = n::models;
TEST(NnDetectionInference, ClassImageSeparationAndIndependentBudgets) {
  const auto boxes = n::tensor(
      {0., 0., 10., 10., 0., 0., 10., 10., 0., 0., 10., 10., 0., 0., 10., 10.}, {4, 4});
  const auto scores = n::tensor({.9, .8, .7, .6}),
             labels = n::from_vector(std::vector<std::int64_t>{0, 0, 1, 0}),
             images = n::from_vector(std::vector<std::int64_t>{0, 0, 0, 1});
  const auto uncapped = m::SuppressDetectionCandidates(boxes, scores, labels, images, .5);
  EXPECT_EQ(uncapped.images.to_vector<int>(), std::vector<int>({0, 0, 1}));
  EXPECT_EQ(uncapped.labels.to_vector<int>(), std::vector<int>({0, 1, 0}));
  const auto capped =
      m::SuppressDetectionCandidates(boxes, scores, labels, images, .5, .05, 1);
  EXPECT_EQ(capped.images.to_vector<int>(), std::vector<int>({0, 1}));
  EXPECT_NEAR(capped.scores.at({1}), .6, 1e-6);
  const auto none = m::SuppressDetectionCandidates(boxes, scores, labels, images, .5, .05, 0);
  EXPECT_EQ(none.boxes.shape(), n::Shape({0, 4}));
}
TEST(NnDetectionInference, StrictScoreEqualityStableTiesAndEmptyCandidates) {
  const auto boxes =
      n::tensor({0., 0., 10., 10., 20., 20., 30., 30., 40., 40., 50., 50.}, {3, 4});
  const auto ids = n::zeros({3}, n::DType::Int64);
  const auto d =
      m::SuppressDetectionCandidates(boxes, n::tensor({.5, .5, .25}), ids, ids, .5, .25);
  EXPECT_EQ(d.boxes.at({0, 0}), 0);
  EXPECT_EQ(d.boxes.at({1, 0}), 20);
  EXPECT_EQ(d.scores.numel(), 2);
  const auto e = m::SuppressDetectionCandidates(n::zeros({0, 4}), n::zeros({0}),
                                                n::zeros({0}, n::DType::Int64),
                                                n::zeros({0}, n::DType::Int64));
  EXPECT_EQ(e.scores.numel(), 0);
  EXPECT_TRUE(m::DetectionEvaluationRecords(e).empty());
}
TEST(NnDetectionInference, PresentationRecordsRoundFloat64AndPreserveOrder) {
  const auto boxes = n::from_vector(std::vector<double>{1, 2, 10, 20}, {1, 4});
  const auto d = m::SuppressDetectionCandidates(boxes, n::from_vector(std::vector<double>{.9}),
                                                n::from_vector(std::vector<std::int64_t>{3}),
                                                n::from_vector(std::vector<std::int64_t>{7}));
  const auto records = m::DetectionEvaluationRecords(d);
  ASSERT_EQ(records.size(), 1);
  EXPECT_EQ(records[0].image, 7);
  EXPECT_EQ(records[0].label, 3);
  EXPECT_EQ(records[0].box, n::Box({1, 2, 10, 20}));
  EXPECT_FLOAT_EQ(records[0].score, .9f);
}
TEST(NnDetectionInference, InvalidInputsFailBeforeNms) {
  const auto ids = n::zeros({1}, n::DType::Int64),
             boxes = n::tensor({0., 0., 10., 10.}, {1, 4});
  EXPECT_THROW(m::SuppressDetectionCandidates(boxes, n::tensor({NAN}), ids, ids),
               std::invalid_argument);
  EXPECT_THROW(m::SuppressDetectionCandidates(n::tensor({10., 0., 0., 10.}, {1, 4}),
                                              n::tensor({.9}), ids, ids),
               std::invalid_argument);
  EXPECT_THROW(m::SuppressDetectionCandidates(boxes, n::tensor({.9}), ids, ids, -1.),
               std::invalid_argument);
  EXPECT_THROW(m::SuppressDetectionCandidates(boxes, n::tensor({.9}), ids - 1, ids),
               std::invalid_argument);
  const m::TensorDetections d{boxes, n::tensor({.9}), ids,
                              n::from_vector(std::vector<std::int64_t>{
                                  std::int64_t(std::numeric_limits<int>::max()) + 1})};
  EXPECT_THROW(m::DetectionEvaluationRecords(d), std::invalid_argument);
}
