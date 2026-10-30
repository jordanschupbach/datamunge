#include <datamunge/algorithms/zero_r.hpp>
#include <gtest/gtest.h>
#include <limits>

namespace alg = datamunge::algorithms;
namespace {
datamunge::dstruct::DataFrame labels(std::vector<double> values) {
    datamunge::dstruct::DataFrame frame;
    frame.add_column("y", values);
    return frame;
}
}
TEST(ZeroR, FrameAdapterPreservesTieAndEmptySemantics) {
    const auto model = alg::zero_r_train(labels({2, -1, 2, -1}), "y");
    EXPECT_EQ(model.prediction, -1);
    EXPECT_EQ(model.total, 4);
    EXPECT_DOUBLE_EQ(model.accuracy(), .5);
    EXPECT_EQ(alg::zero_r_train(labels({}), "y").total, 0);
    EXPECT_EQ(alg::zero_r_counts(labels({}), "y").nrows(), 0);
    const auto counts = alg::zero_r_counts(labels({2, -1, 2, -1}), "y");
    EXPECT_EQ(counts.double_at("class", 0), -1);
    EXPECT_EQ(counts.double_at("count", 1), 2);
    EXPECT_DOUBLE_EQ(counts.double_at("proportion", 0), .5);
}
TEST(ZeroR, EvaluationNeverSelectsTheTestMajority) {
    const auto model = alg::zero_r_train(labels({0, 0, 1}), "y");
    const auto test = labels({0, 1, 1, 1});
    const auto scores = alg::zero_r_evaluate(model, test, "y");
    EXPECT_EQ(scores.double_at("prediction", 0), 0);
    EXPECT_DOUBLE_EQ(scores.double_at("training_accuracy", 0), 2.0/3);
    EXPECT_DOUBLE_EQ(scores.double_at("accuracy", 0), .25);
    EXPECT_DOUBLE_EQ(scores.double_at("balanced_accuracy", 0), .5);
    const auto confusion = alg::zero_r_confusion(model, test, "y");
    EXPECT_EQ(confusion.double_at("predicted", 0), 4);
    EXPECT_EQ(confusion.double_at("recall", 0), 1);
    EXPECT_EQ(confusion.double_at("recall", 1), 0);
    EXPECT_DOUBLE_EQ(confusion.double_at("precision", 0), .25);
}
TEST(ZeroR, BalancedAccuracyHandlesMulticlassAndUnseenLabels) {
    const auto model = alg::zero_r_train(labels({0, 0, 1}), "y");
    EXPECT_DOUBLE_EQ(alg::zero_r_evaluate(model, labels({0, 1, 2, 3}), "y").double_at("balanced_accuracy", 0), .25);
    const auto missing_majority = alg::zero_r_evaluate(model, labels({1, 2, 3}), "y");
    EXPECT_EQ(missing_majority.double_at("observed_classes", 0), 3);
    EXPECT_EQ(missing_majority.double_at("accuracy", 0), 0);
    EXPECT_EQ(missing_majority.double_at("balanced_accuracy", 0), 0);
    const auto absent = alg::zero_r_confusion(model, labels({1, 2, 3}), "y");
    EXPECT_EQ(absent.nrows(), 4);
    EXPECT_EQ(absent.double_at("support", 0), 0);
    EXPECT_EQ(absent.double_at("recall", 0), 0);
}
TEST(ZeroR, RejectsInvalidLabelsAndEmptyEvaluation) {
    const auto model = alg::zero_r_train(labels({0,1,0}), "y");
    EXPECT_THROW(alg::zero_r_evaluate(model, labels({}), "y"), std::invalid_argument);
    EXPECT_THROW(alg::zero_r_evaluate(alg::ZeroRModel{}, labels({0}), "y"), std::invalid_argument);
    for (double x : {0.5, std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN(),
                    static_cast<double>(std::numeric_limits<int>::max())+1})
        EXPECT_THROW(alg::zero_r_train(labels({x}), "y"), std::invalid_argument);
    auto missing = labels({1}); missing.set_null("y", 0);
    EXPECT_THROW(alg::zero_r_train(missing, "y"), std::invalid_argument);
    EXPECT_THROW(alg::zero_r_train(missing, "missing"), std::out_of_range);
    datamunge::dstruct::DataFrame text;
    text.add_column("y", std::vector<std::string>{"yes"});
    EXPECT_THROW(alg::zero_r_train(text, "y"), std::invalid_argument);
}
