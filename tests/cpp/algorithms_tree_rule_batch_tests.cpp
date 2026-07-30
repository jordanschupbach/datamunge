#include <gtest/gtest.h>

#include <datamunge/algorithms/decision_tree_induction.hpp>
#include <datamunge/algorithms/rule_learners.hpp>

#include <cstddef>
#include <vector>

using datamunge::algorithms::c45_predict;
using datamunge::algorithms::c45_train;
using datamunge::algorithms::id3_predict;
using datamunge::algorithms::id3_train;
using datamunge::algorithms::one_r_predict;
using datamunge::algorithms::one_r_train;
using datamunge::algorithms::zero_r_predict;
using datamunge::algorithms::zero_r_train;

namespace {

// The classic "play tennis" dataset, integer-coded.
// Attributes: outlook{0=sunny,1=overcast,2=rain}, temp{0=hot,1=mild,2=cool},
//             humidity{0=high,1=normal}, windy{0=false,1=true}. Label: play{0=no,1=yes}.
void tennis(std::vector<std::vector<int>>& X, std::vector<int>& y) {
    X = {{0, 0, 0, 0}, {0, 0, 0, 1}, {1, 0, 0, 0}, {2, 1, 0, 0}, {2, 2, 1, 0},
         {2, 2, 1, 1}, {1, 2, 1, 1}, {0, 1, 0, 0}, {0, 2, 1, 0}, {2, 1, 1, 0},
         {0, 1, 1, 1}, {1, 1, 0, 1}, {1, 0, 1, 0}, {2, 1, 0, 1}};
    y = {0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0};
}

}  // namespace

TEST(ZeroR, PredictsMajorityClass) {
    std::vector<std::vector<int>> X;
    std::vector<int>              y;
    tennis(X, y);  // 9 "yes", 5 "no"
    auto m = zero_r_train(y);
    EXPECT_EQ(m.prediction, 1);          // "yes" is the majority
    EXPECT_EQ(m.majority_count, 9u);
    EXPECT_NEAR(m.accuracy(), 9.0 / 14.0, 1e-12);
    EXPECT_EQ(zero_r_predict(m, {0, 0, 0, 0}), 1);
}

TEST(OneR, PicksBestSingleAttribute) {
    std::vector<std::vector<int>> X;
    std::vector<int>              y;
    tennis(X, y);
    auto m = one_r_train(X, y);
    // OneR on this dataset selects "outlook" (attribute 0), the strongest single predictor.
    EXPECT_EQ(m.feature, 0u);
    // Its rule beats the ZeroR baseline (9/14).
    EXPECT_GT(m.accuracy(), 9.0 / 14.0);
    // Overcast (outlook=1) always means play=yes.
    EXPECT_EQ(one_r_predict(m, {1, 0, 0, 0}), 1);
}

TEST(ID3, FitsTennisPerfectly) {
    std::vector<std::vector<int>> X;
    std::vector<int>              y;
    tennis(X, y);
    auto tree = id3_train(X, y);
    // ID3 grows until pure -> zero training error on this classic (consistent) dataset.
    std::size_t correct = 0;
    for (std::size_t i = 0; i < X.size(); ++i) correct += (id3_predict(tree, X[i]) == y[i]);
    EXPECT_EQ(correct, X.size());
    // The root split is on outlook (max information gain).
    ASSERT_FALSE(tree.nodes.empty());
    EXPECT_FALSE(tree.nodes[0].leaf);
    EXPECT_EQ(tree.nodes[0].feature, 0u);
}

TEST(C45, SeparatesContinuousClasses) {
    // Two 1-D-separable classes in 2-D: class 0 near origin, class 1 far along x.
    std::vector<std::vector<double>> X;
    std::vector<int>                 y;
    for (int i = 0; i < 30; ++i) { X.push_back({0.0 + 0.01 * i, 5.0}); y.push_back(0); }
    for (int i = 0; i < 30; ++i) { X.push_back({10.0 + 0.01 * i, 5.0}); y.push_back(1); }
    auto tree = c45_train(X, y, 0, 2);
    for (std::size_t i = 0; i < X.size(); ++i) EXPECT_EQ(c45_predict(tree, X[i]), y[i]);
    // A held-out point on each side.
    EXPECT_EQ(c45_predict(tree, {0.5, 5.0}), 0);
    EXPECT_EQ(c45_predict(tree, {10.5, 5.0}), 1);
    // Root splits on feature 0 (the discriminative one) at a threshold between the classes.
    ASSERT_FALSE(tree.nodes.empty());
    EXPECT_FALSE(tree.nodes[0].leaf);
    EXPECT_EQ(tree.nodes[0].feature, 0u);
    EXPECT_GT(tree.nodes[0].threshold, 0.3);
    EXPECT_LT(tree.nodes[0].threshold, 10.0);
}

TEST(C45, GreedySplitCannotSolvePureXOR) {
    // A documented limitation: on pure XOR every single-feature split has ZERO
    // information gain (each side is 50/50), so a greedy info-gain/gain-ratio tree
    // cannot even make the first split -- the root stays a leaf and accuracy is ~0.5.
    // This is a real property of ID3/C4.5, not a defect.
    std::vector<std::vector<double>> X = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    std::vector<int>                 y = {0, 1, 1, 0};
    std::vector<std::vector<double>> XX;
    std::vector<int>                 yy;
    for (int r = 0; r < 5; ++r)
        for (std::size_t i = 0; i < X.size(); ++i) { XX.push_back(X[i]); yy.push_back(y[i]); }
    auto tree = c45_train(XX, yy, 0, 1);
    ASSERT_FALSE(tree.nodes.empty());
    EXPECT_TRUE(tree.nodes[0].leaf);  // no useful first split exists for XOR
    std::size_t correct = 0;
    for (std::size_t i = 0; i < XX.size(); ++i) correct += (c45_predict(tree, XX[i]) == yy[i]);
    EXPECT_EQ(correct, XX.size() / 2);  // majority-class leaf -> exactly half right
}
