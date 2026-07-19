#include <gtest/gtest.h>

#include <datamunge/dstruct/dstruct.hpp>
#include <datamunge/stats/agglomerative_clustering.hpp>
#include <datamunge/stats/dbscan.hpp>
#include <datamunge/stats/kmeans.hpp>

#include <cmath>
#include <vector>

using datamunge::dstruct::DataFrame;
using datamunge::stats::AgglomerativeClustering;
using datamunge::stats::AgglomerativeClusteringOptions;
using datamunge::stats::DBSCAN;
using datamunge::stats::DBSCANOptions;
using datamunge::stats::KMeans;
using datamunge::stats::KMeansOptions;
using datamunge::stats::LinkageCriterion;

// Reference values below were computed with scikit-learn 1.8.0 (a genuine, independently
// implemented oracle -- see scratchpad/cluster_oracle/{gen_data,oracle}.py) on a small
// synthetic three-blob dataset (well-separated, so the globally optimal partition is
// essentially unique up to relabeling). Since cluster *label numbering* is an implementation
// detail (arbitrary relabeling), comparisons use a permutation-invariant "same partition"
// check rather than exact label equality.

namespace {

DataFrame make_oracle_dataframe() {
    const std::vector<double> x0{-1.009722, 8.407348,  0.097652,  8.070011,  0.280506,  7.834915,  0.039618,
                                  7.480501,  8.173472,  0.247640,  9.284989,  7.600694,  0.733525,  -0.256997,
                                  -0.209235, 7.931632,  0.288448,  0.527070,  8.522857,  -0.717504, 0.274065,
                                  7.692654,  0.450271,  7.125707,  0.076704,  -0.551671, 7.505311,  0.221250,
                                  0.514786,  0.085455,  0.319386,  8.445953,  0.375354,  -0.256352, 7.717776,
                                  0.527639,  0.182830,  -0.765412, -0.281641, 0.426736,  -0.010081, 8.369588,
                                  -0.217832, -1.170621, -0.110917};
    const std::vector<double> x1{7.799069,  8.040547, 8.351733,  8.131213,  -0.515575, 8.896965,  0.676345,
                                  8.580967,  8.378773, 0.258493,  7.756151,  8.139297,  -0.092718, -0.211280,
                                  7.722589,  7.495906, 8.267919,  -0.029956, 8.134157,  8.292183,  7.602844,
                                  7.511736,  0.564339, 7.808197,  -0.189746, 8.298296,  8.390356,  -0.575330,
                                  7.885217,  8.414291, 0.219266,  8.325893,  7.814392,  8.095124,  7.616673,
                                  0.466675,  -0.623990, 7.320028, 8.007496,  8.476008,  -0.511826, 8.677383,
                                  7.770957,  -0.781308, -0.408558};
    DataFrame df;
    df.add_column("x0", x0);
    df.add_column("x1", x1);
    return df;
}

// True iff `a` and `b` induce the same partition of {0, ..., n-1} up to relabeling: for
// every pair (i, j), a[i] == a[j] exactly when b[i] == b[j]. Points with a *negative* label
// in either vector (DBSCAN noise) are excluded from the comparison.
template <typename LabelT>
bool same_partition(const std::vector<LabelT>& a, const std::vector<LabelT>& b) {
    const std::size_t n = a.size();
    for (std::size_t i = 0; i < n; ++i) {
        if (a[i] < LabelT{0} || b[i] < LabelT{0}) continue;
        for (std::size_t j = i + 1; j < n; ++j) {
            if (a[j] < LabelT{0} || b[j] < LabelT{0}) continue;
            if ((a[i] == a[j]) != (b[i] == b[j])) return false;
        }
    }
    return true;
}

} // namespace

TEST(KMeans, MatchesSklearnInertiaAndPartitionOnWellSeparatedBlobs) {
    const auto df = make_oracle_dataframe();
    KMeansOptions options;
    options.n_clusters = 3;
    options.n_init = 10;
    const KMeans model(df, {"x0", "x1"}, options);

    EXPECT_NEAR(model.inertia(), 17.26117233052395, 1e-2);

    const std::vector<int> oracle_labels{0, 2, 0, 2, 1, 2, 1, 2, 2, 1, 2, 2, 1, 1, 0, 2, 0, 1, 2, 0, 0,
                                          2, 1, 2, 1, 0, 2, 1, 0, 0, 1, 2, 0, 0, 2, 1, 1, 0, 0, 0, 1, 2,
                                          0, 1, 1};
    std::vector<int> fitted_labels(model.labels().begin(), model.labels().end());
    EXPECT_TRUE(same_partition(fitted_labels, oracle_labels));

    // Sanity checks independent of the oracle: three clusters of 15 points each.
    std::vector<std::size_t> counts(3, 0);
    for (const auto l : model.labels()) ++counts[l];
    for (const auto c : counts) EXPECT_EQ(c, 15u);
}

TEST(KMeans, PredictAssignsNewPointToNearestCenter) {
    const auto df = make_oracle_dataframe();
    KMeansOptions options;
    options.n_clusters = 3;
    const KMeans model(df, {"x0", "x1"}, options);

    DataFrame newdata;
    newdata.add_column("x0", std::vector<double>{8.0});
    newdata.add_column("x1", std::vector<double>{8.0});
    const auto preds = model.predict(newdata);
    ASSERT_EQ(preds.size(), 1u);
    const auto& centers = model.cluster_centers();
    EXPECT_NEAR(centers(preds[0], 0), 8.0, 1.0);
    EXPECT_NEAR(centers(preds[0], 1), 8.0, 1.0);
}

class AgglomerativeLinkage : public ::testing::TestWithParam<std::pair<LinkageCriterion, std::vector<int>>> {};

TEST_P(AgglomerativeLinkage, MatchesSklearnPartition) {
    const auto df = make_oracle_dataframe();
    AgglomerativeClusteringOptions options;
    options.n_clusters = 3;
    options.linkage = GetParam().first;
    const AgglomerativeClustering model(df, {"x0", "x1"}, options);

    std::vector<int> fitted_labels(model.labels().begin(), model.labels().end());
    EXPECT_TRUE(same_partition(fitted_labels, GetParam().second));
}

INSTANTIATE_TEST_SUITE_P(
    AllLinkages, AgglomerativeLinkage,
    ::testing::Values(
        std::make_pair(LinkageCriterion::Single,
                       std::vector<int>{2, 1, 2, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 2, 1, 2, 0, 1, 2, 2,
                                        1, 0, 1, 0, 2, 1, 0, 2, 2, 0, 1, 2, 2, 1, 0, 0, 2, 2, 2, 0, 1,
                                        2, 0, 0}),
        std::make_pair(LinkageCriterion::Complete,
                       std::vector<int>{2, 0, 2, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 2, 0, 2, 1, 0, 2, 2,
                                        0, 1, 0, 1, 2, 0, 1, 2, 2, 1, 0, 2, 2, 0, 1, 1, 2, 2, 2, 1, 0,
                                        2, 1, 1}),
        std::make_pair(LinkageCriterion::Average,
                       std::vector<int>{2, 1, 2, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 2, 1, 2, 0, 1, 2, 2,
                                        1, 0, 1, 0, 2, 1, 0, 2, 2, 0, 1, 2, 2, 1, 0, 0, 2, 2, 2, 0, 1,
                                        2, 0, 0}),
        std::make_pair(LinkageCriterion::Ward,
                       std::vector<int>{2, 1, 2, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 2, 1, 2, 0, 1, 2, 2,
                                        1, 0, 1, 0, 2, 1, 0, 2, 2, 0, 1, 2, 2, 1, 0, 0, 2, 2, 2, 0, 1,
                                        2, 0, 0})));

TEST(AgglomerativeClustering, MergeHistoryHasExpectedLengthAndIncreasingDistance) {
    const auto df = make_oracle_dataframe();
    AgglomerativeClusteringOptions options;
    options.n_clusters = 1;
    options.linkage = LinkageCriterion::Average;
    const AgglomerativeClustering model(df, {"x0", "x1"}, options);

    ASSERT_EQ(model.merge_history().size(), model.observations() - 1);
    for (std::size_t i = 1; i < model.merge_history().size(); ++i)
        EXPECT_GE(model.merge_history()[i].distance, model.merge_history()[i - 1].distance - 1e-9);
}

TEST(AgglomerativeClustering, CutReproducesFittedLabelsAtSameK) {
    const auto df = make_oracle_dataframe();
    AgglomerativeClusteringOptions options;
    options.n_clusters = 3;
    const AgglomerativeClustering model(df, {"x0", "x1"}, options);
    const auto recut = model.cut(3);
    EXPECT_TRUE(same_partition(std::vector<int>(model.labels().begin(), model.labels().end()),
                               std::vector<int>(recut.begin(), recut.end())));
}

TEST(AgglomerativeClustering, RejectsWardWithManhattanMetric) {
    const auto df = make_oracle_dataframe();
    AgglomerativeClusteringOptions options;
    options.linkage = LinkageCriterion::Ward;
    options.metric = datamunge::stats::DistanceMetric::Manhattan;
    EXPECT_THROW(AgglomerativeClustering(df, {"x0", "x1"}, options), std::invalid_argument);
}

TEST(DBSCAN, MatchesSklearnPartitionAndNoiseCount) {
    const auto df = make_oracle_dataframe();
    DBSCANOptions options;
    options.eps = 1.5;
    options.min_samples = 3;
    const DBSCAN model(df, {"x0", "x1"}, options);

    EXPECT_EQ(model.n_clusters(), 3u);
    EXPECT_EQ(model.n_noise(), 0u);

    const std::vector<int> oracle_labels{0, 1, 0, 1, 2, 1, 2, 1, 1, 2, 1, 1, 2, 2, 0, 1, 0, 2, 1, 0, 0,
                                          1, 2, 1, 2, 0, 1, 2, 0, 0, 2, 1, 0, 0, 1, 2, 2, 0, 0, 0, 2, 1,
                                          0, 2, 2};
    EXPECT_TRUE(same_partition(model.labels(), oracle_labels));
}

TEST(DBSCAN, TightEpsProducesNoise) {
    const auto df = make_oracle_dataframe();
    DBSCANOptions options;
    options.eps = 0.05; // far smaller than the within-blob point spacing
    options.min_samples = 3;
    const DBSCAN model(df, {"x0", "x1"}, options);
    EXPECT_GT(model.n_noise(), 0u);
}

TEST(Clustering, RejectsUnknownFeatureColumn) {
    const auto df = make_oracle_dataframe();
    EXPECT_THROW(KMeans(df, {"x0", "nonexistent"}), std::invalid_argument);
    EXPECT_THROW(AgglomerativeClustering(df, {"x0", "nonexistent"}), std::invalid_argument);
    EXPECT_THROW(DBSCAN(df, {"x0", "nonexistent"}), std::invalid_argument);
}
