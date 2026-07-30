#include <gtest/gtest.h>

#include <datamunge/algorithms/fuzzy_c_means.hpp>
#include <datamunge/algorithms/k_medoids.hpp>
#include <datamunge/algorithms/optics.hpp>

#include <cmath>
#include <cstddef>
#include <random>
#include <set>
#include <vector>

using datamunge::algorithms::fuzzy_c_means;
using datamunge::algorithms::k_medoids;
using datamunge::algorithms::optics;
using datamunge::algorithms::optics_extract_dbscan;

namespace {

// Three tight, well-separated blobs of 30 points each.
std::vector<std::vector<double>> three_blobs(std::uint64_t seed) {
    std::mt19937_64                        rng(seed);
    std::normal_distribution<double>       j(0.0, 0.25);
    std::vector<std::vector<double>>       data;
    const std::vector<std::vector<double>> c = {{0, 0}, {10, 0}, {5, 9}};
    for (const auto& ctr : c)
        for (int i = 0; i < 30; ++i) data.push_back({ctr[0] + j(rng), ctr[1] + j(rng)});
    return data;
}

}  // namespace

TEST(KMedoids, RecoversThreeBlobs) {
    auto data = three_blobs(1);
    auto res  = k_medoids(data, 3, 100, 2);

    EXPECT_EQ(res.medoids.size(), 3u);
    // Points 0..29, 30..59, 60..89 are the three blobs; each blob is one pure cluster.
    for (std::size_t blob = 0; blob < 3; ++blob) {
        std::size_t base = blob * 30;
        for (std::size_t i = base + 1; i < base + 30; ++i)
            EXPECT_EQ(res.assignment[i], res.assignment[base]) << "point " << i;
    }
    // The three blobs get three different labels.
    std::set<std::size_t> labels = {res.assignment[0], res.assignment[30], res.assignment[60]};
    EXPECT_EQ(labels.size(), 3u);
}

TEST(FuzzyCMeans, SoftMembershipsBehaveSensibly) {
    // Two well-separated 1D-ish blobs around x=0 and x=10.
    std::mt19937_64                  rng(3);
    std::normal_distribution<double> j(0.0, 0.3);
    std::vector<std::vector<double>> data;
    for (int i = 0; i < 40; ++i) data.push_back({0.0 + j(rng), 0.0 + j(rng)});
    for (int i = 0; i < 40; ++i) data.push_back({10.0 + j(rng), 0.0 + j(rng)});

    auto res = fuzzy_c_means(data, 2, 2.0, 200, 1e-6, 1);

    // Every membership row is a valid distribution.
    for (const auto& row : res.membership) {
        double s = 0.0;
        for (double u : row) {
            EXPECT_GE(u, -1e-9);
            EXPECT_LE(u, 1.0 + 1e-9);
            s += u;
        }
        EXPECT_NEAR(s, 1.0, 1e-6);
    }
    // A point deep in one blob is almost certain; the midpoint is near 50/50.
    auto deep = fuzzy_c_means(data, 2, 2.0, 200, 1e-6, 1);  // same fit
    // Identify which center is near x=0.
    std::size_t near0 = (deep.centers[0][0] < deep.centers[1][0]) ? 0 : 1;
    // Point 0 sits in the x=0 blob: high membership in near0.
    EXPECT_GT(deep.membership[0][near0], 0.8);
    // The exact midpoint (5,0) is maximally ambiguous.
    // Recompute its membership from centers directly.
    double d0 = std::hypot(5.0 - deep.centers[0][0], 0.0 - deep.centers[0][1]);
    double d1 = std::hypot(5.0 - deep.centers[1][0], 0.0 - deep.centers[1][1]);
    EXPECT_NEAR(d0, d1, 1.0);  // centers roughly symmetric about the midpoint
}

TEST(Optics, SeparatesTwoDenseClustersByGap) {
    // Two dense blobs separated by a wide gap.
    std::mt19937_64                  rng(4);
    std::normal_distribution<double> j(0.0, 0.2);
    std::vector<std::vector<double>> data;
    for (int i = 0; i < 40; ++i) data.push_back({0.0 + j(rng), 0.0 + j(rng)});
    for (int i = 0; i < 40; ++i) data.push_back({8.0 + j(rng), 0.0 + j(rng)});

    auto result = optics(data, 5.0, 4);
    ASSERT_EQ(result.ordering.size(), data.size());

    auto labels = optics_extract_dbscan(result, 1.0);
    // Exactly two clusters (ignoring any noise), each blob internally consistent.
    std::set<int> clusters;
    for (int l : labels)
        if (l >= 0) clusters.insert(l);
    EXPECT_EQ(clusters.size(), 2u);

    // Points within a blob share a label; the two blobs differ.
    for (std::size_t i = 1; i < 40; ++i) EXPECT_EQ(labels[i], labels[0]);
    for (std::size_t i = 41; i < 80; ++i) EXPECT_EQ(labels[i], labels[40]);
    EXPECT_NE(labels[0], labels[40]);
}
