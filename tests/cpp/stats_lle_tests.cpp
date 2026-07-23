#include <gtest/gtest.h>

#include <datamunge/datasets/datasets.hpp>
#include <datamunge/stats/lle.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

using datamunge::dstruct::DataFrame;
using datamunge::stats::DistanceMetric;
using datamunge::stats::LLE;
using datamunge::stats::LLEOptions;

namespace {

const std::vector<std::string> kIrisFeatures = {"Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"};

} // namespace

TEST(LLE, EmbeddingShapeAndFiniteness) {
    const auto iris = datamunge::datasets::iris();
    LLEOptions opts;
    opts.n_components = 2;
    opts.n_neighbors = 12;
    LLE lle(iris, kIrisFeatures, opts);

    EXPECT_EQ(lle.observations(), iris.nrows());
    EXPECT_EQ(lle.n_components(), 2u);
    EXPECT_EQ(lle.kept_row_indices().size(), iris.nrows());
    EXPECT_EQ(lle.eigenvalues().size(), iris.nrows());

    for (std::size_t i = 0; i < lle.observations(); ++i)
        for (std::size_t c = 0; c < lle.n_components(); ++c)
            EXPECT_TRUE(std::isfinite(lle.embedding()(i, c))) << "row " << i << " col " << c;
}

TEST(LLE, DiscardedEigenvalueIsNearZeroRelativeToTheSpectrum) {
    // The classical LLE derivation discards the smallest (trivial, ~0) eigenvalue of
    // M = (I-W)^T(I-W). Verify empirically that the very smallest eigenvalue reported really is
    // near-zero relative to the largest, confirming jacobi_eigen's descending order puts it last
    // as the implementation assumes.
    const auto iris = datamunge::datasets::iris();
    LLE lle(iris, kIrisFeatures, LLEOptions{2, 12, 1e-3, DistanceMetric::Euclidean});

    const auto& eig = lle.eigenvalues();
    ASSERT_FALSE(eig.empty());
    const double largest = eig.front();
    const double smallest = eig.back();
    EXPECT_GT(largest, 0.0);
    EXPECT_LT(std::abs(smallest), 1e-6 * largest)
        << "smallest eigenvalue should be near-zero relative to the largest";

    // Eigenvalues must be non-increasing (descending), matching jacobi_eigen's documented order.
    for (std::size_t i = 1; i < eig.size(); ++i) EXPECT_LE(eig[i], eig[i - 1] + 1e-9);
}

TEST(LLE, NoSpeciesCollapsesToASinglePoint) {
    // Regression test: on iris, the k-nearest-neighbor graph splits into (at least) two
    // connected components (setosa is geometrically isolated from versicolor/virginica for any
    // reasonable n_neighbors), so M has more than one trivial near-zero eigenvalue -- one per
    // connected component, not just one overall. An implementation that discards only a single
    // trailing eigenvalue leaks a degenerate "constant on one component" eigenvector into the
    // kept embedding dimensions, which pins every point of a whole disconnected component to the
    // exact same coordinate (verified: before the fix, all 50 setosa rows landed on exactly
    // (0, 0)). Note it IS mathematically expected, for a graph with exactly as many connected
    // components as n_components, that each species varies along only one of the two axes (the
    // other held exactly at 0) -- that is the correct block-diagonal-eigenvector consequence, not
    // a bug. What must never happen is every point of a species landing on the exact same overall
    // coordinate.
    const auto iris = datamunge::datasets::iris();
    LLE lle(iris, kIrisFeatures, LLEOptions{2, 12, 1e-3, DistanceMetric::Euclidean});

    std::vector<std::string> species(lle.observations());
    for (std::size_t i = 0; i < lle.observations(); ++i)
        species[i] = iris.string_at("Species", lle.kept_row_indices()[i]);

    for (const std::string& target : {std::string("setosa"), std::string("versicolor"), std::string("virginica")}) {
        double sum_sq_spread = 0.0;
        std::vector<double> mins(lle.n_components(), std::numeric_limits<double>::infinity());
        std::vector<double> maxs(lle.n_components(), -std::numeric_limits<double>::infinity());
        for (std::size_t i = 0; i < lle.observations(); ++i) {
            if (species[i] != target) continue;
            for (std::size_t c = 0; c < lle.n_components(); ++c) {
                mins[c] = std::min(mins[c], lle.embedding()(i, c));
                maxs[c] = std::max(maxs[c], lle.embedding()(i, c));
            }
        }
        for (std::size_t c = 0; c < lle.n_components(); ++c) sum_sq_spread += (maxs[c] - mins[c]) * (maxs[c] - mins[c]);
        EXPECT_GT(sum_sq_spread, 1e-6) << "species " << target << " collapsed to a single point across all dimensions";
    }
}

TEST(LLE, SpeciesAreSeparatedOnAverageInEmbeddedSpace) {
    const auto iris = datamunge::datasets::iris();
    LLE lle(iris, kIrisFeatures, LLEOptions{2, 12, 1e-3, DistanceMetric::Euclidean});

    std::vector<std::string> species(lle.observations());
    for (std::size_t i = 0; i < lle.observations(); ++i)
        species[i] = iris.string_at("Species", lle.kept_row_indices()[i]);

    double within_sum = 0.0, between_sum = 0.0;
    std::size_t within_n = 0, between_n = 0;
    for (std::size_t i = 0; i < lle.observations(); ++i) {
        for (std::size_t j = i + 1; j < lle.observations(); ++j) {
            double d2 = 0.0;
            for (std::size_t c = 0; c < lle.n_components(); ++c) {
                const double d = lle.embedding()(i, c) - lle.embedding()(j, c);
                d2 += d * d;
            }
            const double dist = std::sqrt(d2);
            if (species[i] == species[j]) {
                within_sum += dist;
                ++within_n;
            } else {
                between_sum += dist;
                ++between_n;
            }
        }
    }
    ASSERT_GT(within_n, 0u);
    ASSERT_GT(between_n, 0u);
    const double avg_within = within_sum / static_cast<double>(within_n);
    const double avg_between = between_sum / static_cast<double>(between_n);
    // LLE only preserves local structure, so a strong global separation isn't guaranteed the way
    // it would be for PCA/MDS -- but on iris, the two disconnected neighbor-graph components
    // (setosa vs. the rest) should still produce a modestly larger average between-species
    // distance than within-species distance.
    EXPECT_GT(avg_between, avg_within);
}

TEST(LLE, DropsRowsWithNullFeatures) {
    DataFrame frame;
    frame.add_column("x", std::vector<std::optional<double>>{1.0, 2.0, std::nullopt, 4.0, 5.0, 6.0, 7.0, 8.0});
    frame.add_column("y", std::vector<std::optional<double>>{2.0, 4.0, 6.0, 8.0, 9.0, 3.0, 5.0, 1.0});

    LLE lle(frame, {"x", "y"}, LLEOptions{1, 3, 1e-3, DistanceMetric::Euclidean});
    EXPECT_EQ(lle.observations(), 7u);
    EXPECT_EQ(lle.kept_row_indices(), (std::vector<std::size_t>{0, 1, 3, 4, 5, 6, 7}));
}

TEST(LLE, RejectsInvalidOptionsAndTooFewRows) {
    const auto iris = datamunge::datasets::iris();
    EXPECT_THROW(LLE(iris, {"NotAColumn"}), std::invalid_argument);
    EXPECT_THROW(LLE(iris, {"Species"}), std::invalid_argument);
    EXPECT_THROW(LLE(iris, kIrisFeatures, LLEOptions{0, 10, 1e-3, DistanceMetric::Euclidean}), std::invalid_argument);
    // n_neighbors must be at least 2.
    EXPECT_THROW(LLE(iris, kIrisFeatures, LLEOptions{2, 1, 1e-3, DistanceMetric::Euclidean}), std::invalid_argument);
    EXPECT_THROW(LLE(iris, kIrisFeatures, LLEOptions{2, 0, 1e-3, DistanceMetric::Euclidean}), std::invalid_argument);
    // n_components must be strictly less than n_neighbors.
    EXPECT_THROW(LLE(iris, kIrisFeatures, LLEOptions{10, 10, 1e-3, DistanceMetric::Euclidean}), std::invalid_argument);
    EXPECT_THROW(LLE(iris, kIrisFeatures, LLEOptions{12, 10, 1e-3, DistanceMetric::Euclidean}), std::invalid_argument);
    // Negative regularization.
    EXPECT_THROW(LLE(iris, kIrisFeatures, LLEOptions{2, 10, -0.1, DistanceMetric::Euclidean}), std::invalid_argument);

    DataFrame tiny;
    tiny.add_column("x", std::vector<double>{1.0, 2.0, 3.0});
    EXPECT_THROW(LLE(tiny, {"x"}, LLEOptions{1, 5, 1e-3, DistanceMetric::Euclidean}), std::invalid_argument);
}

TEST(LLE, PlotEmbeddingGroupedMatchesUngrouped) {
    const auto iris = datamunge::datasets::iris();
    LLE lle(iris, kIrisFeatures, LLEOptions{2, 12, 1e-3, DistanceMetric::Euclidean});

    std::vector<std::string> species(lle.observations());
    for (std::size_t i = 0; i < lle.observations(); ++i) species[i] = iris.string_at("Species", lle.kept_row_indices()[i]);

    const auto grouped = lle.plot_embedding(species);
    EXPECT_EQ(grouped.series().size(), 3u);

    std::size_t total_points = 0;
    for (const auto& series : grouped.series()) total_points += series.x.size();
    EXPECT_EQ(total_points, lle.observations());

    const auto ungrouped = lle.plot_embedding();
    EXPECT_EQ(ungrouped.series().size(), 1u);

    EXPECT_THROW(lle.plot_embedding(std::vector<std::string>{"only-one-label"}), std::invalid_argument);
    EXPECT_THROW(lle.plot_embedding(10, 0), std::out_of_range);
}
