#include <gtest/gtest.h>

#include <datamunge/datasets/datasets.hpp>
#include <datamunge/stats/umap.hpp>

#include <cmath>
#include <map>
#include <stdexcept>
#include <utility>

using datamunge::dstruct::DataFrame;
using datamunge::stats::DistanceMetric;
using datamunge::stats::UMAP;
using datamunge::stats::UMAPOptions;

namespace {

const std::vector<std::string> kIrisFeatures = {"Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"};

} // namespace

TEST(UMAP, EmbeddingShapeAndFiniteness) {
    const auto iris = datamunge::datasets::iris();
    UMAP umap(iris, kIrisFeatures);

    EXPECT_EQ(umap.observations(), iris.nrows());
    EXPECT_EQ(umap.n_components(), 2u);
    EXPECT_EQ(umap.kept_row_indices().size(), iris.nrows());

    for (std::size_t i = 0; i < umap.observations(); ++i)
        for (std::size_t c = 0; c < umap.n_components(); ++c) EXPECT_TRUE(std::isfinite(umap.embedding()(i, c)));
}

TEST(UMAP, SmoothKnnCalibrationConverges) {
    // Every point's binary-searched sigma must satisfy the smooth-kNN target
    // sum_a exp(-max(0, dist_a - rho_i)/sigma_i) ~= log2(n_neighbors) -- re-derive the achieved sum
    // from the exposed rho/sigma diagnostics and the same nearest-neighbor distances used to fit,
    // recomputed independently here from the raw feature data.
    const auto iris = datamunge::datasets::iris();
    UMAPOptions opts;
    opts.n_neighbors = 15;
    UMAP umap(iris, kIrisFeatures, opts);

    ASSERT_EQ(umap.sigmas().size(), umap.observations());
    ASSERT_EQ(umap.rhos().size(), umap.observations());

    const double target = std::log2(static_cast<double>(opts.n_neighbors));
    const auto& kept = umap.kept_row_indices();

    for (std::size_t i = 0; i < umap.observations(); ++i) {
        std::vector<double> dists;
        for (std::size_t j = 0; j < umap.observations(); ++j) {
            if (j == i) continue;
            double sum_sq = 0.0;
            for (const auto& f : kIrisFeatures) {
                const double d = iris.double_at(f, kept[i]) - iris.double_at(f, kept[j]);
                sum_sq += d * d;
            }
            dists.push_back(std::sqrt(sum_sq));
        }
        std::sort(dists.begin(), dists.end());
        dists.resize(opts.n_neighbors);

        const double rho = umap.rhos()[i];
        const double sigma = umap.sigmas()[i];
        EXPECT_NEAR(rho, dists.front(), 1e-9);

        double achieved = 0.0;
        for (const auto d : dists) achieved += std::exp(-std::max(0.0, d - rho) / sigma);

        EXPECT_NEAR(achieved, target, 1e-3) << "point " << i << " smooth-kNN calibration did not converge";
    }
}

TEST(UMAP, SpeciesAreWellSeparated) {
    // UMAP is well known to produce tight, well-separated clusters on iris -- verify mean
    // within-species embedded distance is substantially smaller than mean between-species
    // embedded distance.
    const auto iris = datamunge::datasets::iris();
    UMAP umap(iris, kIrisFeatures);

    std::vector<std::string> species(umap.observations());
    for (std::size_t i = 0; i < umap.observations(); ++i) species[i] = iris.string_at("Species", umap.kept_row_indices()[i]);

    double within_sum = 0.0, between_sum = 0.0;
    std::size_t within_n = 0, between_n = 0;
    for (std::size_t i = 0; i < umap.observations(); ++i) {
        for (std::size_t j = i + 1; j < umap.observations(); ++j) {
            double d2 = 0.0;
            for (std::size_t c = 0; c < umap.n_components(); ++c) {
                const double diff = umap.embedding()(i, c) - umap.embedding()(j, c);
                d2 += diff * diff;
            }
            const double dist = std::sqrt(d2);
            if (species[i] == species[j]) {
                within_sum += dist;
                within_n++;
            } else {
                between_sum += dist;
                between_n++;
            }
        }
    }

    const double mean_within = within_sum / static_cast<double>(within_n);
    const double mean_between = between_sum / static_cast<double>(between_n);
    EXPECT_GT(mean_between, mean_within) << "between-species embedded distance should exceed within-species distance";
    EXPECT_GT(mean_between / mean_within, 1.3);
}

TEST(UMAP, DropsRowsWithNullFeatures) {
    DataFrame frame;
    std::vector<std::optional<double>> xs, ys;
    for (int i = 0; i < 20; ++i) {
        xs.push_back(static_cast<double>(i));
        ys.push_back(static_cast<double>(i) * 2.0);
    }
    xs[5] = std::nullopt;
    frame.add_column("x", xs);
    frame.add_column("y", ys);

    UMAPOptions opts;
    opts.n_neighbors = 3;
    opts.max_iterations = 20;
    UMAP umap(frame, {"x", "y"}, opts);
    EXPECT_EQ(umap.observations(), 19u);
    EXPECT_EQ(umap.kept_row_indices().size(), 19u);
    for (const auto idx : umap.kept_row_indices()) EXPECT_NE(idx, 5u);
}

TEST(UMAP, RejectsInvalidOptionsAndTooFewRows) {
    const auto iris = datamunge::datasets::iris();
    EXPECT_THROW(UMAP(iris, {"NotAColumn"}), std::invalid_argument);
    EXPECT_THROW(UMAP(iris, {"Species"}), std::invalid_argument);

    EXPECT_THROW(UMAP(iris, kIrisFeatures, UMAPOptions{0}), std::invalid_argument);

    UMAPOptions bad_neighbors;
    bad_neighbors.n_neighbors = 1;
    EXPECT_THROW(UMAP(iris, kIrisFeatures, bad_neighbors), std::invalid_argument);

    UMAPOptions bad_min_dist;
    bad_min_dist.min_dist = 0.0;
    EXPECT_THROW(UMAP(iris, kIrisFeatures, bad_min_dist), std::invalid_argument);

    UMAPOptions bad_lr;
    bad_lr.learning_rate = 0.0;
    EXPECT_THROW(UMAP(iris, kIrisFeatures, bad_lr), std::invalid_argument);

    UMAPOptions bad_epochs;
    bad_epochs.max_iterations = 0;
    EXPECT_THROW(UMAP(iris, kIrisFeatures, bad_epochs), std::invalid_argument);

    DataFrame tiny;
    tiny.add_column("x", std::vector<double>{1.0, 2.0, 3.0});
    UMAPOptions tiny_opts;
    tiny_opts.n_neighbors = 5;
    EXPECT_THROW(UMAP(tiny, {"x"}, tiny_opts), std::invalid_argument);
}

TEST(UMAP, DeterministicGivenSameSeed) {
    const auto iris = datamunge::datasets::iris();
    UMAPOptions opts;
    opts.max_iterations = 50;
    UMAP umap_a(iris, kIrisFeatures, opts);
    UMAP umap_b(iris, kIrisFeatures, opts);

    for (std::size_t i = 0; i < umap_a.observations(); ++i)
        for (std::size_t c = 0; c < umap_a.n_components(); ++c)
            EXPECT_DOUBLE_EQ(umap_a.embedding()(i, c), umap_b.embedding()(i, c));
}

TEST(UMAP, PlotEmbeddingGroupedMatchesUngrouped) {
    const auto iris = datamunge::datasets::iris();
    UMAPOptions opts;
    opts.max_iterations = 50;
    UMAP umap(iris, kIrisFeatures, opts);

    std::vector<std::string> species(umap.observations());
    for (std::size_t i = 0; i < umap.observations(); ++i) species[i] = iris.string_at("Species", umap.kept_row_indices()[i]);

    const auto grouped = umap.plot_embedding(species);
    EXPECT_EQ(grouped.series().size(), 3u);

    std::size_t total_points = 0;
    for (const auto& series : grouped.series()) total_points += series.x.size();
    EXPECT_EQ(total_points, umap.observations());

    const auto ungrouped = umap.plot_embedding();
    EXPECT_EQ(ungrouped.series().size(), 1u);

    EXPECT_THROW(umap.plot_embedding(std::vector<std::string>{"only-one-label"}), std::invalid_argument);
    EXPECT_THROW(umap.plot_embedding(10, 0), std::out_of_range);
}
