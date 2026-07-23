#include <gtest/gtest.h>

#include <datamunge/datasets/datasets.hpp>
#include <datamunge/stats/tsne.hpp>

#include <cmath>
#include <stdexcept>

using datamunge::dstruct::DataFrame;
using datamunge::stats::DistanceMetric;
using datamunge::stats::TSNE;
using datamunge::stats::TSNEOptions;

namespace {

const std::vector<std::string> kIrisFeatures = {"Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"};

} // namespace

TEST(TSNE, EmbeddingHasCorrectShapeAndIsFinite) {
  const auto iris = datamunge::datasets::iris();
  TSNEOptions opts;
  opts.max_iterations = 300;
  TSNE tsne(iris, kIrisFeatures, opts);

  EXPECT_EQ(tsne.observations(), iris.nrows());
  EXPECT_EQ(tsne.n_components(), 2u);

  for (std::size_t i = 0; i < tsne.observations(); ++i) {
    for (std::size_t c = 0; c < tsne.n_components(); ++c) {
      EXPECT_TRUE(std::isfinite(tsne.embedding()(i, c))) << "row " << i << " dim " << c << " is not finite";
    }
  }
}

TEST(TSNE, PerplexityBinarySearchConverges) {
  const auto iris = datamunge::datasets::iris();
  TSNEOptions opts;
  opts.perplexity = 30.0;
  opts.max_iterations = 10; // convergence of the beta search happens before the gradient descent loop
  TSNE tsne(iris, kIrisFeatures, opts);

  const auto& achieved = tsne.achieved_perplexity();
  ASSERT_EQ(achieved.size(), tsne.observations());
  for (std::size_t i = 0; i < achieved.size(); ++i) {
    EXPECT_TRUE(std::isfinite(achieved[i]));
    EXPECT_NEAR(achieved[i], opts.perplexity, 0.01) << "point " << i << " perplexity calibration did not converge";
  }
}

TEST(TSNE, SeparatesIrisSpecies) {
  const auto iris = datamunge::datasets::iris();
  TSNEOptions opts;
  opts.seed = 42;
  opts.max_iterations = 1000;
  TSNE tsne(iris, kIrisFeatures, opts);

  std::vector<std::string> species(tsne.observations());
  for (std::size_t i = 0; i < tsne.observations(); ++i)
    species[i] = iris.string_at("Species", tsne.kept_row_indices()[i]);

  double within_sum = 0.0, between_sum = 0.0;
  std::size_t within_n = 0, between_n = 0;
  for (std::size_t i = 0; i < tsne.observations(); ++i) {
    for (std::size_t j = i + 1; j < tsne.observations(); ++j) {
      const double dx = tsne.embedding()(i, 0) - tsne.embedding()(j, 0);
      const double dy = tsne.embedding()(i, 1) - tsne.embedding()(j, 1);
      const double d = std::sqrt(dx * dx + dy * dy);
      if (species[i] == species[j]) {
        within_sum += d;
        ++within_n;
      } else {
        between_sum += d;
        ++between_n;
      }
    }
  }
  const double avg_within = within_sum / static_cast<double>(within_n);
  const double avg_between = between_sum / static_cast<double>(between_n);
  EXPECT_LT(avg_within, avg_between)
      << "iris species should be visibly separated in the t-SNE embedding (within=" << avg_within
      << ", between=" << avg_between << ")";
  // t-SNE's famous strong separation on iris -- expect a large margin, not just a marginal one.
  EXPECT_LT(avg_within * 2.0, avg_between);
}

TEST(TSNE, DropsRowsWithNullFeatures) {
  DataFrame frame;
  frame.add_column("x", std::vector<std::optional<double>>{1.0, 2.0, std::nullopt, 4.0, 5.0, 6.0, 7.0, 8.0});
  frame.add_column("y", std::vector<std::optional<double>>{2.0, 4.0, 6.0, 8.0, 9.0, 3.0, 1.0, 5.0});

  TSNEOptions opts;
  opts.perplexity = 2.0;
  opts.max_iterations = 10;
  TSNE tsne(frame, {"x", "y"}, opts);
  EXPECT_EQ(tsne.observations(), 7u);
  EXPECT_EQ(tsne.kept_row_indices(), (std::vector<std::size_t>{0, 1, 3, 4, 5, 6, 7}));
}

TEST(TSNE, RejectsInvalidOptionsAndTooFewRows) {
  const auto iris = datamunge::datasets::iris();
  EXPECT_THROW(TSNE(iris, {"NotAColumn"}), std::invalid_argument);
  EXPECT_THROW(TSNE(iris, {"Species"}), std::invalid_argument);
  EXPECT_THROW(TSNE(iris, kIrisFeatures, TSNEOptions{0}), std::invalid_argument);

  TSNEOptions non_positive_perplexity;
  non_positive_perplexity.perplexity = 0.0;
  EXPECT_THROW(TSNE(iris, kIrisFeatures, non_positive_perplexity), std::invalid_argument);

  TSNEOptions non_positive_lr;
  non_positive_lr.learning_rate = 0.0;
  EXPECT_THROW(TSNE(iris, kIrisFeatures, non_positive_lr), std::invalid_argument);

  // Perplexity must be strictly less than the number of complete observations (150 for iris).
  TSNEOptions too_large_perplexity;
  too_large_perplexity.perplexity = 150.0;
  EXPECT_THROW(TSNE(iris, kIrisFeatures, too_large_perplexity), std::invalid_argument);

  DataFrame tiny;
  tiny.add_column("x", std::vector<double>{1.0});
  EXPECT_THROW(TSNE(tiny, {"x"}), std::invalid_argument);
}

TEST(TSNE, PlotEmbeddingGroupedMatchesUngrouped) {
  const auto iris = datamunge::datasets::iris();
  TSNEOptions opts;
  opts.max_iterations = 50;
  TSNE tsne(iris, kIrisFeatures, opts);

  std::vector<std::string> species(tsne.observations());
  for (std::size_t i = 0; i < tsne.observations(); ++i)
    species[i] = iris.string_at("Species", tsne.kept_row_indices()[i]);

  const auto grouped = tsne.plot_embedding(species);
  EXPECT_EQ(grouped.series().size(), 3u);

  std::size_t total_points = 0;
  for (const auto& series : grouped.series()) total_points += series.x.size();
  EXPECT_EQ(total_points, tsne.observations());

  const auto ungrouped = tsne.plot_embedding();
  EXPECT_EQ(ungrouped.series().size(), 1u);

  EXPECT_THROW(tsne.plot_embedding(std::vector<std::string>{"only-one-label"}), std::invalid_argument);
  EXPECT_THROW(tsne.plot_embedding(10, 0), std::out_of_range);
}
