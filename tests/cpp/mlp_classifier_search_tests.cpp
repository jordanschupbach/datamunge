#include <gtest/gtest.h>
#include <datamunge/datamunge.hpp>
#include <cmath>
#include <limits>
#include <memory>
#include <numeric>

namespace {
using datamunge::DataFrame;
using datamunge::MLPClassifierSearch;

TEST(MLPClassifierSearch, ReproducesEachValidationFitAndFinalRefit) {
  std::unique_ptr<DataFrame> iris(datamunge::data::iris());
  const MLPClassifierSearch search(*iris, "Species", "Petal.Length", "Petal.Width");
  EXPECT_EQ(search.training_rows(), 90u);
  EXPECT_EQ(search.validation_rows(), 30u);
  EXPECT_EQ(search.test_rows(), 30u);
  std::unique_ptr<DataFrame> trials(search.trials());
  ASSERT_EQ(trials->nrows(), 12u);
  const auto [dev, test] = datamunge::stats::train_test_split(datamunge::datasets::iris(), .2, 42, "Species");
  const auto [train, validation] = datamunge::stats::train_test_split(dev, .25, 43, "Species");
  double best = std::numeric_limits<double>::infinity();
  std::size_t best_width = 0;
  datamunge::stats::MLPClassifierOptions options;
  options.activation = "tanh";
  options.l2 = .0001;
  for (std::size_t i = 0; i < trials->nrows(); ++i) {
    const auto width = static_cast<std::size_t>(trials->numeric_at("hidden_nodes", i));
    EXPECT_GE(width, 2u); EXPECT_LE(width, 24u);
    options.hidden_layer_sizes = {width};
    const datamunge::stats::MLPClassifier model(train, "Species ~ Petal.Length + Petal.Width", options);
    const auto pred = model.predict_detail(validation);
    double loss = 0;
    for (std::size_t j = 0; j < validation.nrows(); ++j) {
      const auto label = validation.get_column<std::string>("Species").values()[j];
      const auto cls = std::find(model.classes().begin(), model.classes().end(), label) - model.classes().begin();
      loss -= std::log(pred.probabilities[j][cls]);
    }
    loss /= validation.nrows();
    EXPECT_NEAR(loss, trials->numeric_at("validation_log_loss", i), 1e-12);
    if (loss < best) { best = loss; best_width = width; }
    EXPECT_NEAR(best, trials->numeric_at("best_log_loss", i), 1e-12);
    EXPECT_EQ(trials->string_at("phase", i), i < 4 ? "initial" : "bayesian");
  }
  EXPECT_EQ(search.best_hidden_nodes(), best_width);
  EXPECT_DOUBLE_EQ(search.validation_log_loss(), best);
  options.hidden_layer_sizes = {best_width};
  const datamunge::stats::MLPClassifier final(dev, "Species ~ Petal.Length + Petal.Width", options);
  EXPECT_DOUBLE_EQ(search.test_accuracy(), final.accuracy(test));
  EXPECT_EQ(search.predict(*iris), final.predict(datamunge::datasets::iris()));
  EXPECT_GE(search.test_accuracy(), .85);
  EXPECT_EQ(iris->nrows(), 150u);
  EXPECT_EQ(iris->ncols(), 5u);
}

TEST(MLPClassifierSearch, HeldOutPredictorChangesCannotAffectSelectionOrRefit) {
  std::unique_ptr<DataFrame> iris(datamunge::data::iris());
  auto core = datamunge::datasets::iris();
  std::vector<double> id(150);
  std::iota(id.begin(), id.end(), 0.0);
  core.add_column("row_id", id);
  const auto partitions = datamunge::stats::train_test_split(core, .2, 42, "Species");
  auto changed_x = core.get_column<double>("Petal.Length").values();
  for (double row : partitions.second.get_column<double>("row_id").values()) changed_x[static_cast<std::size_t>(row)] += 100;
  std::unique_ptr<DataFrame> changed(iris->mutate_numeric("Petal.Length", changed_x));
  const MLPClassifierSearch a(*iris,"Species","Petal.Length","Petal.Width",2,6,3,2,40);
  const MLPClassifierSearch b(*changed,"Species","Petal.Length","Petal.Width",2,6,3,2,40);
  std::unique_ptr<DataFrame> at(a.trials()), bt(b.trials());
  EXPECT_EQ(at->to_md(15), bt->to_md(15));
  EXPECT_EQ(a.predict(*iris), b.predict(*iris));
  EXPECT_EQ(a.plot_decision_regions(12).to_svg(), b.plot_decision_regions(12).to_svg());
}

TEST(MLPClassifierSearch, DeterministicAndCachesRoundedWidths) {
  std::unique_ptr<DataFrame> iris(datamunge::data::iris());
  const MLPClassifierSearch a(*iris, "Species", "Petal.Length", "Petal.Width", 2, 3, 5, 2, 10, 9);
  const MLPClassifierSearch b(*iris, "Species", "Petal.Length", "Petal.Width", 2, 3, 5, 2, 10, 9);
  std::unique_ptr<DataFrame> at(a.trials()), bt(b.trials());
  EXPECT_EQ(at->to_md(12), bt->to_md(12));
  EXPECT_GE(at->numeric_sum("cached"), 5.0);
  EXPECT_EQ(a.predict(*iris), b.predict(*iris));
}

TEST(MLPClassifierSearch, InitialOnlySearchAndOwnedHistory) {
  std::unique_ptr<DataFrame> iris(datamunge::data::iris());
  const MLPClassifierSearch a(*iris, "Species", "Petal.Length", "Petal.Width", 2, 3, 1, 0, 5);
  std::unique_ptr<DataFrame> t(a.trials());
  EXPECT_EQ(t->nrows(), 1u);
  t->add_numeric_column("extra", {1});
  std::unique_ptr<DataFrame> fresh(a.trials());
  EXPECT_EQ(fresh->ncols(), 6u);
}

TEST(MLPClassifierSearch, RejectsInvalidBudgetsAndColumns) {
  std::unique_ptr<DataFrame> iris(datamunge::data::iris());
  EXPECT_THROW(MLPClassifierSearch(*iris,"Species","Petal.Length","Petal.Width",0,24),std::invalid_argument);
  EXPECT_THROW(MLPClassifierSearch(*iris,"Species","Petal.Length","Petal.Width",2,2),std::invalid_argument);
  EXPECT_THROW(MLPClassifierSearch(*iris,"Species","Petal.Length","Petal.Width",2,24,0),std::invalid_argument);
  EXPECT_THROW(MLPClassifierSearch(*iris,"Species","Petal.Length","Petal.Width",2,24,1,0,0),std::invalid_argument);
  EXPECT_THROW(MLPClassifierSearch(*iris,"Species","Petal.Length","Petal.Length"),std::invalid_argument);
  EXPECT_THROW(MLPClassifierSearch(*iris,"Species","nope","Petal.Width"),std::invalid_argument);
  EXPECT_THROW(MLPClassifierSearch(*iris,"Petal.Length","Species","Petal.Width"),std::invalid_argument);
  DataFrame bad;
  bad.add_string_column("class", {"a","a","a","a","a","b","b","b","b","b"});
  bad.add_numeric_column("x", {0,1,2,3,4,5,6,7,8,std::numeric_limits<double>::infinity()});
  bad.add_numeric_column("y", {0,1,2,3,4,5,6,7,8,9});
  EXPECT_THROW(MLPClassifierSearch(bad,"class","x","y"),std::invalid_argument);
  DataFrame small;
  small.add_string_column("class", {"a","a","b","b"});
  small.add_numeric_column("x", {0,1,2,3}); small.add_numeric_column("y", {0,1,2,3});
  EXPECT_THROW(MLPClassifierSearch(small,"class","x","y"),std::invalid_argument);
}

TEST(MLPClassifierSearch, PlotHasRegionsAndOnlyDevelopmentObservations) {
  std::unique_ptr<DataFrame> iris(datamunge::data::iris());
  const MLPClassifierSearch search(*iris,"Species","Petal.Length","Petal.Width",2,4,2,1,10);
  const auto plot = search.plot_decision_regions(24);
  std::size_t points = 0;
  for (const auto& series : plot.series())
    if (series.label == "setosa" || series.label == "versicolor" || series.label == "virginica") points += series.x.size();
  EXPECT_EQ(points, 120u);
  const auto svg = plot.to_svg();
  EXPECT_NE(svg.find("Petal.Length"), std::string::npos);
  EXPECT_NE(svg.find("polygon"), std::string::npos);
  EXPECT_THROW(search.plot_decision_regions(1), std::invalid_argument);
}
} // namespace
