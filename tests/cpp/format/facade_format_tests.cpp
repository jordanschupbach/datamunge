// std::format / std::print support for the top-level facade classes in datamunge.hpp (dm::LM, ...).
#include <dm/dm.hpp>

#include <gtest/gtest.h>

#include <filesystem>
#include <format>
#include <memory>
#include <string>
#include <vector>

static_assert(std::formattable<dm::LM, char>);
static_assert(std::formattable<dm::LMM, char>);
static_assert(std::formattable<dm::GLMM, char>);
static_assert(std::formattable<dm::INLAMixedModel, char>);
static_assert(std::formattable<dm::LDA, char>);
static_assert(std::formattable<dm::SVM, char>);
static_assert(std::formattable<dm::DecisionTreeClassifier, char>);
static_assert(std::formattable<dm::DecisionTreeRegressor, char>);
static_assert(std::formattable<dm::RandomForestClassifier, char>);
static_assert(std::formattable<dm::RandomForestRegressor, char>);
static_assert(std::formattable<dm::ElasticNet, char>);
static_assert(std::formattable<dm::Ridge, char>);
static_assert(std::formattable<dm::Lasso, char>);
static_assert(std::formattable<dm::KNNClassifier, char>);
static_assert(std::formattable<dm::KNNRegressor, char>);
static_assert(std::formattable<dm::KMeans, char>);
static_assert(std::formattable<dm::PCA, char>);
static_assert(std::formattable<dm::MDS, char>);
static_assert(std::formattable<dm::Isomap, char>);
static_assert(std::formattable<dm::LLE, char>);
static_assert(std::formattable<dm::TSNE, char>);
static_assert(std::formattable<dm::LaplacianEigenmaps, char>);
static_assert(std::formattable<dm::DiffusionMaps, char>);
static_assert(std::formattable<dm::KernelPCA, char>);
static_assert(std::formattable<dm::SammonMapping, char>);
static_assert(std::formattable<dm::UMAP, char>);
static_assert(std::formattable<dm::AgglomerativeClustering, char>);
static_assert(std::formattable<dm::DBSCAN, char>);
static_assert(std::formattable<dm::GBMClassifier, char>);
static_assert(std::formattable<dm::GBMRegressor, char>);
static_assert(std::formattable<dm::XGBoostClassifier, char>);
static_assert(std::formattable<dm::XGBoostRegressor, char>);
static_assert(std::formattable<dm::KernelRegression, char>);
static_assert(std::formattable<dm::GaussianProcessRegression, char>);
static_assert(std::formattable<dm::NaiveBayesClassifier, char>);
static_assert(std::formattable<dm::GLM, char>);
static_assert(std::formattable<dm::DataFrame, char>);
static_assert(std::formattable<dm::Tensor, char>);
static_assert(std::formattable<dm::ShapeLayer, char>);
static_assert(std::formattable<dm::GGPlot, char>);
static_assert(std::formattable<dm::Splom, char>);
static_assert(std::formattable<dm::Dual, char>);
static_assert(std::formattable<dm::HyperDual, char>);
static_assert(std::formattable<dm::Tape, char>);
static_assert(std::formattable<dm::Var, char>);
static_assert(std::formattable<dm::BSplineRegression, char>);
static_assert(std::formattable<dm::PSplineRegression, char>);

TEST(FacadeFormat, LmPrintsItsSummary) {
    const std::unique_ptr<dm::DataFrame> iris(dm::DataFrame::iris());
    const dm::LM model(*iris, "Sepal.Length ~ Petal.Length");
    const std::string text = std::format("{}", model);
    EXPECT_EQ(text, model.summary());
    EXPECT_NE(text.find("Petal.Length"), std::string::npos);
    EXPECT_EQ(std::format("{:>{}}", model, text.size() + 2), "  " + text);
}

TEST(FacadeFormat, OtherModelsPrintTheirSummary) {
    const std::unique_ptr<dm::DataFrame> iris(dm::DataFrame::iris());
    const dm::GLM glm(*iris, "Sepal.Length ~ Petal.Length");
    EXPECT_EQ(std::format("{}", glm), glm.summary());
    const dm::LDA lda(*iris, "Species ~ Sepal.Length + Petal.Length");
    EXPECT_EQ(std::format("{}", lda), lda.summary());
    const dm::DecisionTreeClassifier tree(*iris, "Species ~ Petal.Length + Petal.Width", 2);
    EXPECT_EQ(std::format("{}", tree), tree.summary());
    const std::vector<std::string> features{"Petal.Length", "Petal.Width"};
    const dm::PCA pca(*iris, features);
    EXPECT_EQ(std::format("{}", pca), pca.summary());
    const dm::KMeans km(*iris, features, 3, 50, 1);
    EXPECT_EQ(std::format("{}", km), km.summary());
    const dm::DBSCAN db(*iris, features, 0.3, 4);
    EXPECT_EQ(std::format("{}", db), db.summary());
}

TEST(FacadeFormat, DataAndAutodiff) {
    const std::unique_ptr<dm::DataFrame> iris(dm::DataFrame::iris());
    EXPECT_EQ(std::format("{}", *iris), iris->to_string());
    const std::unique_ptr<dm::Tensor> t(dm::Tensor::from_values({2, 2}, {1, 2, 3, 4}));
    EXPECT_EQ(std::format("{}", *t), t->to_string());
    const dm::Dual d(2.0, 1.0);
    EXPECT_EQ(std::format("{}", d.multiply(d)), "Dual(value=4, derivative=4)");
    EXPECT_EQ(std::format("{}", dm::HyperDual(1.5, 1, 0, 0)), "HyperDual(value=1.5, eps1=1, eps2=0, eps1eps2=0)");
    dm::Tape tape;
    const dm::Var x(tape, 3.0);
    EXPECT_EQ(std::format("{}", x), "Var(value=3, index=0)");
    EXPECT_EQ(std::format("{}", tape), "Tape(size=1)");
}

TEST(FacadeFormat, PlotFacadesAndSplines) {
    const std::unique_ptr<dm::DataFrame> iris(dm::DataFrame::iris());
    EXPECT_EQ(std::format("{}", dm::GGPlot(*iris, "Sepal.Length", "Sepal.Width")),
              "GGPlot rows=150 x=\"Sepal.Length\" y=\"Sepal.Width\" layers=[] theme=gray");
    dm::GGPlot gg(*iris, "Sepal.Length", "Sepal.Width", "Species");
    gg.geom_point().geom_smooth().theme_bw().facet_wrap("Species").labs("Iris", "Length");
    EXPECT_EQ(std::format("{}", gg),
              "GGPlot rows=150 x=\"Sepal.Length\" y=\"Sepal.Width\" color=\"Species\" layers=[point, smooth] "
              "theme=bw facet=\"Species\" title=\"Iris\" xlab=\"Length\"");
    EXPECT_EQ(std::format("{}", dm::Splom(*iris, {"Sepal.Length", "Petal.Length"})),
              "Splom 960x960 variables=[Sepal.Length, Petal.Length]");
    EXPECT_EQ(std::format("{}", dm::Splom(*iris, {"Sepal.Length", "Petal.Length"}).set_title("Pairs").size(600, 500)),
              "Splom 600x500 title=\"Pairs\" variables=[Sepal.Length, Petal.Length]");
    const auto xs = dm::c(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0);
    const auto ys = dm::c(0.0, 0.2, 0.4, 0.5, 0.7, 0.9, 1.0, 1.2, 1.5, 1.6, 2.0);
    const dm::BSplineRegression bs(xs, ys, 5);
    EXPECT_EQ(std::format("{}", bs).rfind("BSplineRegression(n=11, basis_size=5, degree=3, domain=[0, 1], r_squared=", 0),
              0u);
    const dm::PSplineRegression ps(xs, ys, 6, 0.5);
    EXPECT_EQ(std::format("{}", ps).rfind("PSplineRegression(n=11, basis_size=6, degree=3, penalty_order=2, lambda=0.5, ", 0),
              0u);
}

TEST(FacadeFormat, ShapeLayer) {
    const auto base = std::filesystem::path(__FILE__).parent_path() / "../../../examples/d/datamunge_gis_ex_counties_d";
    if (!std::filesystem::exists(base.string() + ".shp")) GTEST_SKIP() << "fixture shapefile not found";
    const std::unique_ptr<dm::ShapeLayer> layer(dm::ShapeLayer::read(base.string()));
    EXPECT_EQ(std::format("{}", *layer), std::format("ShapeLayer(type={}, shapes={}, bounds={})", layer->shape_type(),
                                                     layer->size(), layer->bounds()));
    EXPECT_EQ(std::format("{}", *layer).rfind("ShapeLayer(type=polygon, shapes=", 0), 0u);
}
