// std::format / std::print support for datamunge::stats types.
#include <dm/dm.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <format>
#include <memory>
#include <string>
#include <vector>

// Every formatted stats type.
static_assert(std::formattable<dm::stats::LinkageCriterion, char>);
static_assert(std::formattable<dm::stats::SplitCriterion, char>);
static_assert(std::formattable<dm::stats::TrendType, char>);
static_assert(std::formattable<dm::stats::SeasonalType, char>);
static_assert(std::formattable<dm::stats::ExprOp, char>);
static_assert(std::formattable<dm::stats::ResolvedColumnKind, char>);
static_assert(std::formattable<dm::stats::ResponseKind, char>);
static_assert(std::formattable<dm::stats::GLMFamily, char>);
static_assert(std::formattable<dm::stats::GLMPredictionInterval, char>);
static_assert(std::formattable<dm::stats::GLMMFamily, char>);
static_assert(std::formattable<dm::stats::Alternative, char>);
static_assert(std::formattable<dm::stats::INLAMixedModelFamily, char>);
static_assert(std::formattable<dm::stats::KernelRegressionKernel, char>);
static_assert(std::formattable<dm::stats::DistanceMetric, char>);
static_assert(std::formattable<dm::stats::PredictionInterval, char>);
static_assert(std::formattable<dm::stats::PAdjustMethod, char>);
static_assert(std::formattable<dm::stats::SVMKernel, char>);
static_assert(std::formattable<dm::stats::AgglomerativeClusteringOptions, char>);
static_assert(std::formattable<dm::stats::AgglomerativeClustering::Merge, char>);
static_assert(std::formattable<dm::stats::ARIMAOptions, char>);
static_assert(std::formattable<dm::stats::BootstrapOptions, char>);
static_assert(std::formattable<dm::stats::BootstrapResult, char>);
static_assert(std::formattable<dm::stats::CNNClassifierOptions, char>);
static_assert(std::formattable<dm::stats::CNNClassifierPrediction, char>);
static_assert(std::formattable<dm::stats::ConformalClassifierOptions, char>);
static_assert(std::formattable<dm::stats::SplitConformalClassifier::Evaluation, char>);
static_assert(std::formattable<dm::stats::ConformalRegressorOptions, char>);
static_assert(std::formattable<dm::stats::SplitConformalRegressor::Evaluation, char>);
static_assert(std::formattable<dm::stats::ConformalCoverage, char>);
static_assert(std::formattable<dm::stats::DBSCANOptions, char>);
static_assert(std::formattable<dm::stats::DecisionTreeClassifierOptions, char>);
static_assert(std::formattable<dm::stats::DecisionTreeClassifierPrediction, char>);
static_assert(std::formattable<dm::stats::DecisionTreeRegressorOptions, char>);
static_assert(std::formattable<dm::stats::DiffusionMapsOptions, char>);
static_assert(std::formattable<dm::stats::ElasticNetOptions, char>);
static_assert(std::formattable<dm::stats::RidgeOptions, char>);
static_assert(std::formattable<dm::stats::LassoOptions, char>);
static_assert(std::formattable<dm::stats::ExponentialSmoothingOptions, char>);
static_assert(std::formattable<dm::stats::FactorAnalysisOptions, char>);
static_assert(std::formattable<dm::stats::TermGroup, char>);
static_assert(std::formattable<dm::stats::GaussianProcessRegressionOptions, char>);
static_assert(std::formattable<dm::stats::GaussianProcessRegressionPrediction, char>);
static_assert(std::formattable<dm::stats::GBMClassifierOptions, char>);
static_assert(std::formattable<dm::stats::GBMClassifierPrediction, char>);
static_assert(std::formattable<dm::stats::GBMRegressorOptions, char>);
static_assert(std::formattable<dm::stats::GLMOptions, char>);
static_assert(std::formattable<dm::stats::GLMPrediction, char>);
static_assert(std::formattable<dm::stats::GLM::Interval, char>);
static_assert(std::formattable<dm::stats::GLMMOptions, char>);
static_assert(std::formattable<dm::stats::ICAOptions, char>);
static_assert(std::formattable<dm::stats::ImageExplanationOptions, char>);
static_assert(std::formattable<dm::stats::INLAMixedModelOptions, char>);
static_assert(std::formattable<dm::stats::IsomapOptions, char>);
static_assert(std::formattable<dm::stats::JackknifeResult, char>);
static_assert(std::formattable<dm::stats::DeleteDJackknifeOptions, char>);
static_assert(std::formattable<dm::stats::DeleteDJackknifeResult, char>);
static_assert(std::formattable<dm::stats::KernelPCAOptions, char>);
static_assert(std::formattable<dm::stats::KernelRegressionOptions, char>);
static_assert(std::formattable<dm::stats::KernelRidgeRegressionOptions, char>);
static_assert(std::formattable<dm::stats::KMeansOptions, char>);
static_assert(std::formattable<dm::stats::KNNClassifierOptions, char>);
static_assert(std::formattable<dm::stats::KNNClassifierPrediction, char>);
static_assert(std::formattable<dm::stats::KNNRegressorOptions, char>);
static_assert(std::formattable<dm::stats::KNNNeighbors, char>);
static_assert(std::formattable<dm::stats::LaplacianEigenmapsOptions, char>);
static_assert(std::formattable<dm::stats::LDAOptions, char>);
static_assert(std::formattable<dm::stats::LDAPrediction, char>);
static_assert(std::formattable<dm::stats::LLEOptions, char>);
static_assert(std::formattable<dm::stats::LmOptions, char>);
static_assert(std::formattable<dm::stats::LmPrediction, char>);
static_assert(std::formattable<dm::stats::AnovaRow, char>);
static_assert(std::formattable<dm::stats::NestedAnova, char>);
static_assert(std::formattable<dm::stats::LM::Interval, char>);
static_assert(std::formattable<dm::stats::LMMOptions, char>);
static_assert(std::formattable<dm::stats::MDSOptions, char>);
static_assert(std::formattable<dm::stats::MLPClassifierOptions, char>);
static_assert(std::formattable<dm::stats::MLPClassifierPrediction, char>);
static_assert(std::formattable<dm::stats::NaiveBayesClassifierOptions, char>);
static_assert(std::formattable<dm::stats::NaiveBayesClassifierPrediction, char>);
static_assert(std::formattable<dm::stats::PCAOptions, char>);
static_assert(std::formattable<dm::stats::PermutationTestOptions, char>);
static_assert(std::formattable<dm::stats::PermutationTestResult, char>);
static_assert(std::formattable<dm::stats::QuantileRegressionOptions, char>);
static_assert(std::formattable<dm::stats::QuantileCoefficientTable, char>);
static_assert(std::formattable<dm::stats::QuantileProcess, char>);
static_assert(std::formattable<dm::stats::QuantileRegressionMethod, char>);
static_assert(std::formattable<dm::stats::QuantileStandardErrors, char>);
static_assert(std::formattable<dm::stats::RandomForestClassifierOptions, char>);
static_assert(std::formattable<dm::stats::RandomForestClassifierPrediction, char>);
static_assert(std::formattable<dm::stats::RandomForestStrength, char>);
static_assert(std::formattable<dm::stats::RandomForestPermutationImportance, char>);
static_assert(std::formattable<dm::stats::RandomForestRegressorOptions, char>);
static_assert(std::formattable<dm::stats::SammonMappingOptions, char>);
static_assert(std::formattable<dm::stats::SVMOptions, char>);
static_assert(std::formattable<dm::stats::SVMPrediction, char>);
static_assert(std::formattable<dm::stats::TSNEOptions, char>);
static_assert(std::formattable<dm::stats::UMAPOptions, char>);
static_assert(std::formattable<dm::stats::UStatisticOptions, char>);
static_assert(std::formattable<dm::stats::UStatisticResult, char>);
static_assert(std::formattable<dm::stats::VStatisticOptions, char>);
static_assert(std::formattable<dm::stats::VStatisticResult, char>);
static_assert(std::formattable<dm::stats::XGBoostClassifierOptions, char>);
static_assert(std::formattable<dm::stats::XGBoostClassifierPrediction, char>);
static_assert(std::formattable<dm::stats::XGBoostRegressorOptions, char>);
static_assert(std::formattable<dm::stats::HypothesisTestResult, char>);
static_assert(std::formattable<dm::stats::ExprNode, char>);
static_assert(std::formattable<dm::stats::ResolvedColumn, char>);
static_assert(std::formattable<dm::stats::DesignMatrixData, char>);
static_assert(std::formattable<dm::stats::ClassificationDesignData, char>);
static_assert(std::formattable<dm::stats::DesignInfo, char>);
static_assert(std::formattable<dm::stats::Formula, char>);
static_assert(std::formattable<dm::stats::ARIMA, char>);
static_assert(std::formattable<dm::stats::ExponentialSmoothing, char>);
static_assert(std::formattable<dm::stats::SplitConformalClassifier, char>);
static_assert(std::formattable<dm::stats::SplitConformalRegressor, char>);
static_assert(std::formattable<dm::stats::LimeImageExplainer, char>);
static_assert(std::formattable<dm::stats::ShapImageExplainer, char>);
static_assert(std::formattable<dm::stats::Bootstrap, char>);
static_assert(std::formattable<dm::stats::UStatistic, char>);
static_assert(std::formattable<dm::stats::VStatistic, char>);
static_assert(std::formattable<dm::stats::AgglomerativeClustering, char>);
static_assert(std::formattable<dm::stats::CNNClassifier, char>);
static_assert(std::formattable<dm::stats::DBSCAN, char>);
static_assert(std::formattable<dm::stats::DecisionTreeClassifier, char>);
static_assert(std::formattable<dm::stats::DecisionTreeRegressor, char>);
static_assert(std::formattable<dm::stats::DiffusionMaps, char>);
static_assert(std::formattable<dm::stats::ElasticNet, char>);
static_assert(std::formattable<dm::stats::Ridge, char>);
static_assert(std::formattable<dm::stats::Lasso, char>);
static_assert(std::formattable<dm::stats::FactorAnalysis, char>);
static_assert(std::formattable<dm::stats::GaussianProcessRegression, char>);
static_assert(std::formattable<dm::stats::GBMClassifier, char>);
static_assert(std::formattable<dm::stats::GBMRegressor, char>);
static_assert(std::formattable<dm::stats::GLM, char>);
static_assert(std::formattable<dm::stats::GLMM, char>);
static_assert(std::formattable<dm::stats::ICA, char>);
static_assert(std::formattable<dm::stats::INLAMixedModel, char>);
static_assert(std::formattable<dm::stats::Isomap, char>);
static_assert(std::formattable<dm::stats::KernelPCA, char>);
static_assert(std::formattable<dm::stats::KernelRegression, char>);
static_assert(std::formattable<dm::stats::KernelRidgeRegression, char>);
static_assert(std::formattable<dm::stats::KMeans, char>);
static_assert(std::formattable<dm::stats::KNNClassifier, char>);
static_assert(std::formattable<dm::stats::KNNRegressor, char>);
static_assert(std::formattable<dm::stats::LaplacianEigenmaps, char>);
static_assert(std::formattable<dm::stats::LDA, char>);
static_assert(std::formattable<dm::stats::LLE, char>);
static_assert(std::formattable<dm::stats::LM, char>);
static_assert(std::formattable<dm::stats::LMM, char>);
static_assert(std::formattable<dm::stats::MDS, char>);
static_assert(std::formattable<dm::stats::MLPClassifier, char>);
static_assert(std::formattable<dm::stats::NaiveBayesClassifier, char>);
static_assert(std::formattable<dm::stats::PCA, char>);
static_assert(std::formattable<dm::stats::QuantileRegression, char>);
static_assert(std::formattable<dm::stats::RandomForestClassifier, char>);
static_assert(std::formattable<dm::stats::RandomForestRegressor, char>);
static_assert(std::formattable<dm::stats::SammonMapping, char>);
static_assert(std::formattable<dm::stats::SVM, char>);
static_assert(std::formattable<dm::stats::TSNE, char>);
static_assert(std::formattable<dm::stats::UMAP, char>);
static_assert(std::formattable<dm::stats::XGBoostClassifier, char>);
static_assert(std::formattable<dm::stats::XGBoostRegressor, char>);

TEST(StatsFormat, EnumsPrintEnumeratorNames) {
    EXPECT_EQ(std::format("{}", dm::stats::LinkageCriterion::Single), "Single");
    EXPECT_EQ(std::format("{}", dm::stats::LinkageCriterion::Complete), "Complete");
    EXPECT_EQ(std::format("{}", dm::stats::LinkageCriterion::Average), "Average");
    EXPECT_EQ(std::format("{}", dm::stats::LinkageCriterion::Ward), "Ward");
    EXPECT_EQ(std::format("{}", dm::stats::SplitCriterion::Gini), "Gini");
    EXPECT_EQ(std::format("{}", dm::stats::SplitCriterion::Entropy), "Entropy");
    EXPECT_EQ(std::format("{}", dm::stats::TrendType::None), "None");
    EXPECT_EQ(std::format("{}", dm::stats::TrendType::Additive), "Additive");
    EXPECT_EQ(std::format("{}", dm::stats::TrendType::AdditiveDamped), "AdditiveDamped");
    EXPECT_EQ(std::format("{}", dm::stats::SeasonalType::None), "None");
    EXPECT_EQ(std::format("{}", dm::stats::SeasonalType::Additive), "Additive");
    EXPECT_EQ(std::format("{}", dm::stats::SeasonalType::Multiplicative), "Multiplicative");
    EXPECT_EQ(std::format("{}", dm::stats::ExprOp::Literal), "Literal");
    EXPECT_EQ(std::format("{}", dm::stats::ExprOp::Column), "Column");
    EXPECT_EQ(std::format("{}", dm::stats::ExprOp::Add), "Add");
    EXPECT_EQ(std::format("{}", dm::stats::ExprOp::Subtract), "Subtract");
    EXPECT_EQ(std::format("{}", dm::stats::ExprOp::Multiply), "Multiply");
    EXPECT_EQ(std::format("{}", dm::stats::ExprOp::Divide), "Divide");
    EXPECT_EQ(std::format("{}", dm::stats::ExprOp::Power), "Power");
    EXPECT_EQ(std::format("{}", dm::stats::ExprOp::Negate), "Negate");
    EXPECT_EQ(std::format("{}", dm::stats::ResolvedColumnKind::Numeric), "Numeric");
    EXPECT_EQ(std::format("{}", dm::stats::ResolvedColumnKind::CategoricalDummy), "CategoricalDummy");
    EXPECT_EQ(std::format("{}", dm::stats::ResolvedColumnKind::Function), "Function");
    EXPECT_EQ(std::format("{}", dm::stats::ResolvedColumnKind::BSpline), "BSpline");
    EXPECT_EQ(std::format("{}", dm::stats::ResolvedColumnKind::Interaction), "Interaction");
    EXPECT_EQ(std::format("{}", dm::stats::ResolvedColumnKind::Expression), "Expression");
    EXPECT_EQ(std::format("{}", dm::stats::ResponseKind::Numeric), "Numeric");
    EXPECT_EQ(std::format("{}", dm::stats::ResponseKind::Categorical), "Categorical");
    EXPECT_EQ(std::format("{}", dm::stats::GLMFamily::Gaussian), "Gaussian");
    EXPECT_EQ(std::format("{}", dm::stats::GLMFamily::Binomial), "Binomial");
    EXPECT_EQ(std::format("{}", dm::stats::GLMFamily::Poisson), "Poisson");
    EXPECT_EQ(std::format("{}", dm::stats::GLMFamily::Gamma), "Gamma");
    EXPECT_EQ(std::format("{}", dm::stats::GLMPredictionInterval::None), "None");
    EXPECT_EQ(std::format("{}", dm::stats::GLMPredictionInterval::Confidence), "Confidence");
    EXPECT_EQ(std::format("{}", dm::stats::GLMMFamily::Binomial), "Binomial");
    EXPECT_EQ(std::format("{}", dm::stats::GLMMFamily::Poisson), "Poisson");
    EXPECT_EQ(std::format("{}", dm::stats::Alternative::TwoSided), "TwoSided");
    EXPECT_EQ(std::format("{}", dm::stats::Alternative::Less), "Less");
    EXPECT_EQ(std::format("{}", dm::stats::Alternative::Greater), "Greater");
    EXPECT_EQ(std::format("{}", dm::stats::INLAMixedModelFamily::Gaussian), "Gaussian");
    EXPECT_EQ(std::format("{}", dm::stats::INLAMixedModelFamily::Binomial), "Binomial");
    EXPECT_EQ(std::format("{}", dm::stats::INLAMixedModelFamily::Poisson), "Poisson");
    EXPECT_EQ(std::format("{}", dm::stats::KernelRegressionKernel::Gaussian), "Gaussian");
    EXPECT_EQ(std::format("{}", dm::stats::KernelRegressionKernel::Epanechnikov), "Epanechnikov");
    EXPECT_EQ(std::format("{}", dm::stats::KernelRegressionKernel::Uniform), "Uniform");
    EXPECT_EQ(std::format("{}", dm::stats::KernelRegressionKernel::Triangular), "Triangular");
    EXPECT_EQ(std::format("{}", dm::stats::DistanceMetric::Euclidean), "Euclidean");
    EXPECT_EQ(std::format("{}", dm::stats::DistanceMetric::Manhattan), "Manhattan");
    EXPECT_EQ(std::format("{}", dm::stats::PredictionInterval::None), "None");
    EXPECT_EQ(std::format("{}", dm::stats::PredictionInterval::Confidence), "Confidence");
    EXPECT_EQ(std::format("{}", dm::stats::PredictionInterval::Prediction), "Prediction");
    EXPECT_EQ(std::format("{}", dm::stats::PAdjustMethod::Bonferroni), "Bonferroni");
    EXPECT_EQ(std::format("{}", dm::stats::PAdjustMethod::Holm), "Holm");
    EXPECT_EQ(std::format("{}", dm::stats::PAdjustMethod::Hochberg), "Hochberg");
    EXPECT_EQ(std::format("{}", dm::stats::PAdjustMethod::Hommel), "Hommel");
    EXPECT_EQ(std::format("{}", dm::stats::PAdjustMethod::BH), "BH");
    EXPECT_EQ(std::format("{}", dm::stats::PAdjustMethod::BY), "BY");
    EXPECT_EQ(std::format("{}", dm::stats::PAdjustMethod::None), "None");
    EXPECT_EQ(std::format("{}", dm::stats::SVMKernel::Linear), "Linear");
    EXPECT_EQ(std::format("{}", dm::stats::SVMKernel::Polynomial), "Polynomial");
    EXPECT_EQ(std::format("{}", dm::stats::SVMKernel::Radial), "Radial");
    EXPECT_EQ(std::format("{}", dm::stats::SVMKernel::Sigmoid), "Sigmoid");
    EXPECT_EQ(std::format("{}", static_cast<dm::stats::PAdjustMethod>(42)), "PAdjustMethod(42)");
    EXPECT_EQ(std::format("{:>6}", dm::stats::PAdjustMethod::Holm), "  Holm");
}

TEST(StatsFormat, DefaultOptionsAndResultsPrintOneLine) {
    EXPECT_EQ(std::format("{}", dm::stats::AgglomerativeClusteringOptions{}),
              "AgglomerativeClusteringOptions(n_clusters=2, linkage=Ward, metric=Euclidean)");
    EXPECT_EQ(std::format("{}", dm::stats::AgglomerativeClustering::Merge{}),
              "Merge(cluster_a=0, cluster_b=0, distance=0, size=0)");
    EXPECT_EQ(std::format("{}", dm::stats::ARIMAOptions{}),
              "ARIMAOptions(p=0, d=0, q=0, seasonal_p=0, seasonal_d=0, seasonal_q=0, seasonal_period=0, include_mean=true, de_population_size=60, de_max_generations=400, coefficient_bound=1.5, seed=42)");
    EXPECT_EQ(std::format("{}", dm::stats::BootstrapOptions{}),
              "BootstrapOptions(num_resamples=2000, confidence_level=0.95, seed=42)");
    EXPECT_EQ(std::format("{}", dm::stats::BootstrapResult{}),
              "BootstrapResult(estimate=0, bias=0, standard_error=0, confidence_level=0.95, percentile_interval=[0, 0], basic_interval=[0, 0], normal_interval=[0, 0], bca_interval=[0, 0], replicates=[])");
    EXPECT_EQ(std::format("{}", dm::stats::CNNClassifierOptions{}),
              "CNNClassifierOptions(n_filters=8, kernel_size=3, pool_size=2, dense_hidden=32, learning_rate=0.05, max_epochs=40, batch_size=16, l2=0, seed=42)");
    EXPECT_EQ(std::format("{}", dm::stats::CNNClassifierPrediction{}),
              "CNNClassifierPrediction(class_label=[], probabilities=[])");
    EXPECT_EQ(std::format("{}", dm::stats::ConformalClassifierOptions{}),
              "ConformalClassifierOptions(alpha=0.1, score=\"lac\", class_conditional=false, nonempty=false)");
    EXPECT_EQ(std::format("{}", dm::stats::SplitConformalClassifier::Evaluation{}),
              "Evaluation(coverage=0, average_set_size=0, empty_rate=0, set_sizes=[], class_coverage=[], size_counts=[])");
    EXPECT_EQ(std::format("{}", dm::stats::ConformalRegressorOptions{}),
              "ConformalRegressorOptions(alpha=0.1)");
    EXPECT_EQ(std::format("{}", dm::stats::SplitConformalRegressor::Evaluation{}),
              "Evaluation(coverage=0, average_width=0, widths=[])");
    EXPECT_EQ(std::format("{}", dm::stats::DBSCANOptions{}),
              "DBSCANOptions(eps=0.5, min_samples=5, metric=Euclidean)");
    EXPECT_EQ(std::format("{}", dm::stats::DecisionTreeClassifierOptions{}),
              "DecisionTreeClassifierOptions(max_depth=5, min_samples_split=2, min_samples_leaf=1, min_impurity_decrease=0, criterion=Gini, max_features=0, random_seed=0)");
    EXPECT_EQ(std::format("{}", dm::stats::DecisionTreeClassifierPrediction{}),
              "DecisionTreeClassifierPrediction(class_label=[], probability=[])");
    EXPECT_EQ(std::format("{}", dm::stats::DecisionTreeRegressorOptions{}),
              "DecisionTreeRegressorOptions(max_depth=5, min_samples_split=2, min_samples_leaf=1, min_impurity_decrease=0, max_features=0, random_seed=0)");
    EXPECT_EQ(std::format("{}", dm::stats::DiffusionMapsOptions{}),
              "DiffusionMapsOptions(n_components=2, heat_kernel_epsilon=1, alpha=0.5, diffusion_time=1)");
    EXPECT_EQ(std::format("{}", dm::stats::ElasticNetOptions{}),
              "ElasticNetOptions(alpha=0.5, lambda=-1, n_lambda=100, cv_folds=5, max_iter=10000, tol=1e-07, seed=42, standardize=true)");
    EXPECT_EQ(std::format("{}", dm::stats::RidgeOptions{}),
              "RidgeOptions(lambda=-1, n_lambda=100, cv_folds=5, max_iter=10000, tol=1e-07, seed=42, standardize=true)");
    EXPECT_EQ(std::format("{}", dm::stats::LassoOptions{}),
              "LassoOptions(lambda=-1, n_lambda=100, cv_folds=5, max_iter=10000, tol=1e-07, seed=42, standardize=true)");
    EXPECT_EQ(std::format("{}", dm::stats::ExponentialSmoothingOptions{}),
              "ExponentialSmoothingOptions(trend=None, seasonal=None, seasonal_period=0, alpha=-1, beta=-1, gamma=-1, phi=-1, de_population_size=40, de_max_generations=250, seed=42)");
    EXPECT_EQ(std::format("{}", dm::stats::FactorAnalysisOptions{}),
              "FactorAnalysisOptions(n_factors=2, standardize=true, rotation=\"varimax\", max_iter=1000, tol=1e-06)");
    EXPECT_EQ(std::format("{}", dm::stats::TermGroup{}),
              "TermGroup(label=\"\", column_indices=[])");
    EXPECT_EQ(std::format("{}", dm::stats::GaussianProcessRegressionOptions{}),
              "GaussianProcessRegressionOptions(length_scale=-1, noise_ratio=-1, n_length_scale_grid=20, n_noise_grid=15, standardize=true, jitter_ratio=1e-08)");
    EXPECT_EQ(std::format("{}", dm::stats::GaussianProcessRegressionPrediction{}),
              "GaussianProcessRegressionPrediction(fit=[], se_fit=[], lower=[], upper=[])");
    EXPECT_EQ(std::format("{}", dm::stats::GBMClassifierOptions{}),
              "GBMClassifierOptions(n_trees=100, learning_rate=0.1, max_depth=3, min_samples_split=2, min_samples_leaf=1, subsample=1, seed=42)");
    EXPECT_EQ(std::format("{}", dm::stats::GBMClassifierPrediction{}),
              "GBMClassifierPrediction(class_label=[], probability=[])");
    EXPECT_EQ(std::format("{}", dm::stats::GBMRegressorOptions{}),
              "GBMRegressorOptions(n_trees=100, learning_rate=0.1, max_depth=3, min_samples_split=2, min_samples_leaf=1, subsample=1, seed=42)");
    EXPECT_EQ(std::format("{}", dm::stats::GLMOptions{}),
              "GLMOptions(family=Gaussian, weights_column=none, max_iter=25, tol=1e-08)");
    EXPECT_EQ(std::format("{}", dm::stats::GLMPrediction{}),
              "GLMPrediction(fit=[], se_fit=[], lower=[], upper=[])");
    EXPECT_EQ(std::format("{}", dm::stats::GLM::Interval{}),
              "Interval(lower=0, upper=0)");
    EXPECT_EQ(std::format("{}", dm::stats::GLMMOptions{}),
              "GLMMOptions(family=Binomial, max_iterations=20, tol=1e-06, de_population_size=30, de_max_generations=150, theta_bound=5, seed=42, fixed_random_intercept_sd=none)");
    EXPECT_EQ(std::format("{}", dm::stats::ICAOptions{}),
              "ICAOptions(n_components=2, contrast=\"logcosh\", max_iter=200, tol=1e-06, seed=42)");
    EXPECT_EQ(std::format("{}", dm::stats::ImageExplanationOptions{}),
              "ImageExplanationOptions(patch_rows=4, patch_cols=4, baseline=0, n_samples=800, lime_kernel_width=0.25, lime_l2=0.001, seed=0)");
    EXPECT_EQ(std::format("{}", dm::stats::INLAMixedModelOptions{}),
              "INLAMixedModelOptions(family=Gaussian, strategy=Grid, fixed_effect_prior_sd=1000, lambda_diag_epsilon=0.001, lambda_bound=5, log_sigma_search_radius=3, grid_points_per_dim=7, grid_span=4, mode_population_size=40, mode_max_generations=200, seed=42)");
    EXPECT_EQ(std::format("{}", dm::stats::IsomapOptions{}),
              "IsomapOptions(n_components=2, n_neighbors=10, metric=Euclidean)");
    EXPECT_EQ(std::format("{}", dm::stats::JackknifeResult{}),
              "JackknifeResult(estimate=0, bias=0, bias_corrected=0, standard_error=0, leave_one_out=[], pseudo_values=[])");
    EXPECT_EQ(std::format("{}", dm::stats::DeleteDJackknifeOptions{}),
              "DeleteDJackknifeOptions(d=1, max_subsets=2000, seed=42)");
    EXPECT_EQ(std::format("{}", dm::stats::DeleteDJackknifeResult{}),
              "DeleteDJackknifeResult(estimate=0, standard_error=0, d=1, num_subsets=0, complete=false)");
    EXPECT_EQ(std::format("{}", dm::stats::KernelPCAOptions{}),
              "KernelPCAOptions(n_components=2, kernel=\"rbf\", gamma=1, degree=3, coef0=1)");
    EXPECT_EQ(std::format("{}", dm::stats::KernelRegressionOptions{}),
              "KernelRegressionOptions(kernel=Gaussian, bandwidth=-1, n_bandwidth=50, standardize=true, bandwidth_candidates=[])");
    EXPECT_EQ(std::format("{}", dm::stats::KernelRidgeRegressionOptions{}),
              "KernelRidgeRegressionOptions(kernel=\"rbf\", lambda=-1, gamma=-1, degree=3, coef0=1, n_lambda_grid=40, n_gamma_grid=12, standardize=true)");
    EXPECT_EQ(std::format("{}", dm::stats::KMeansOptions{}),
              "KMeansOptions(n_clusters=8, max_iterations=300, n_init=10, tolerance=1e-04, seed=42)");
    EXPECT_EQ(std::format("{}", dm::stats::KNNClassifierOptions{}),
              "KNNClassifierOptions(k=5, metric=Euclidean, weighted=false, standardize=true)");
    EXPECT_EQ(std::format("{}", dm::stats::KNNClassifierPrediction{}),
              "KNNClassifierPrediction(class_label=[], vote_share=[])");
    EXPECT_EQ(std::format("{}", dm::stats::KNNNeighbors{}), "KNNNeighbors(index=[], distance=[])");
    EXPECT_EQ(std::format("{}", dm::stats::KNNRegressorOptions{}),
              "KNNRegressorOptions(k=5, metric=Euclidean, weighted=false, standardize=true)");
    EXPECT_EQ(std::format("{}", dm::stats::LaplacianEigenmapsOptions{}),
              "LaplacianEigenmapsOptions(n_components=2, n_neighbors=10, heat_kernel_t=1)");
    EXPECT_EQ(std::format("{}", dm::stats::LDAOptions{}),
              "LDAOptions(priors=none)");
    EXPECT_EQ(std::format("{}", dm::stats::LDAPrediction{}),
              "LDAPrediction(class_label=[], posterior=[], discriminants=[])");
    EXPECT_EQ(std::format("{}", dm::stats::LLEOptions{}),
              "LLEOptions(n_components=2, n_neighbors=10, regularization=0.001, metric=Euclidean)");
    EXPECT_EQ(std::format("{}", dm::stats::LmOptions{}),
              "LmOptions(weights_column=none)");
    EXPECT_EQ(std::format("{}", dm::stats::LmPrediction{}),
              "LmPrediction(fit=[], se_fit=[], lower=[], upper=[])");
    EXPECT_EQ(std::format("{}", dm::stats::AnovaRow{}),
              "AnovaRow(term=\"\", degrees_of_freedom=0, sum_sq=0, mean_sq=0, f_value=0, p_value=0)");
    EXPECT_EQ(std::format("{}", dm::stats::NestedAnova{}),
              "NestedAnova(reduced_df=0, full_df=0, reduced_rss=0, full_rss=0, df=0, sum_sq=0, f_value=0, p_value=0)");
    EXPECT_EQ(std::format("{}", dm::stats::LM::Interval{}),
              "Interval(lower=0, upper=0)");
    EXPECT_EQ(std::format("{}", dm::stats::LMMOptions{}),
              "LMMOptions(reml=true, de_population_size=40, de_max_generations=300, theta_bound=5, seed=42)");
    EXPECT_EQ(std::format("{}", dm::stats::MDSOptions{}),
              "MDSOptions(n_components=2, metric=Euclidean)");
    EXPECT_EQ(std::format("{}", dm::stats::MLPClassifierOptions{}),
              "MLPClassifierOptions(hidden_layer_sizes=[16], activation=\"relu\", learning_rate=0.05, max_epochs=400, batch_size=32, l2=0, seed=42, standardize=true)");
    EXPECT_EQ(std::format("{}", dm::stats::MLPClassifierPrediction{}),
              "MLPClassifierPrediction(class_label=[], probabilities=[])");
    EXPECT_EQ(std::format("{}", dm::stats::NaiveBayesClassifierOptions{}),
              "NaiveBayesClassifierOptions(laplace_smoothing=1, var_smoothing=1e-09, priors=[])");
    EXPECT_EQ(std::format("{}", dm::stats::NaiveBayesClassifierPrediction{}),
              "NaiveBayesClassifierPrediction(class_label=[], probability=[])");
    EXPECT_EQ(std::format("{}", dm::stats::PCAOptions{}),
              "PCAOptions(center=true, scale=true)");
    EXPECT_EQ(std::format("{}", dm::stats::PermutationTestOptions{}),
              "PermutationTestOptions(num_permutations=5000, alternative=TwoSided, seed=42)");
    EXPECT_EQ(std::format("{}", dm::stats::PermutationTestResult{}),
              "PermutationTestResult(observed=0, p_value=0, null_mean=0, null_sd=0, num_permutations=0, null_distribution=[])");
    EXPECT_EQ(std::format("{}", dm::stats::QuantileRegressionOptions{}),
              "QuantileRegressionOptions(quantile=0.5, rho=1, max_iterations=10000, absolute_tolerance=1e-06, relative_tolerance=1e-05, adaptive_rho=false, weights_column=none, method=Simplex)");
    EXPECT_EQ(std::format("{}", dm::stats::RandomForestStrength{.strength = 0.5, .margin_variance = 0.1, .mean_tree_sd = 0.5, .correlation = 0.4, .error_bound = 1.2, .rows = 10}),
              "RandomForestStrength(strength=0.5, margin_variance=0.1, mean_tree_sd=0.5, correlation=0.4, error_bound=1.2, rows=10)");
    EXPECT_EQ(std::format("{}", dm::stats::RandomForestPermutationImportance{{"x"}, {0.25}, {0.5}}),
              "RandomForestPermutationImportance(predictor_names=[x], importance=[0.25], standard_error=[0.5])");
    EXPECT_EQ(std::format("{}", dm::stats::RandomForestClassifierOptions{}),
              "RandomForestClassifierOptions(n_trees=100, max_depth=10, min_samples_split=2, min_samples_leaf=1, max_features=0, criterion=Gini, bootstrap=true, sample_fraction=1, seed=42)");
    EXPECT_EQ(std::format("{}", dm::stats::RandomForestClassifierPrediction{}),
              "RandomForestClassifierPrediction(class_label=[], vote_share=[])");
    EXPECT_EQ(std::format("{}", dm::stats::RandomForestRegressorOptions{}),
              "RandomForestRegressorOptions(n_trees=100, max_depth=10, min_samples_split=2, min_samples_leaf=1, max_features=0, bootstrap=true, sample_fraction=1, seed=42)");
    EXPECT_EQ(std::format("{}", dm::stats::SammonMappingOptions{}),
              "SammonMappingOptions(n_components=2, learning_rate=0.3, max_iterations=500, tolerance=1e-09, metric=Euclidean, seed=42)");
    EXPECT_EQ(std::format("{}", dm::stats::SVMOptions{}),
              "SVMOptions(kernel=Radial, cost=1, gamma=-1, coef0=0, degree=3, tolerance=0.001, max_passes=50, scale=true)");
    EXPECT_EQ(std::format("{}", dm::stats::SVMPrediction{}),
              "SVMPrediction(class_label=[], votes=[])");
    EXPECT_EQ(std::format("{}", dm::stats::TSNEOptions{}),
              "TSNEOptions(n_components=2, perplexity=30, max_iterations=1000, learning_rate=200, early_exaggeration=12, early_exaggeration_iterations=250, initial_momentum=0.5, final_momentum=0.8, momentum_switch_iteration=250, metric=Euclidean, seed=42)");
    EXPECT_EQ(std::format("{}", dm::stats::UMAPOptions{}),
              "UMAPOptions(n_components=2, n_neighbors=15, min_dist=0.1, max_iterations=500, learning_rate=1, negative_sample_rate=5, metric=Euclidean, seed=42)");
    EXPECT_EQ(std::format("{}", dm::stats::UStatisticOptions{}),
              "UStatisticOptions(max_terms=0, seed=42)");
    EXPECT_EQ(std::format("{}", dm::stats::UStatisticResult{}),
              "UStatisticResult(estimate=0, zeta1=0, variance=0, standard_error=0, degree=0, num_terms=0, complete=false)");
    EXPECT_EQ(std::format("{}", dm::stats::VStatisticOptions{}),
              "VStatisticOptions(max_terms=0, seed=42)");
    EXPECT_EQ(std::format("{}", dm::stats::VStatisticResult{}),
              "VStatisticResult(estimate=0, degree=0, num_terms=0, exact=false)");
    EXPECT_EQ(std::format("{}", dm::stats::XGBoostClassifierOptions{}),
              "XGBoostClassifierOptions(n_trees=100, learning_rate=0.3, max_depth=6, lambda=1, alpha=0, gamma=0, min_child_weight=1, min_samples_leaf=1, subsample=1, colsample_bytree=1, seed=42)");
    EXPECT_EQ(std::format("{}", dm::stats::XGBoostClassifierPrediction{}),
              "XGBoostClassifierPrediction(class_label=[], probability=[])");
    EXPECT_EQ(std::format("{}", dm::stats::XGBoostRegressorOptions{}),
              "XGBoostRegressorOptions(n_trees=100, learning_rate=0.3, max_depth=6, lambda=1, alpha=0, gamma=0, min_child_weight=1, min_samples_leaf=1, subsample=1, colsample_bytree=1, seed=42)");
}

namespace {

namespace st = dm::stats;

// A tiny deterministic frame: numeric x1..x3/y, Poisson counts, a 0/1 response, three classes, five groups.
dm::dstruct::DataFrame tiny_frame() {
    std::vector<double> x1, x2, x3, y, cnt, bin;
    std::vector<std::string> cls, group;
    for (int i = 0; i < 30; ++i) {
        const int c = i % 3;
        x1.push_back(std::sin(i * 1.3) + 2.0 * c);
        x2.push_back(std::cos(i * 0.7) * 1.5 - c);
        x3.push_back((i % 7) * 0.3 + std::sin(i));
        y.push_back(1.0 + 2.0 * x1.back() - x2.back() + 0.3 * std::sin(i * 2.1));
        cnt.push_back(static_cast<double>((i * 7) % 5));
        bin.push_back(((i * 5) % 7) < 3 ? 1.0 : 0.0);
        cls.push_back(std::string(1, static_cast<char>('a' + c)));
        group.push_back("g" + std::to_string(i % 5));
    }
    dm::dstruct::DataFrame df;
    df.add_column("x1", x1);
    df.add_column("x2", x2);
    df.add_column("x3", x3);
    df.add_column("y", y);
    df.add_column("cnt", cnt);
    df.add_column("bin", bin);
    df.add_column("cls", cls);
    df.add_column("group", group);
    return df;
}

const std::vector<std::string> kFeatures{"x1", "x2", "x3"};

template <typename Model>
void expect_summary(const Model& model) {
    const std::string text = std::format("{}", model);
    EXPECT_FALSE(text.empty());
    EXPECT_EQ(text, model.summary());
}

}  // namespace

TEST(StatsFormat, LmPrintsItsSummary) {
    const auto iris = dm::datasets::iris();
    const dm::stats::LM model(iris, "Sepal.Length ~ Petal.Length");
    const std::string text = std::format("{}", model);
    EXPECT_EQ(text, model.summary());
    EXPECT_NE(text.find("Petal.Length"), std::string::npos);
    const auto ci = model.confidence_intervals();
    EXPECT_TRUE(std::format("{}", ci.front()).starts_with("Interval(lower="));
    EXPECT_TRUE(std::format("{}", model.anova().front()).starts_with("AnovaRow(term=\"Petal.Length\""));
}

TEST(StatsFormat, RegressionModelsPrintTheirSummary) {
    const auto df = tiny_frame();
    st::GLMOptions poisson;
    poisson.family = st::GLMFamily::Poisson;
    expect_summary(st::GLM(df, "cnt ~ x1", poisson));
    expect_summary(st::QuantileRegression(df, "y ~ x1 + x2"));
    expect_summary(st::ElasticNet(df, "y ~ x1 + x2 + x3", {.n_lambda = 5, .cv_folds = 3}));
    expect_summary(st::Ridge(df, "y ~ x1 + x2 + x3", {.n_lambda = 5, .cv_folds = 3}));
    expect_summary(st::Lasso(df, "y ~ x1 + x2 + x3", {.n_lambda = 5, .cv_folds = 3}));
    expect_summary(st::KNNRegressor(df, "y ~ x1 + x2"));
    expect_summary(st::KernelRegression(df, "y ~ x1", {.n_bandwidth = 5}));
    expect_summary(st::KernelRidgeRegression(df, "y ~ x1 + x2", {.n_lambda_grid = 3, .n_gamma_grid = 2}));
    expect_summary(st::GaussianProcessRegression(df, "y ~ x1", {.n_length_scale_grid = 3, .n_noise_grid = 3}));
    expect_summary(st::DecisionTreeRegressor(df, "y ~ x1 + x2"));
    expect_summary(st::RandomForestRegressor(df, "y ~ x1 + x2", {.n_trees = 5}));
    expect_summary(st::GBMRegressor(df, "y ~ x1 + x2", {.n_trees = 5}));
    expect_summary(st::XGBoostRegressor(df, "y ~ x1 + x2", {.n_trees = 5}));
}

TEST(StatsFormat, MixedModelsPrintTheirSummary) {
    const auto df = tiny_frame();
    expect_summary(st::LMM(df, "y ~ x1 + (1 | group)", {.de_population_size = 8, .de_max_generations = 10}));
    expect_summary(st::GLMM(df, "bin ~ x1 + (1 | group)",
                            {.max_iterations = 5, .de_population_size = 8, .de_max_generations = 10}));
    st::INLAMixedModelOptions inla;
    inla.grid_points_per_dim = 3;
    inla.mode_population_size = 8;
    inla.mode_max_generations = 10;
    expect_summary(st::INLAMixedModel(df, "y ~ x1 + (1 | group)", inla));
}

TEST(StatsFormat, ClassifiersPrintTheirSummary) {
    const auto df = tiny_frame();
    const std::string f = "cls ~ x1 + x2";
    expect_summary(st::LDA(df, f));
    expect_summary(st::NaiveBayesClassifier(df, f));
    expect_summary(st::KNNClassifier(df, f));
    expect_summary(st::SVM(df, f, {.max_passes = 5}));
    expect_summary(st::DecisionTreeClassifier(df, f));
    expect_summary(st::RandomForestClassifier(df, f, {.n_trees = 5}));
    expect_summary(st::GBMClassifier(df, f, {.n_trees = 5}));
    expect_summary(st::XGBoostClassifier(df, f, {.n_trees = 5}));
    expect_summary(st::MLPClassifier(df, f, {.hidden_layer_sizes = {4}, .max_epochs = 5}));

    std::vector<dm::linalg::DenseMatrix<double>> images;
    std::vector<std::string> labels;
    for (int i = 0; i < 8; ++i) {
        dm::linalg::DenseMatrix<double> image(6, 6, 0.0);
        for (std::size_t r = 0; r < 6; ++r) image(r, static_cast<std::size_t>(i % 2 == 0 ? 1 : 4)) = 1.0;
        images.push_back(image);
        labels.push_back(i % 2 == 0 ? "left" : "right");
    }
    expect_summary(st::CNNClassifier(images, labels,
                                     {.n_filters = 2, .dense_hidden = 4, .max_epochs = 2, .batch_size = 4}));
}

TEST(StatsFormat, UnsupervisedModelsPrintTheirSummary) {
    const auto df = tiny_frame();
    expect_summary(st::PCA(df, kFeatures));
    expect_summary(st::KernelPCA(df, kFeatures));
    expect_summary(st::FactorAnalysis(df, kFeatures, {.n_factors = 1}));
    expect_summary(st::ICA(df, kFeatures));
    expect_summary(st::MDS(df, kFeatures));
    expect_summary(st::Isomap(df, kFeatures, {.n_neighbors = 6}));
    expect_summary(st::LLE(df, kFeatures, {.n_neighbors = 6}));
    expect_summary(st::LaplacianEigenmaps(df, kFeatures, {.n_neighbors = 6}));
    expect_summary(st::DiffusionMaps(df, kFeatures));
    expect_summary(st::SammonMapping(df, kFeatures, {.max_iterations = 20}));
    expect_summary(st::TSNE(df, kFeatures, {.perplexity = 5.0, .max_iterations = 30}));
    expect_summary(st::UMAP(df, kFeatures, {.n_neighbors = 5, .max_iterations = 20}));
    expect_summary(st::KMeans(df, kFeatures, {.n_clusters = 3}));
    expect_summary(st::DBSCAN(df, kFeatures, {.eps = 1.0, .min_samples = 3}));
    const st::AgglomerativeClustering agglo(df, kFeatures, {.n_clusters = 3});
    expect_summary(agglo);
    EXPECT_TRUE(std::format("{}", agglo.merge_history().front()).starts_with("Merge(cluster_a="));
}

TEST(StatsFormat, HypothesisTestResultPrintsOneLine) {
    st::HypothesisTestResult r;
    r.method = "Welch Two Sample t-test";
    r.statistic = 2.5;
    r.parameter1 = 7.25;
    r.p_value = 0.04;
    r.estimate1 = 1;
    r.estimate2 = 0.5;
    EXPECT_EQ(std::format("{}", r),
              "HypothesisTestResult(method=\"Welch Two Sample t-test\", statistic=2.5, parameter1=7.25, parameter2=0, "
              "p_value=0.04, estimate1=1, estimate2=0.5, conf_int=none, alternative=TwoSided)");
    r.has_conf_int = true;
    r.conf_int_lower = -0.25;
    r.conf_int_upper = 1.5;
    r.alternative = st::Alternative::Greater;
    EXPECT_NE(std::format("{}", r).find("conf_int=[-0.25, 1.5], alternative=Greater)"), std::string::npos);

    const auto t = st::t_test_one_sample({4.9, 5.1, 5.3, 4.7, 5.0}, 5.0);
    EXPECT_NE(std::format("{}", t).find(t.method), std::string::npos);
}

TEST(StatsFormat, ResamplingTypesPrint) {
    const auto mean = [](const std::vector<double>& v) {
        double s = 0;
        for (double x : v) s += x;
        return s / static_cast<double>(v.size());
    };
    const std::vector<double> sample{1, 2, 3, 4, 5, 6, 7, 8};
    const st::Bootstrap boot({.num_resamples = 20});
    EXPECT_EQ(std::format("{}", boot), "Bootstrap(num_resamples=20, confidence_level=0.95, seed=42)");
    EXPECT_TRUE(std::format("{}", boot.run(sample, mean)).starts_with("BootstrapResult(estimate=4.5, "));
    EXPECT_TRUE(std::format("{}", st::jackknife(sample, mean)).starts_with("JackknifeResult(estimate=4.5, "));
    const st::UStatistic u(1);
    const st::VStatistic v(1);
    EXPECT_EQ(std::format("{}", u), "UStatistic(degree=1, max_terms=0, seed=42)");
    EXPECT_EQ(std::format("{}", v), "VStatistic(degree=1, max_terms=0, seed=42)");
    EXPECT_EQ(std::format("{}", st::UStatistic(2, {.max_terms = 50, .seed = 7})),
              "UStatistic(degree=2, max_terms=50, seed=7)");
    EXPECT_TRUE(std::format("{}", u.run(sample, mean)).starts_with("UStatisticResult(estimate=4.5, "));
    EXPECT_EQ(std::format("{}", v.run(sample, mean)), "VStatisticResult(estimate=4.5, degree=1, num_terms=8, exact=true)");
}

TEST(StatsFormat, FormulaTypesPrint) {
    const auto df = tiny_frame();
    const st::Formula formula("y ~ x1 + x2");
    EXPECT_EQ(std::format("{}", formula), "y ~ x1 + x2");

    const auto info = formula.resolve(df);
    const std::string info_text = std::format("{}", info);
    EXPECT_TRUE(info_text.starts_with("DesignInfo(response_name=\"y\", has_intercept=true, columns=2"));
    EXPECT_NE(info_text.find("x2]"), std::string::npos);
    EXPECT_TRUE(std::format("{}", info.build_matrix(df, true)).starts_with("DesignMatrixData(X=30x3, y=["));
    EXPECT_TRUE(std::format("{}", info.columns.front()).starts_with("ResolvedColumn(kind=Numeric, name=\"x1\""));

    const auto cinfo = st::Formula("cls ~ x1").resolve(df, st::ResponseKind::Categorical);
    EXPECT_TRUE(std::format("{}", cinfo.build_classification_matrix(df))
                    .starts_with("ClassificationDesignData(X=30x1, labels=[a, b, c, a, b, ..., c] (n=30)"))
        << std::format("{}", cinfo.build_classification_matrix(df));

    st::ResolvedColumn dummy;
    dummy.kind = st::ResolvedColumnKind::CategoricalDummy;
    dummy.name = "suppVC";
    dummy.source_column = "supp";
    dummy.level = "VC";
    EXPECT_EQ(std::format("{}", dummy),
              "ResolvedColumn(kind=CategoricalDummy, name=\"suppVC\", source_column=\"supp\", level=\"VC\")");

    auto column = [](std::string name) {
        auto n = std::make_shared<st::ExprNode>();
        n->op = st::ExprOp::Column;
        n->column = std::move(name);
        return n;
    };
    auto two = std::make_shared<st::ExprNode>();
    two->op = st::ExprOp::Literal;
    two->literal = 2;
    auto sum = std::make_shared<st::ExprNode>();
    sum->op = st::ExprOp::Add;
    sum->left = column("y");
    sum->right = two;
    st::ExprNode product;
    product.op = st::ExprOp::Multiply;
    product.left = column("x");
    product.right = sum;
    EXPECT_EQ(std::format("{}", product), "ExprNode(x * (y + 2))");
}

TEST(StatsFormat, TimeSeriesModelsPrint) {
    std::vector<double> y;
    for (int i = 0; i < 30; ++i) y.push_back(10 + 0.5 * i + std::sin(i * 1.7));
    const st::ARIMA arima(y, {.p = 1, .de_population_size = 8, .de_max_generations = 10});
    const std::string a = std::format("{}", arima);
    EXPECT_TRUE(a.starts_with("ARIMA(1,0,0) with mean, n=30\nar=[")) << a;
    EXPECT_NE(a.find("\nsigma2="), std::string::npos);

    const st::ExponentialSmoothing ets(
        y, {.trend = st::TrendType::Additive, .de_population_size = 8, .de_max_generations = 10});
    const std::string e = std::format("{}", ets);
    EXPECT_TRUE(e.starts_with("ExponentialSmoothing(trend=Additive, seasonal=None, seasonal_period=0), n=30\nalpha="))
        << e;
}

TEST(StatsFormat, ConformalAndExplainersPrint) {
    const dm::linalg::DenseMatrix<double> probs(4, 2, {0.9, 0.1, 0.2, 0.8, 0.6, 0.4, 0.3, 0.7});
    const st::SplitConformalClassifier cls(probs, {0, 1, 0, 1}, {"no", "yes"});
    EXPECT_TRUE(std::format("{}", cls).starts_with("SplitConformalClassifier(alpha=0.1, score=\"lac\", quantile="));
    EXPECT_TRUE(std::format("{}", cls).ends_with("n_calibration=4, classes=[no, yes])"));
    EXPECT_TRUE(std::format("{}", cls.evaluate(probs, {0, 1, 0, 1})).starts_with("Evaluation(coverage="));

    EXPECT_EQ(std::format("{}", st::conformal_coverage(19, 0.1)).substr(0, 47), "ConformalCoverage(rank=18, mean=0.9, beta_a=18,");
    const st::SplitConformalRegressor reg({1, 2, 3, 4}, {1.1, 1.8, 3.3, 4.0});
    EXPECT_TRUE(std::format("{}", reg).ends_with("n_calibration=4)"));
    EXPECT_TRUE(std::format("{}", reg.evaluate({1, 2}, {1, 2})).starts_with("Evaluation(coverage="));

    const dm::linalg::DenseMatrix<double> image(8, 8, 1.0);
    const auto predict = [](const dm::linalg::DenseMatrix<double>& m) { return m(0, 0) + m(7, 7); };
    const st::LimeImageExplainer lime(image, predict, {.n_samples = 20});
    EXPECT_TRUE(std::format("{}", lime).starts_with("LimeImageExplainer(patches=4x4, intercept=")) << std::format("{}", lime);
    const st::ShapImageExplainer shap(image, predict);
    EXPECT_TRUE(std::format("{}", shap).starts_with("ShapImageExplainer(patches=4x4, base_value=")) << std::format("{}", shap);
}
