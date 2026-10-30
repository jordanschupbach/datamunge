// std::format / std::print support for datamunge::optim types.
#include <dm/dm.hpp>

#include <gtest/gtest.h>

#include <format>
#include <string>
#include <string_view>

namespace op = dm::optim;

static_assert(std::formattable<op::ACOROptions, char>);
static_assert(std::formattable<op::AdaDeltaOptions, char>);
static_assert(std::formattable<op::AdaGradOptions, char>);
static_assert(std::formattable<op::AdamOptions, char>);
static_assert(std::formattable<op::AMSGradOptions, char>);
static_assert(std::formattable<op::ArtificialBeeColonyOptions, char>);
static_assert(std::formattable<op::AugmentedLagrangianOptions, char>);
static_assert(std::formattable<op::BayesianOptimizationOptions, char>);
static_assert(std::formattable<op::BayesianOptimizationResult, char>);
static_assert(std::formattable<op::SurrogatePrediction, char>);
static_assert(std::formattable<op::GaussianProcessSurrogateOptions, char>);
static_assert(std::formattable<op::Acquisition, char>);
static_assert(std::formattable<op::CMAESOptions, char>);
static_assert(std::formattable<op::ConjugateGradientOptions, char>);
static_assert(std::formattable<op::CoordinateDescentOptions, char>);
static_assert(std::formattable<op::CrossEntropyMethodOptions, char>);
static_assert(std::formattable<op::CuckooSearchOptions, char>);
static_assert(std::formattable<op::DEOptions, char>);
static_assert(std::formattable<op::EstimationOfDistributionOptions, char>);
static_assert(std::formattable<op::EvolutionStrategyOptions, char>);
static_assert(std::formattable<op::FireflyAlgorithmOptions, char>);
static_assert(std::formattable<op::FISTAOptions, char>);
static_assert(std::formattable<op::GAOptions, char>);
static_assert(std::formattable<op::GradientDescentOptions, char>);
static_assert(std::formattable<op::GreyWolfOptimizerOptions, char>);
static_assert(std::formattable<op::HarmonySearchOptions, char>);
static_assert(std::formattable<op::InteriorPointOptions, char>);
static_assert(std::formattable<op::LBFGSOptions, char>);
static_assert(std::formattable<op::LevenbergMarquardtOptions, char>);
static_assert(std::formattable<op::MOEADOptions, char>);
static_assert(std::formattable<op::ParetoPoint, char>);
static_assert(std::formattable<op::NadamOptions, char>);
static_assert(std::formattable<op::NelderMeadOptions, char>);
static_assert(std::formattable<op::NesterovAcceleratedGradientOptions, char>);
static_assert(std::formattable<op::NewtonOptions, char>);
static_assert(std::formattable<op::NSGA2Options, char>);
static_assert(std::formattable<op::ParallelTemperingOptions, char>);
static_assert(std::formattable<op::ProximalGradientOptions, char>);
static_assert(std::formattable<op::PSOOptions, char>);
static_assert(std::formattable<op::RandomizedBlockCoordinateDescentOptions, char>);
static_assert(std::formattable<op::RMSPropOptions, char>);
static_assert(std::formattable<op::SAGAOptions, char>);
static_assert(std::formattable<op::SGDOptions, char>);
static_assert(std::formattable<op::SimulatedAnnealingOptions, char>);
static_assert(std::formattable<op::SPEA2Options, char>);
static_assert(std::formattable<op::SQPOptions, char>);
static_assert(std::formattable<op::SVRGOptions, char>);
static_assert(std::formattable<op::TrustRegionNewtonOptions, char>);
static_assert(std::formattable<op::WhaleOptimizationOptions, char>);
static_assert(std::formattable<op::ACOR, char>);
static_assert(std::formattable<op::AdaDelta, char>);
static_assert(std::formattable<op::AdaGrad, char>);
static_assert(std::formattable<op::Adam, char>);
static_assert(std::formattable<op::AMSGrad, char>);
static_assert(std::formattable<op::ArtificialBeeColony, char>);
static_assert(std::formattable<op::AugmentedLagrangian, char>);
static_assert(std::formattable<op::BayesianOptimization, char>);
static_assert(std::formattable<op::CMAES, char>);
static_assert(std::formattable<op::ConjugateGradient, char>);
static_assert(std::formattable<op::CoordinateDescent, char>);
static_assert(std::formattable<op::CrossEntropyMethod, char>);
static_assert(std::formattable<op::CuckooSearch, char>);
static_assert(std::formattable<op::DifferentialEvolution, char>);
static_assert(std::formattable<op::EstimationOfDistribution, char>);
static_assert(std::formattable<op::EvolutionStrategy, char>);
static_assert(std::formattable<op::FireflyAlgorithm, char>);
static_assert(std::formattable<op::FISTA, char>);
static_assert(std::formattable<op::GeneticAlgorithm, char>);
static_assert(std::formattable<op::GradientDescent, char>);
static_assert(std::formattable<op::GreyWolfOptimizer, char>);
static_assert(std::formattable<op::HarmonySearch, char>);
static_assert(std::formattable<op::InteriorPoint, char>);
static_assert(std::formattable<op::LBFGS, char>);
static_assert(std::formattable<op::LevenbergMarquardt, char>);
static_assert(std::formattable<op::MOEAD, char>);
static_assert(std::formattable<op::Nadam, char>);
static_assert(std::formattable<op::NelderMead, char>);
static_assert(std::formattable<op::NesterovAcceleratedGradient, char>);
static_assert(std::formattable<op::Newton, char>);
static_assert(std::formattable<op::NSGA2, char>);
static_assert(std::formattable<op::ParallelTempering, char>);
static_assert(std::formattable<op::ProximalGradient, char>);
static_assert(std::formattable<op::PSO, char>);
static_assert(std::formattable<op::RandomizedBlockCoordinateDescent, char>);
static_assert(std::formattable<op::RBFGaussianProcessSurrogate, char>);
static_assert(std::formattable<op::RMSProp, char>);
static_assert(std::formattable<op::SAGA, char>);
static_assert(std::formattable<op::SGD, char>);
static_assert(std::formattable<op::SimulatedAnnealing, char>);
static_assert(std::formattable<op::SPEA2, char>);
static_assert(std::formattable<op::SQP, char>);
static_assert(std::formattable<op::SVRG, char>);
static_assert(std::formattable<op::TrustRegionNewton, char>);
static_assert(std::formattable<op::WhaleOptimization, char>);

TEST(OptimFormat, OptionsExactText) {
    EXPECT_EQ(std::format("{}", op::LBFGSOptions{}),
              "LBFGSOptions(max_iterations=500, tolerance=1e-08, history_size=10, armijo_c1=1e-04, wolfe_c2=0.9, "
              "max_line_search_trials=50)");
    EXPECT_EQ(std::format("{}", op::NewtonOptions{.max_iterations = 7, .tolerance = 0.5, .damping = 0.25}),
              "NewtonOptions(max_iterations=7, tolerance=0.5, damping=0.25)");
}

TEST(OptimFormat, ParetoPoint) {
    EXPECT_EQ(std::format("{}", op::ParetoPoint{{1, 2}, {0.5, 3}}), "ParetoPoint(coordinates=[1, 2], objectives=[0.5, 3])");
}

TEST(OptimFormat, EveryOptionsStructPrintsItsNameAndFields) {
    EXPECT_EQ(std::format("{}", op::ACOROptions{}).rfind("ACOROptions(", 0), 0u);
    EXPECT_EQ(std::format("{}", op::AdaDeltaOptions{}).rfind("AdaDeltaOptions(", 0), 0u);
    EXPECT_EQ(std::format("{}", op::AdaGradOptions{}).rfind("AdaGradOptions(", 0), 0u);
    EXPECT_EQ(std::format("{}", op::AdamOptions{}).rfind("AdamOptions(", 0), 0u);
    EXPECT_EQ(std::format("{}", op::AMSGradOptions{}).rfind("AMSGradOptions(", 0), 0u);
    EXPECT_EQ(std::format("{}", op::ArtificialBeeColonyOptions{}).rfind("ArtificialBeeColonyOptions(", 0), 0u);
    EXPECT_EQ(std::format("{}", op::AugmentedLagrangianOptions{}).rfind("AugmentedLagrangianOptions(", 0), 0u);
    EXPECT_EQ(std::format("{}", op::BayesianOptimizationOptions{}).rfind("BayesianOptimizationOptions(", 0), 0u);
    EXPECT_EQ(std::format("{}", op::CMAESOptions{}).rfind("CMAESOptions(", 0), 0u);
    EXPECT_EQ(std::format("{}", op::ConjugateGradientOptions{}).rfind("ConjugateGradientOptions(", 0), 0u);
    EXPECT_EQ(std::format("{}", op::CoordinateDescentOptions{}).rfind("CoordinateDescentOptions(", 0), 0u);
    EXPECT_EQ(std::format("{}", op::CrossEntropyMethodOptions{}).rfind("CrossEntropyMethodOptions(", 0), 0u);
    EXPECT_EQ(std::format("{}", op::CuckooSearchOptions{}).rfind("CuckooSearchOptions(", 0), 0u);
    EXPECT_EQ(std::format("{}", op::DEOptions{}).rfind("DEOptions(", 0), 0u);
    EXPECT_EQ(std::format("{}", op::EstimationOfDistributionOptions{}).rfind("EstimationOfDistributionOptions(", 0), 0u);
    EXPECT_EQ(std::format("{}", op::EvolutionStrategyOptions{}).rfind("EvolutionStrategyOptions(", 0), 0u);
    EXPECT_EQ(std::format("{}", op::FireflyAlgorithmOptions{}).rfind("FireflyAlgorithmOptions(", 0), 0u);
    EXPECT_EQ(std::format("{}", op::FISTAOptions{}).rfind("FISTAOptions(", 0), 0u);
    EXPECT_EQ(std::format("{}", op::GAOptions{}).rfind("GAOptions(", 0), 0u);
    EXPECT_EQ(std::format("{}", op::GradientDescentOptions{}).rfind("GradientDescentOptions(", 0), 0u);
    EXPECT_EQ(std::format("{}", op::GreyWolfOptimizerOptions{}).rfind("GreyWolfOptimizerOptions(", 0), 0u);
    EXPECT_EQ(std::format("{}", op::HarmonySearchOptions{}).rfind("HarmonySearchOptions(", 0), 0u);
    EXPECT_EQ(std::format("{}", op::InteriorPointOptions{}).rfind("InteriorPointOptions(", 0), 0u);
    EXPECT_EQ(std::format("{}", op::LBFGSOptions{}).rfind("LBFGSOptions(", 0), 0u);
    EXPECT_EQ(std::format("{}", op::LevenbergMarquardtOptions{}).rfind("LevenbergMarquardtOptions(", 0), 0u);
    EXPECT_EQ(std::format("{}", op::MOEADOptions{}).rfind("MOEADOptions(", 0), 0u);
    EXPECT_EQ(std::format("{}", op::NadamOptions{}).rfind("NadamOptions(", 0), 0u);
    EXPECT_EQ(std::format("{}", op::NelderMeadOptions{}).rfind("NelderMeadOptions(", 0), 0u);
    EXPECT_EQ(std::format("{}", op::NesterovAcceleratedGradientOptions{}).rfind("NesterovAcceleratedGradientOptions(", 0), 0u);
    EXPECT_EQ(std::format("{}", op::NewtonOptions{}).rfind("NewtonOptions(", 0), 0u);
    EXPECT_EQ(std::format("{}", op::NSGA2Options{}).rfind("NSGA2Options(", 0), 0u);
    EXPECT_EQ(std::format("{}", op::ParallelTemperingOptions{}).rfind("ParallelTemperingOptions(", 0), 0u);
    EXPECT_EQ(std::format("{}", op::ProximalGradientOptions{}).rfind("ProximalGradientOptions(", 0), 0u);
    EXPECT_EQ(std::format("{}", op::PSOOptions{}).rfind("PSOOptions(", 0), 0u);
    EXPECT_EQ(std::format("{}", op::RandomizedBlockCoordinateDescentOptions{}).rfind("RandomizedBlockCoordinateDescentOptions(", 0), 0u);
    EXPECT_EQ(std::format("{}", op::RMSPropOptions{}).rfind("RMSPropOptions(", 0), 0u);
    EXPECT_EQ(std::format("{}", op::SAGAOptions{}).rfind("SAGAOptions(", 0), 0u);
    EXPECT_EQ(std::format("{}", op::SGDOptions{}).rfind("SGDOptions(", 0), 0u);
    EXPECT_EQ(std::format("{}", op::SimulatedAnnealingOptions{}).rfind("SimulatedAnnealingOptions(", 0), 0u);
    EXPECT_EQ(std::format("{}", op::SPEA2Options{}).rfind("SPEA2Options(", 0), 0u);
    EXPECT_EQ(std::format("{}", op::SQPOptions{}).rfind("SQPOptions(", 0), 0u);
    EXPECT_EQ(std::format("{}", op::SVRGOptions{}).rfind("SVRGOptions(", 0), 0u);
    EXPECT_EQ(std::format("{}", op::TrustRegionNewtonOptions{}).rfind("TrustRegionNewtonOptions(", 0), 0u);
    EXPECT_EQ(std::format("{}", op::WhaleOptimizationOptions{}).rfind("WhaleOptimizationOptions(", 0), 0u);
}

/// @brief Expected optimizer text: its options' text with the options type name replaced.
template <typename Optimizer>
static std::string expected_optimizer_text(std::string_view name, const Optimizer& optimizer) {
    std::string options = std::format("{}", optimizer.options());
    return std::string(name) + options.substr(options.find('('));
}

TEST(OptimFormat, OptimizersPrintTheirOptions) {
    EXPECT_EQ(std::format("{}", op::LBFGS{}),
              "LBFGS(max_iterations=500, tolerance=1e-08, history_size=10, armijo_c1=1e-04, wolfe_c2=0.9, "
              "max_line_search_trials=50)");
    EXPECT_EQ(std::format("{}", op::Newton{op::NewtonOptions{.max_iterations = 7, .tolerance = 0.5, .damping = 0.25}}),
              "Newton(max_iterations=7, tolerance=0.5, damping=0.25)");
    EXPECT_EQ(std::format("{}", op::ACOR{}), expected_optimizer_text("ACOR", op::ACOR{}));
    EXPECT_EQ(std::format("{}", op::AdaDelta{}), expected_optimizer_text("AdaDelta", op::AdaDelta{}));
    EXPECT_EQ(std::format("{}", op::AdaGrad{}), expected_optimizer_text("AdaGrad", op::AdaGrad{}));
    EXPECT_EQ(std::format("{}", op::Adam{}), expected_optimizer_text("Adam", op::Adam{}));
    EXPECT_EQ(std::format("{}", op::AMSGrad{}), expected_optimizer_text("AMSGrad", op::AMSGrad{}));
    EXPECT_EQ(std::format("{}", op::ArtificialBeeColony{}), expected_optimizer_text("ArtificialBeeColony", op::ArtificialBeeColony{}));
    EXPECT_EQ(std::format("{}", op::AugmentedLagrangian{}), expected_optimizer_text("AugmentedLagrangian", op::AugmentedLagrangian{}));
    EXPECT_EQ(std::format("{}", op::BayesianOptimization{}), expected_optimizer_text("BayesianOptimization", op::BayesianOptimization{}));
    EXPECT_EQ(std::format("{}", op::CMAES{}), expected_optimizer_text("CMAES", op::CMAES{}));
    EXPECT_EQ(std::format("{}", op::ConjugateGradient{}), expected_optimizer_text("ConjugateGradient", op::ConjugateGradient{}));
    EXPECT_EQ(std::format("{}", op::CoordinateDescent{}), expected_optimizer_text("CoordinateDescent", op::CoordinateDescent{}));
    EXPECT_EQ(std::format("{}", op::CrossEntropyMethod{}), expected_optimizer_text("CrossEntropyMethod", op::CrossEntropyMethod{}));
    EXPECT_EQ(std::format("{}", op::CuckooSearch{}), expected_optimizer_text("CuckooSearch", op::CuckooSearch{}));
    EXPECT_EQ(std::format("{}", op::DifferentialEvolution{}), expected_optimizer_text("DifferentialEvolution", op::DifferentialEvolution{}));
    EXPECT_EQ(std::format("{}", op::EstimationOfDistribution{}), expected_optimizer_text("EstimationOfDistribution", op::EstimationOfDistribution{}));
    EXPECT_EQ(std::format("{}", op::EvolutionStrategy{}), expected_optimizer_text("EvolutionStrategy", op::EvolutionStrategy{}));
    EXPECT_EQ(std::format("{}", op::FireflyAlgorithm{}), expected_optimizer_text("FireflyAlgorithm", op::FireflyAlgorithm{}));
    EXPECT_EQ(std::format("{}", op::FISTA{}), expected_optimizer_text("FISTA", op::FISTA{}));
    EXPECT_EQ(std::format("{}", op::GeneticAlgorithm{}), expected_optimizer_text("GeneticAlgorithm", op::GeneticAlgorithm{}));
    EXPECT_EQ(std::format("{}", op::GradientDescent{}), expected_optimizer_text("GradientDescent", op::GradientDescent{}));
    EXPECT_EQ(std::format("{}", op::GreyWolfOptimizer{}), expected_optimizer_text("GreyWolfOptimizer", op::GreyWolfOptimizer{}));
    EXPECT_EQ(std::format("{}", op::HarmonySearch{}), expected_optimizer_text("HarmonySearch", op::HarmonySearch{}));
    EXPECT_EQ(std::format("{}", op::InteriorPoint{}), expected_optimizer_text("InteriorPoint", op::InteriorPoint{}));
    EXPECT_EQ(std::format("{}", op::LBFGS{}), expected_optimizer_text("LBFGS", op::LBFGS{}));
    EXPECT_EQ(std::format("{}", op::LevenbergMarquardt{}), expected_optimizer_text("LevenbergMarquardt", op::LevenbergMarquardt{}));
    EXPECT_EQ(std::format("{}", op::MOEAD{}), expected_optimizer_text("MOEAD", op::MOEAD{}));
    EXPECT_EQ(std::format("{}", op::Nadam{}), expected_optimizer_text("Nadam", op::Nadam{}));
    EXPECT_EQ(std::format("{}", op::NelderMead{}), expected_optimizer_text("NelderMead", op::NelderMead{}));
    EXPECT_EQ(std::format("{}", op::NesterovAcceleratedGradient{}), expected_optimizer_text("NesterovAcceleratedGradient", op::NesterovAcceleratedGradient{}));
    EXPECT_EQ(std::format("{}", op::Newton{}), expected_optimizer_text("Newton", op::Newton{}));
    EXPECT_EQ(std::format("{}", op::NSGA2{}), expected_optimizer_text("NSGA2", op::NSGA2{}));
    EXPECT_EQ(std::format("{}", op::ParallelTempering{}), expected_optimizer_text("ParallelTempering", op::ParallelTempering{}));
    EXPECT_EQ(std::format("{}", op::ProximalGradient{}), expected_optimizer_text("ProximalGradient", op::ProximalGradient{}));
    EXPECT_EQ(std::format("{}", op::PSO{}), expected_optimizer_text("PSO", op::PSO{}));
    EXPECT_EQ(std::format("{}", op::RandomizedBlockCoordinateDescent{}), expected_optimizer_text("RandomizedBlockCoordinateDescent", op::RandomizedBlockCoordinateDescent{}));
    EXPECT_EQ(std::format("{}", op::RMSProp{}), expected_optimizer_text("RMSProp", op::RMSProp{}));
    EXPECT_EQ(std::format("{}", op::SAGA{}), expected_optimizer_text("SAGA", op::SAGA{}));
    EXPECT_EQ(std::format("{}", op::SGD{}), expected_optimizer_text("SGD", op::SGD{}));
    EXPECT_EQ(std::format("{}", op::SimulatedAnnealing{}), expected_optimizer_text("SimulatedAnnealing", op::SimulatedAnnealing{}));
    EXPECT_EQ(std::format("{}", op::SPEA2{}), expected_optimizer_text("SPEA2", op::SPEA2{}));
    EXPECT_EQ(std::format("{}", op::SQP{}), expected_optimizer_text("SQP", op::SQP{}));
    EXPECT_EQ(std::format("{}", op::SVRG{}), expected_optimizer_text("SVRG", op::SVRG{}));
    EXPECT_EQ(std::format("{}", op::TrustRegionNewton{}), expected_optimizer_text("TrustRegionNewton", op::TrustRegionNewton{}));
    EXPECT_EQ(std::format("{}", op::WhaleOptimization{}), expected_optimizer_text("WhaleOptimization", op::WhaleOptimization{}));
}

TEST(OptimFormat, BayesianOptimizationTypes) {
    EXPECT_EQ(std::format("{}", op::BayesianOptimizationOptions{}),
              "BayesianOptimizationOptions(initial_samples=8, max_iterations=50, seed=42, candidates=128, refine_iterations=20)");
    EXPECT_EQ(std::format("{}", op::SurrogatePrediction{1.5, 0.25}), "SurrogatePrediction(mean=1.5, sd=0.25)");
    EXPECT_EQ(std::format("{}", op::Acquisition::LowerConfidenceBound), "LowerConfidenceBound");
    EXPECT_EQ(std::format("{}", op::GaussianProcessSurrogateOptions{}),
              "GaussianProcessSurrogateOptions(length_scale=1, signal_variance=1, noise=1e-06, normalize_values=true, "
              "fit_hyperparameters=true, fit_noise=false, min_length_scale=0, max_length_scale=0, "
              "acquisition=ExpectedImprovement, exploration=0, kappa=2)");
    op::BayesianOptimizationResult r;
    r.best_value = -1.0, r.best_point = {0.5, 2.0}, r.values = {3.0, -1.0};
    EXPECT_EQ(std::format("{}", r), "BayesianOptimizationResult(best_value=-1, best_point=[0.5, 2], evaluations=2)");
}

TEST(OptimFormat, RBFGaussianProcessSurrogateShowsHyperparametersAndFit) {
    op::RBFGaussianProcessSurrogate surrogate(0.5, 0.01);
    EXPECT_EQ(std::format("{}", surrogate), "RBFGaussianProcessSurrogate(length_scale=0.5, noise=0.01, unfitted)");
    surrogate.fit({{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}}, {1.0, 2.0, 3.0});
    EXPECT_EQ(std::format("{}", surrogate), "RBFGaussianProcessSurrogate(length_scale=0.5, noise=0.01, n=3, dim=2)");
}
