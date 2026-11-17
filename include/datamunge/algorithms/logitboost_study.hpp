#pragma once
/// @file logitboost_study.hpp
/// @brief Reproducible original LogitBoost fixture and a separate held-out Iris study.
#include <datamunge/algorithms/logitboost.hpp>
#include <datamunge/datasets/datasets.hpp>
#include <random>
namespace datamunge::algorithms {
/// @brief Owned binary classification observations.
struct LogitBoostData {
 std::vector<std::vector<double>> features; ///< Ordered two-dimensional feature rows.
 std::vector<int> labels; ///< Binary labels in the same order.
};
/// @brief Generate the original300 candidate diagonal problem, discarding abs(x+y)<0.3.
/// @param seed Seed for mt19937_64; default3 retains the original fixture.
/// @return Owned retained observations, in generator order.
/// @note Uniform-real distribution bit patterns depend on the standard library implementation.
/// @code
/// auto data=dm::algorithms::logitboost_diagonal_data();
/// std::print("{}\n",data.labels.size()); // 271
/// @endcode
inline LogitBoostData logitboost_diagonal_data(std::uint64_t seed=3) {
 std::mt19937_64 rng(seed);std::uniform_real_distribution<double>u(-3,3);LogitBoostData data;
 for(int i=0;i<300;++i){const double a=u(rng),b=u(rng);if(std::abs(a+b)<.3)continue;data.features.push_back({a,b});data.labels.push_back(a+b>0);}
 return data;
}
/// @brief Owned deterministic binary Iris train/test partition and fitted test predictions.
struct LogitBoostIrisStudy {
 LogitBoostData training; ///< Versicolor/virginica petal rows except every fifth row per species.
 LogitBoostData testing; ///< Every fifth row per species, unseen during fitting.
 LogitBoostModel model; ///< Ensemble fitted only on training rows, with fixed settings.
 std::vector<int> predictions; ///< Actual held-out predicted labels.
 std::vector<double> probabilities; ///< Actual held-out model probabilities, without calibration claims.
};
/// @brief Fit a fixed40-round two-petal binary Iris model and predict a separate deterministic holdout.
/// @return Owned80 training and20 test rows, fitted ensemble and actual test predictions.
/// @note Labels0/1 mean versicolor/virginica. Every fifth original row within each species is held out.
/// This fixed illustrative split is not cross-validation; no hyperparameters are tuned on its test set.
/// @code
/// auto study=dm::algorithms::logitboost_iris_study();
/// std::print("{} {}\n",study.training.labels.size(),study.testing.labels.size()); // 80 20
/// @endcode
inline LogitBoostIrisStudy logitboost_iris_study() {
 const auto iris=datamunge::datasets::iris();LogitBoostIrisStudy out;std::size_t counts[2]={0,0};
 for(std::size_t i=0;i<iris.nrows();++i){const auto row=iris.row(i);const auto species=row.get_string("Species");if(species=="setosa")continue;const int label=species=="virginica"?1:0;auto&part=(counts[label]++%5==0)?out.testing:out.training;part.features.push_back({row.get_double("Petal.Length"),row.get_double("Petal.Width")});part.labels.push_back(label);}
 out.model=logitboost_train(out.training.features,out.training.labels,40);
 for(const auto&x:out.testing.features){const auto p=logitboost_probability(out.model,x);out.probabilities.push_back(p);out.predictions.push_back(p>=.5);}
 return out;
}
}
