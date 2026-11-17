#pragma once
/// @file tabular_classification.hpp
/// @brief Native tensor fixtures, train-only scaling and two-variable classification figures.
#include <datamunge/nn/layers.hpp>
#include <datamunge/plot/plot.hpp>
#include <functional>
namespace datamunge::nn::study {
/// @brief Owned two-predictor classification fixture, with no fitted model.
/// @par Example
/// @code
/// auto d=dm::nn::study::xor_classification_data(1);std::print("{}\n",d.features.size(0)); // 4
/// @endcode
/// @see xor_classification_data(), moon_classification_data(), iris_petal_classification_data()
struct TabularClassificationData {
  Tensor                   features;   ///< Float32 [N,2] raw features, no gradient history.
  Tensor                   labels;     ///< Int64 [N] class IDs.
  std::vector<std::string> classes;    ///< Ordered display names matching IDs.
  std::vector<std::string> predictors; ///< Two original feature names.
};
/// @brief Generate the original four noisy XOR corners with a local MT19937_64 stream.
/// Corner order(0,0),(0,1),(1,0),(1,1), drawing x jitter then y jitter for each row.
/// Noise is normal(0,.12), seed1; labels0 for equal corners and1 otherwise.
/// O(N) work/storage, no global RNG state mutation or fitting.
/// @param per_corner Positive rows per corner, default150, at most250000.
/// @return Owned raw tensor data with classes A,B and predictors x,y.
/// @throws std::invalid_argument If row control is outside admitted bounds.
/// @par Example
/// @code
/// auto d=dm::nn::study::xor_classification_data();std::print("{}\n",d.labels.size(0)); // 600
/// @endcode
/// @see TabularClassificationData
TabularClassificationData xor_classification_data(std::size_t per_corner = 150);
/// @brief Generate the original noisy interleaved moons using its exact draw schedule.
/// Uses theta=pi*i/(per_moon-1), normal(0,.12), MT19937_64 seed3. Each row pair
/// draws upper x/y then lower x/y; lower=(1-cos(theta),.5-sin(theta)).
/// O(N) work/storage; no fitting or caller RNG mutation.
/// @param per_moon At least2 rows per moon, default250, at most250000.
/// @return Owned interleaved raw features, classes upper,lower and predictors x,y.
/// @throws std::invalid_argument If row control is outside admitted bounds.
/// @par Example
/// @code
/// auto d=dm::nn::study::moon_classification_data();std::print("{}\n",d.labels.size(0)); // 500
/// @endcode
/// @see xor_classification_data()
TabularClassificationData moon_classification_data(std::size_t per_moon = 250);
/// @brief Load bundled Iris with only Petal.Length and Petal.Width, without fitting/scaling.
/// @return Owned Float32 [150,2], Int64 labels, classes setosa/versicolor/virginica.
/// @par Example
/// @code
/// auto d=dm::nn::study::iris_petal_classification_data();std::print("{}\n",d.features.size(1)); // 2
/// @endcode
/// @par References
/// - Fisher(1936), The use of multiple measurements in taxonomic problems, doi:10.1111/j.1469-1809.1936.tb02137.x.
/// @see datasets::iris()
TabularClassificationData iris_petal_classification_data();
/// @brief Owned actual seeded row partition; no labels or features are consulted.
/// @par Example
/// @code
/// auto s=dm::nn::study::tabular_classification_split(10,.7,11);std::print("{}\n",s.training.size()); // 7
/// @endcode
/// @see tabular_classification_split()
struct TabularClassificationSplit {
  std::vector<std::int64_t> training; ///< First floor(fraction*N) shuffled row IDs.
  std::vector<std::int64_t> heldout;  ///< Remaining shuffled row IDs; disjoint and complete.
};
/// @brief Shuffle original row IDs with MT19937_64 and partition without replacement.
/// O(N) work/storage, deliberately not stratified; preserves original report splits.
/// @param rows At least2 rows, representable by Int64 and container geometry.
/// @param training_fraction Finite fraction strictly between0 and1, both subsets nonempty.
/// @param seed Local MT19937_64 shuffle seed.
/// @return Owned complete disjoint IDs in actual shuffled order.
/// @throws std::invalid_argument If controls or resulting subset sizes are invalid.
/// @throws std::length_error If row geometry is unrepresentable.
/// @par Example
/// @code
/// auto s=dm::nn::study::tabular_classification_split(150,.7,2024);std::print("{} {}\n",s.training.size(),s.heldout.size()); // 105 45
/// @endcode
/// @see TabularClassificationSplit
TabularClassificationSplit
tabular_classification_split(std::size_t rows, double training_fraction, std::uint64_t seed);
/// @brief Frozen training-column population mean and scale computed using tensor reductions.
/// Constant columns use scale1. Values own detached storage; inference never updates them.
/// @par Example
/// @code
/// auto s=dm::nn::study::fit_tabular_standardization(dm::nn::tensor({0.,2.},{2,1}));std::print("{}\n",s.scale.item()); // 1
/// @endcode
/// @see fit_tabular_standardization(), transform_tabular_features()
struct TabularStandardization {
  Tensor mean;  ///< Float32/Float64 [1,D] detached training-column means.
  Tensor scale; ///< Same dtype/shape positive population SD, constant columns1.
};
/// @brief Fit standardization on supplied training rows only, with no model/optimizer changes.
/// O(ND) tensor reductions. Population variance uses divisor N, not N-1.
/// @param training Nonempty finite Float32/Float64 [N,D], positive D.
/// @return Owned detached row-broadcast means and scales.
/// @throws std::invalid_argument If training tensor shape/dtype/values are invalid.
/// @throws std::overflow_error If mean/variance scaling is nonfinite.
/// @par Example
/// @code
/// auto s=dm::nn::study::fit_tabular_standardization(dm::nn::tensor({0.,2.},{2,1}));std::print("{}\n",s.mean.item()); // 1
/// @endcode
/// @see TabularStandardization
TabularStandardization fit_tabular_standardization(const Tensor& training);
/// @brief Apply frozen training statistics through differentiable tensor arithmetic.
/// No refit or input mutation. O(ND) work/output; gradients flow to input when enabled.
/// @param input Finite floating [N,D] matching fitted statistics, N positive.
/// @param fitted Finite same-dtype [1,D] mean and positive scale.
/// @return Tensor(input-mean)/scale; rejects nonfinite transformed values.
/// @throws std::invalid_argument If complete tensor/statistic geometry or values are invalid.
/// @throws std::overflow_error If standardized values become nonfinite.
/// @par Example
/// @code
/// auto x=dm::nn::tensor({0.,2.},{2,1});auto s=dm::nn::study::fit_tabular_standardization(x);std::print("{}\n",dm::nn::study::transform_tabular_features(x,s).at({0,0})); // -1
/// @endcode
/// @see fit_tabular_standardization()
Tensor transform_tabular_features(const Tensor& input, const TabularStandardization& fitted);
/// @brief Plot actual tensor classifier argmax on a raw two-variable grid with labeled observations.
/// Uses training-feature extrema plus10% padding; does not inspect held-out features
/// to choose ranges or refit statistics. Query logits callback must apply the same
/// frozen training standardization as its model. Evaluation runs with NoGradGuard.
/// Cells display argmax class IDs; no posterior calibration or extrapolation claim.
/// O(G squared) callback rows/output, plus observation copy; caller model is borrowed.
/// @param raw_features Finite floating [N,2] observations, N positive.
/// @param labels Matching Int32/Int64 [N] class IDs in0..classes.size()-1.
/// @param classes Two or three nonempty ordered class names.
/// @param logits Nonempty sequential callback returning finite same-dtype [G*G,C] logits.
/// @param grid_points At least2, at most201 points on each axis, default61.
/// @return Native plot with actual grid classes and class-colored observed points.
/// @throws std::invalid_argument If complete shapes/labels/control/callback output is invalid.
/// @throws std::overflow_error If plotting bounds/grid become nonfinite.
/// @par Example
/// @code
/// auto d=dm::nn::study::xor_classification_data(1);auto p=dm::nn::study::tabular_decision_regions(d.features,d.labels,d.classes,[](const auto& x){return x;},5);
/// @endcode
/// @see graphics::RPlot::image(), transform_tabular_features()
graphics::RPlot tabular_decision_regions(
    const Tensor&                               raw_features,
    const Tensor&                               labels,
    const std::vector<std::string>&             classes,
    const std::function<Tensor(const Tensor&)>& logits,
    int                                         grid_points = 61);
} // namespace datamunge::nn::study
