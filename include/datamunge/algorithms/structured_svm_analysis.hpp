#pragma once
/// @file structured_svm_analysis.hpp
/// @brief Reproducible two-feature classification cohorts and native decision-grid predictions.
#include <datamunge/algorithms/structured_svm.hpp>
namespace datamunge::algorithms {
/// @brief Own the StructuredSVMCohort contract for one independent native call.
/// Field descriptions specify actual producer values or policy units. Defaults are in-class initializers. Containers own their values; no borrowed storage or shared mutable state. Creating the record alone performs no algorithm execution.
/// @par Example
/// @code
/// dm::algorithms::StructuredSVMCohort r{};
/// (void)r;
/// @endcode
/// @see structured_svm_run()
struct StructuredSVMCohort {
  std::vector<std::vector<double>> features; ///< Owned two-feature rows, contiguous class groups.
  std::vector<int>                 labels;   ///< Matching actual generating class0,1,2.
};
/// @brief Own the StructuredSVMRegions contract for one independent native call.
/// Field descriptions specify actual producer values or policy units. Defaults are in-class initializers. Containers own their values; no borrowed storage or shared mutable state. Creating the record alone performs no algorithm execution.
/// @par Example
/// @code
/// dm::algorithms::StructuredSVMRegions r{};
/// (void)r;
/// @endcode
/// @see structured_svm_run()
struct StructuredSVMRegions {
  std::vector<std::vector<double>> x; ///< Actual mesh x coordinates grouped by model prediction.
  std::vector<std::vector<double>> y; ///< Matching actual mesh y coordinates.
};
/// @brief Generate the original three Gaussian class clusters or an independent seeded cohort.
/// Centers(0,0),(6,0),(3,6), independent normal jitter standard deviation0.5; local mt19937_64, all rows in class order. Default60/class and seed3 reproduce the original180 training rows. A distinct seed is an independent controlled held-out cohort, not real-world validation. At most100000 rows/class; zero yields empty.
/// @param per_class Owned rows per class, default60.
/// @param seed Local engine seed, default3.
/// @return Owned two-feature rows and matching class labels.
/// @throws std::length_error More than100000 rows per class.
/// @par Example
/// @code
/// auto d=dm::algorithms::structured_svm_example_cohort();
/// std::print("{} {}\n",d.features.size(),d.labels.size()); // 180 180
/// @endcode
/// @see structured_svm_regions()
StructuredSVMCohort structured_svm_example_cohort(
    std::size_t   per_class = 60,
    std::uint64_t seed      = 3);
/// @brief Predict every point of a native two-dimensional square decision grid.
/// Evaluates actual checked model predictions at evenly spaced inclusive axis coordinates. Requires finite low<high, two-feature nonempty valid model and points2..1000. Returns owned class-grouped coordinates with no report-side classification replay; no caller mutation/shared state. O(points squared times validated model size).
/// @param model Borrowed complete two-feature fitted model.
/// @param low Finite lower x/y endpoint, default-2.1.
/// @param high Finite upper x/y endpoint, default7.
/// @param points Inclusive count on each axis, default80.
/// @return Actual mesh coordinates grouped by predicted class.
/// @throws std::invalid_argument Invalid model, range or grid count.
/// @throws std::overflow_error Actual score is not finite.
/// @par Example
/// @code
/// auto m=dm::algorithms::structured_svm_train({{0,0}},{0},1,{});
/// auto r=dm::algorithms::structured_svm_regions(m,-1,1,2);
/// std::print("{}\n",r.x[0].size()); // 4
/// @endcode
/// @see structured_svm_predict()
StructuredSVMRegions structured_svm_regions(
    const StructuredSVMModel& model,
    double                    low    = -2.1,
    double                    high   = 7,
    std::size_t               points = 80);
} // namespace datamunge::algorithms
