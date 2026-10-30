#pragma once

/// @file stats.hpp
/// @brief std::format / std::print support for datamunge::stats.
/// Included by <datamunge/format.hpp>; requires C++20 <format>.
///
/// Fitted models print their summary() (the text print_summary() writes);
/// options, predictions and test results print as one line,
/// `TypeName(field=value, ...)`, with long vectors elided; enums print their
/// enumerator name.
///
/// @par Example
/// @code
/// const auto r = dm::stats::t_test_one_sample({4.9, 5.1, 5.3, 4.7, 5.0}, 4.8);
/// std::print("{}\n", r);
/// // HypothesisTestResult(method="One Sample t-test", statistic=2.0000000000000036, parameter1=4,
/// //   parameter2=0, p_value=0.11611652351681512, estimate1=5, estimate2=0,
/// //   conf_int=[4.722355489480129, 5.277644510519871], alternative=TwoSided)   (one line)
/// std::print("{}\n", dm::stats::KMeansOptions{});
/// // KMeansOptions(n_clusters=8, max_iterations=300, n_init=10, tolerance=1e-04, seed=42)
/// std::print("{}\n", dm::stats::PAdjustMethod::Holm);  // Holm
/// std::print("{}\n", dm::stats::LM(df, "len ~ supp * dose"));  // the model's summary()
/// @endcode

#include <datamunge/format/text_formatter.hpp>
#include <datamunge/stats/stats.hpp>

#include <cstddef>
#include <format>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

/// @brief Shared building blocks for the library's std::format / std::print support.
namespace datamunge::format_detail {
/// @brief Text helpers used by the datamunge::stats formatters; not part of the public API.
namespace stats_fmt {

/// @brief Quotes and escapes a string, as `{:?}` does.
/// @param s The string to quote.
/// @return The quoted string, e.g. `"lac"`.
inline std::string quoted(const std::string& s) { return std::format("{:?}", s); }

/// @brief Elided preview of a vector.
/// @tparam T A formattable element type.
/// @param v The values.
/// @return E.g. `[1, 2, 3, 4, 5, ..., 99] (n=100)`.
template <typename T>
std::string vec(const std::vector<T>& v) { return preview(v); }

/// @brief Elided preview of a vector of rows (e.g. per-observation class probabilities).
/// @param rows The rows; each row is printed in full.
/// @return E.g. `[[0.9, 0.1], [0.2, 0.8], ..., [0.5, 0.5]] (n=150)`.
inline std::string rows(const std::vector<std::vector<double>>& rows) { return preview(rows, 3); }

/// @brief An optional column name.
/// @param value The optional name.
/// @return The quoted name, or `none`.
inline std::string optional_text(const std::optional<std::string>& value) {
  return value ? quoted(*value) : std::string("none");
}

/// @brief An optional vector.
/// @param value The optional values.
/// @return The vector preview, or `none`.
inline std::string optional_vec(const std::optional<std::vector<double>>& value) {
  return value ? preview(*value) : std::string("none");
}

/// @brief A (lower, upper) interval.
/// @param bounds The interval bounds.
/// @return `[lower, upper]`.
inline std::string interval(const std::pair<double, double>& bounds) {
  return std::format("[{}, {}]", bounds.first, bounds.second);
}

/// @brief A matrix's shape.
/// @param m The matrix.
/// @return `rows x cols`.
inline std::string shape(const datamunge::linalg::DenseMatrix<double>& m) {
  return std::format("{}x{}", m.rows(), m.cols());
}

/// @brief Names a bayes::INLAIntegrationStrategy (local mapping so this header does not depend on the bayes formatters).
/// @param strategy The strategy.
/// @return The enumerator name.
inline std::string strategy_text(datamunge::bayes::INLAIntegrationStrategy strategy) {
  switch (strategy) {
    case datamunge::bayes::INLAIntegrationStrategy::Grid: return "Grid";
    case datamunge::bayes::INLAIntegrationStrategy::EmpiricalBayes: return "EmpiricalBayes";
  }
  return std::format("INLAIntegrationStrategy({})", static_cast<int>(strategy));
}

/// @brief Renders a formula expression tree as infix text.
/// @param node The node to render.
/// @param top Whether this is the outermost node (binary operations are parenthesized only below the top).
/// @return E.g. `x * (y + 2)`.
inline std::string expr(const datamunge::stats::ExprNode& node, bool top = true) {
  using datamunge::stats::ExprOp;
  const auto side = [](const std::shared_ptr<datamunge::stats::ExprNode>& child) {
    return child ? expr(*child, false) : std::string("?");
  };
  const char* op = nullptr;
  switch (node.op) {
    case ExprOp::Literal: return std::format("{}", node.literal);
    case ExprOp::Column: return node.column;
    case ExprOp::Negate: return "-" + side(node.left);
    case ExprOp::Add: op = " + "; break;
    case ExprOp::Subtract: op = " - "; break;
    case ExprOp::Multiply: op = " * "; break;
    case ExprOp::Divide: op = " / "; break;
    case ExprOp::Power: op = "^"; break;
  }
  if (op == nullptr) return std::format("<op {}>", static_cast<int>(node.op));
  const std::string inner = side(node.left) + op + side(node.right);
  return top ? inner : "(" + inner + ")";
}

}  // namespace stats_fmt
}  // namespace datamunge::format_detail

// ---------------------------------------------------------------- enums

/// @brief Prints a stats::LinkageCriterion as its enumerator name (e.g. `Single`).
template <>
struct std::formatter<datamunge::stats::LinkageCriterion> : datamunge::format_detail::text_formatter<datamunge::stats::LinkageCriterion> {
  /// @brief The enumerator name.
  /// @param x The value to print.
  /// @return The enumerator name, or `LinkageCriterion(<int>)` for an unknown value.
  static std::string text(const datamunge::stats::LinkageCriterion& x) {
    switch (x) {
      case datamunge::stats::LinkageCriterion::Single: return "Single";
      case datamunge::stats::LinkageCriterion::Complete: return "Complete";
      case datamunge::stats::LinkageCriterion::Average: return "Average";
      case datamunge::stats::LinkageCriterion::Ward: return "Ward";
    }
    return std::format("LinkageCriterion({})", static_cast<int>(x));
  }
};

/// @brief Prints a stats::SplitCriterion as its enumerator name (e.g. `Gini`).
template <>
struct std::formatter<datamunge::stats::SplitCriterion> : datamunge::format_detail::text_formatter<datamunge::stats::SplitCriterion> {
  /// @brief The enumerator name.
  /// @param x The value to print.
  /// @return The enumerator name, or `SplitCriterion(<int>)` for an unknown value.
  static std::string text(const datamunge::stats::SplitCriterion& x) {
    switch (x) {
      case datamunge::stats::SplitCriterion::Gini: return "Gini";
      case datamunge::stats::SplitCriterion::Entropy: return "Entropy";
    }
    return std::format("SplitCriterion({})", static_cast<int>(x));
  }
};

/// @brief Prints a stats::QuantileRegressionMethod as its enumerator name (e.g. `ADMM`).
template <>
struct std::formatter<datamunge::stats::QuantileRegressionMethod> : datamunge::format_detail::text_formatter<datamunge::stats::QuantileRegressionMethod> {
  /// @brief The enumerator name.
  /// @param x The value to print.
  /// @return The enumerator name, or `QuantileRegressionMethod(<int>)` for an unknown value.
  static std::string text(const datamunge::stats::QuantileRegressionMethod& x) {
    switch (x) {
      case datamunge::stats::QuantileRegressionMethod::ADMM: return "ADMM";
      case datamunge::stats::QuantileRegressionMethod::Simplex: return "Simplex";
    }
    return std::format("QuantileRegressionMethod({})", static_cast<int>(x));
  }
};

/// @brief Prints a stats::QuantileStandardErrors as its enumerator name (e.g. `NID`).
template <>
struct std::formatter<datamunge::stats::QuantileStandardErrors> : datamunge::format_detail::text_formatter<datamunge::stats::QuantileStandardErrors> {
  /// @brief The enumerator name.
  /// @param x The value to print.
  /// @return The enumerator name, or `QuantileStandardErrors(<int>)` for an unknown value.
  static std::string text(const datamunge::stats::QuantileStandardErrors& x) {
    switch (x) {
      case datamunge::stats::QuantileStandardErrors::IID: return "IID";
      case datamunge::stats::QuantileStandardErrors::NID: return "NID";
      case datamunge::stats::QuantileStandardErrors::Bootstrap: return "Bootstrap";
    }
    return std::format("QuantileStandardErrors({})", static_cast<int>(x));
  }
};

/// @brief Prints a stats::TrendType as its enumerator name (e.g. `None`).
template <>
struct std::formatter<datamunge::stats::TrendType> : datamunge::format_detail::text_formatter<datamunge::stats::TrendType> {
  /// @brief The enumerator name.
  /// @param x The value to print.
  /// @return The enumerator name, or `TrendType(<int>)` for an unknown value.
  static std::string text(const datamunge::stats::TrendType& x) {
    switch (x) {
      case datamunge::stats::TrendType::None: return "None";
      case datamunge::stats::TrendType::Additive: return "Additive";
      case datamunge::stats::TrendType::AdditiveDamped: return "AdditiveDamped";
    }
    return std::format("TrendType({})", static_cast<int>(x));
  }
};

/// @brief Prints a stats::SeasonalType as its enumerator name (e.g. `None`).
template <>
struct std::formatter<datamunge::stats::SeasonalType> : datamunge::format_detail::text_formatter<datamunge::stats::SeasonalType> {
  /// @brief The enumerator name.
  /// @param x The value to print.
  /// @return The enumerator name, or `SeasonalType(<int>)` for an unknown value.
  static std::string text(const datamunge::stats::SeasonalType& x) {
    switch (x) {
      case datamunge::stats::SeasonalType::None: return "None";
      case datamunge::stats::SeasonalType::Additive: return "Additive";
      case datamunge::stats::SeasonalType::Multiplicative: return "Multiplicative";
    }
    return std::format("SeasonalType({})", static_cast<int>(x));
  }
};

/// @brief Prints a stats::ExprOp as its enumerator name (e.g. `Literal`).
template <>
struct std::formatter<datamunge::stats::ExprOp> : datamunge::format_detail::text_formatter<datamunge::stats::ExprOp> {
  /// @brief The enumerator name.
  /// @param x The value to print.
  /// @return The enumerator name, or `ExprOp(<int>)` for an unknown value.
  static std::string text(const datamunge::stats::ExprOp& x) {
    switch (x) {
      case datamunge::stats::ExprOp::Literal: return "Literal";
      case datamunge::stats::ExprOp::Column: return "Column";
      case datamunge::stats::ExprOp::Add: return "Add";
      case datamunge::stats::ExprOp::Subtract: return "Subtract";
      case datamunge::stats::ExprOp::Multiply: return "Multiply";
      case datamunge::stats::ExprOp::Divide: return "Divide";
      case datamunge::stats::ExprOp::Power: return "Power";
      case datamunge::stats::ExprOp::Negate: return "Negate";
    }
    return std::format("ExprOp({})", static_cast<int>(x));
  }
};

/// @brief Prints a stats::ResolvedColumnKind as its enumerator name (e.g. `Numeric`).
template <>
struct std::formatter<datamunge::stats::ResolvedColumnKind> : datamunge::format_detail::text_formatter<datamunge::stats::ResolvedColumnKind> {
  /// @brief The enumerator name.
  /// @param x The value to print.
  /// @return The enumerator name, or `ResolvedColumnKind(<int>)` for an unknown value.
  static std::string text(const datamunge::stats::ResolvedColumnKind& x) {
    switch (x) {
      case datamunge::stats::ResolvedColumnKind::Numeric: return "Numeric";
      case datamunge::stats::ResolvedColumnKind::CategoricalDummy: return "CategoricalDummy";
      case datamunge::stats::ResolvedColumnKind::Function: return "Function";
      case datamunge::stats::ResolvedColumnKind::BSpline: return "BSpline";
      case datamunge::stats::ResolvedColumnKind::Interaction: return "Interaction";
      case datamunge::stats::ResolvedColumnKind::Expression: return "Expression";
    }
    return std::format("ResolvedColumnKind({})", static_cast<int>(x));
  }
};

/// @brief Prints a stats::ResponseKind as its enumerator name (e.g. `Numeric`).
template <>
struct std::formatter<datamunge::stats::ResponseKind> : datamunge::format_detail::text_formatter<datamunge::stats::ResponseKind> {
  /// @brief The enumerator name.
  /// @param x The value to print.
  /// @return The enumerator name, or `ResponseKind(<int>)` for an unknown value.
  static std::string text(const datamunge::stats::ResponseKind& x) {
    switch (x) {
      case datamunge::stats::ResponseKind::Numeric: return "Numeric";
      case datamunge::stats::ResponseKind::Categorical: return "Categorical";
    }
    return std::format("ResponseKind({})", static_cast<int>(x));
  }
};

/// @brief Prints a stats::GLMFamily as its enumerator name (e.g. `Gaussian`).
template <>
struct std::formatter<datamunge::stats::GLMFamily> : datamunge::format_detail::text_formatter<datamunge::stats::GLMFamily> {
  /// @brief The enumerator name.
  /// @param x The value to print.
  /// @return The enumerator name, or `GLMFamily(<int>)` for an unknown value.
  static std::string text(const datamunge::stats::GLMFamily& x) {
    switch (x) {
      case datamunge::stats::GLMFamily::Gaussian: return "Gaussian";
      case datamunge::stats::GLMFamily::Binomial: return "Binomial";
      case datamunge::stats::GLMFamily::Poisson: return "Poisson";
      case datamunge::stats::GLMFamily::Gamma: return "Gamma";
    }
    return std::format("GLMFamily({})", static_cast<int>(x));
  }
};

/// @brief Prints a stats::GLMPredictionInterval as its enumerator name (e.g. `None`).
template <>
struct std::formatter<datamunge::stats::GLMPredictionInterval> : datamunge::format_detail::text_formatter<datamunge::stats::GLMPredictionInterval> {
  /// @brief The enumerator name.
  /// @param x The value to print.
  /// @return The enumerator name, or `GLMPredictionInterval(<int>)` for an unknown value.
  static std::string text(const datamunge::stats::GLMPredictionInterval& x) {
    switch (x) {
      case datamunge::stats::GLMPredictionInterval::None: return "None";
      case datamunge::stats::GLMPredictionInterval::Confidence: return "Confidence";
    }
    return std::format("GLMPredictionInterval({})", static_cast<int>(x));
  }
};

/// @brief Prints a stats::GLMMFamily as its enumerator name (e.g. `Binomial`).
template <>
struct std::formatter<datamunge::stats::GLMMFamily> : datamunge::format_detail::text_formatter<datamunge::stats::GLMMFamily> {
  /// @brief The enumerator name.
  /// @param x The value to print.
  /// @return The enumerator name, or `GLMMFamily(<int>)` for an unknown value.
  static std::string text(const datamunge::stats::GLMMFamily& x) {
    switch (x) {
      case datamunge::stats::GLMMFamily::Binomial: return "Binomial";
      case datamunge::stats::GLMMFamily::Poisson: return "Poisson";
    }
    return std::format("GLMMFamily({})", static_cast<int>(x));
  }
};

/// @brief Prints a stats::Alternative as its enumerator name (e.g. `TwoSided`).
template <>
struct std::formatter<datamunge::stats::Alternative> : datamunge::format_detail::text_formatter<datamunge::stats::Alternative> {
  /// @brief The enumerator name.
  /// @param x The value to print.
  /// @return The enumerator name, or `Alternative(<int>)` for an unknown value.
  static std::string text(const datamunge::stats::Alternative& x) {
    switch (x) {
      case datamunge::stats::Alternative::TwoSided: return "TwoSided";
      case datamunge::stats::Alternative::Less: return "Less";
      case datamunge::stats::Alternative::Greater: return "Greater";
    }
    return std::format("Alternative({})", static_cast<int>(x));
  }
};

/// @brief Prints a stats::INLAMixedModelFamily as its enumerator name (e.g. `Gaussian`).
template <>
struct std::formatter<datamunge::stats::INLAMixedModelFamily> : datamunge::format_detail::text_formatter<datamunge::stats::INLAMixedModelFamily> {
  /// @brief The enumerator name.
  /// @param x The value to print.
  /// @return The enumerator name, or `INLAMixedModelFamily(<int>)` for an unknown value.
  static std::string text(const datamunge::stats::INLAMixedModelFamily& x) {
    switch (x) {
      case datamunge::stats::INLAMixedModelFamily::Gaussian: return "Gaussian";
      case datamunge::stats::INLAMixedModelFamily::Binomial: return "Binomial";
      case datamunge::stats::INLAMixedModelFamily::Poisson: return "Poisson";
    }
    return std::format("INLAMixedModelFamily({})", static_cast<int>(x));
  }
};

/// @brief Prints a stats::KernelRegressionKernel as its enumerator name (e.g. `Gaussian`).
template <>
struct std::formatter<datamunge::stats::KernelRegressionKernel> : datamunge::format_detail::text_formatter<datamunge::stats::KernelRegressionKernel> {
  /// @brief The enumerator name.
  /// @param x The value to print.
  /// @return The enumerator name, or `KernelRegressionKernel(<int>)` for an unknown value.
  static std::string text(const datamunge::stats::KernelRegressionKernel& x) {
    switch (x) {
      case datamunge::stats::KernelRegressionKernel::Gaussian: return "Gaussian";
      case datamunge::stats::KernelRegressionKernel::Epanechnikov: return "Epanechnikov";
      case datamunge::stats::KernelRegressionKernel::Uniform: return "Uniform";
      case datamunge::stats::KernelRegressionKernel::Triangular: return "Triangular";
    }
    return std::format("KernelRegressionKernel({})", static_cast<int>(x));
  }
};

/// @brief Prints a stats::DistanceMetric as its enumerator name (e.g. `Euclidean`).
template <>
struct std::formatter<datamunge::stats::DistanceMetric> : datamunge::format_detail::text_formatter<datamunge::stats::DistanceMetric> {
  /// @brief The enumerator name.
  /// @param x The value to print.
  /// @return The enumerator name, or `DistanceMetric(<int>)` for an unknown value.
  static std::string text(const datamunge::stats::DistanceMetric& x) {
    switch (x) {
      case datamunge::stats::DistanceMetric::Euclidean: return "Euclidean";
      case datamunge::stats::DistanceMetric::Manhattan: return "Manhattan";
    }
    return std::format("DistanceMetric({})", static_cast<int>(x));
  }
};

/// @brief Prints a stats::PredictionInterval as its enumerator name (e.g. `None`).
template <>
struct std::formatter<datamunge::stats::PredictionInterval> : datamunge::format_detail::text_formatter<datamunge::stats::PredictionInterval> {
  /// @brief The enumerator name.
  /// @param x The value to print.
  /// @return The enumerator name, or `PredictionInterval(<int>)` for an unknown value.
  static std::string text(const datamunge::stats::PredictionInterval& x) {
    switch (x) {
      case datamunge::stats::PredictionInterval::None: return "None";
      case datamunge::stats::PredictionInterval::Confidence: return "Confidence";
      case datamunge::stats::PredictionInterval::Prediction: return "Prediction";
    }
    return std::format("PredictionInterval({})", static_cast<int>(x));
  }
};

/// @brief Prints a stats::PAdjustMethod as its enumerator name (e.g. `Bonferroni`).
template <>
struct std::formatter<datamunge::stats::PAdjustMethod> : datamunge::format_detail::text_formatter<datamunge::stats::PAdjustMethod> {
  /// @brief The enumerator name.
  /// @param x The value to print.
  /// @return The enumerator name, or `PAdjustMethod(<int>)` for an unknown value.
  static std::string text(const datamunge::stats::PAdjustMethod& x) {
    switch (x) {
      case datamunge::stats::PAdjustMethod::Bonferroni: return "Bonferroni";
      case datamunge::stats::PAdjustMethod::Holm: return "Holm";
      case datamunge::stats::PAdjustMethod::Hochberg: return "Hochberg";
      case datamunge::stats::PAdjustMethod::Hommel: return "Hommel";
      case datamunge::stats::PAdjustMethod::BH: return "BH";
      case datamunge::stats::PAdjustMethod::BY: return "BY";
      case datamunge::stats::PAdjustMethod::None: return "None";
    }
    return std::format("PAdjustMethod({})", static_cast<int>(x));
  }
};

/// @brief Prints a stats::SVMKernel as its enumerator name (e.g. `Linear`).
template <>
struct std::formatter<datamunge::stats::SVMKernel> : datamunge::format_detail::text_formatter<datamunge::stats::SVMKernel> {
  /// @brief The enumerator name.
  /// @param x The value to print.
  /// @return The enumerator name, or `SVMKernel(<int>)` for an unknown value.
  static std::string text(const datamunge::stats::SVMKernel& x) {
    switch (x) {
      case datamunge::stats::SVMKernel::Linear: return "Linear";
      case datamunge::stats::SVMKernel::Polynomial: return "Polynomial";
      case datamunge::stats::SVMKernel::Radial: return "Radial";
      case datamunge::stats::SVMKernel::Sigmoid: return "Sigmoid";
    }
    return std::format("SVMKernel({})", static_cast<int>(x));
  }
};

// ---------------------------------------------------------------- options / results

/// @brief Prints a stats::AgglomerativeClusteringOptions (agglomerative clustering options) as one line, `AgglomerativeClusteringOptions(field=value, ...)`.
template <>
struct std::formatter<datamunge::stats::AgglomerativeClusteringOptions> : datamunge::format_detail::text_formatter<datamunge::stats::AgglomerativeClusteringOptions> {
  /// @brief One-line field listing.
  /// @param x The value to print.
  /// @return `AgglomerativeClusteringOptions(...)` with every field; vectors are elided previews.
  static std::string text(const datamunge::stats::AgglomerativeClusteringOptions& x) {
    return std::format("AgglomerativeClusteringOptions(n_clusters={}, linkage={}, metric={})", x.n_clusters, x.linkage, x.metric);
  }
};

/// @brief Prints a stats::AgglomerativeClustering::Merge (one dendrogram merge step) as one line, `Merge(field=value, ...)`.
template <>
struct std::formatter<datamunge::stats::AgglomerativeClustering::Merge> : datamunge::format_detail::text_formatter<datamunge::stats::AgglomerativeClustering::Merge> {
  /// @brief One-line field listing.
  /// @param x The value to print.
  /// @return `Merge(...)` with every field; vectors are elided previews.
  static std::string text(const datamunge::stats::AgglomerativeClustering::Merge& x) {
    return std::format("Merge(cluster_a={}, cluster_b={}, distance={}, size={})", x.cluster_a, x.cluster_b, x.distance, x.size);
  }
};

/// @brief Prints a stats::ARIMAOptions (ARIMA options) as one line, `ARIMAOptions(field=value, ...)`.
template <>
struct std::formatter<datamunge::stats::ARIMAOptions> : datamunge::format_detail::text_formatter<datamunge::stats::ARIMAOptions> {
  /// @brief One-line field listing.
  /// @param x The value to print.
  /// @return `ARIMAOptions(...)` with every field; vectors are elided previews.
  static std::string text(const datamunge::stats::ARIMAOptions& x) {
    return std::format("ARIMAOptions(p={}, d={}, q={}, seasonal_p={}, seasonal_d={}, seasonal_q={}, seasonal_period={}, include_mean={}, de_population_size={}, de_max_generations={}, coefficient_bound={}, seed={})", x.p, x.d, x.q, x.seasonal_p, x.seasonal_d, x.seasonal_q, x.seasonal_period, x.include_mean, x.de_population_size, x.de_max_generations, x.coefficient_bound, x.seed);
  }
};

/// @brief Prints a stats::BootstrapOptions (bootstrap options) as one line, `BootstrapOptions(field=value, ...)`.
template <>
struct std::formatter<datamunge::stats::BootstrapOptions> : datamunge::format_detail::text_formatter<datamunge::stats::BootstrapOptions> {
  /// @brief One-line field listing.
  /// @param x The value to print.
  /// @return `BootstrapOptions(...)` with every field; vectors are elided previews.
  static std::string text(const datamunge::stats::BootstrapOptions& x) {
    return std::format("BootstrapOptions(num_resamples={}, confidence_level={}, seed={})", x.num_resamples, x.confidence_level, x.seed);
  }
};

/// @brief Prints a stats::BootstrapResult (bootstrap result) as one line, `BootstrapResult(field=value, ...)`.
template <>
struct std::formatter<datamunge::stats::BootstrapResult> : datamunge::format_detail::text_formatter<datamunge::stats::BootstrapResult> {
  /// @brief One-line field listing.
  /// @param x The value to print.
  /// @return `BootstrapResult(...)` with every field; vectors are elided previews.
  static std::string text(const datamunge::stats::BootstrapResult& x) {
    namespace f = datamunge::format_detail::stats_fmt;
    return std::format("BootstrapResult(estimate={}, bias={}, standard_error={}, confidence_level={}, percentile_interval={}, basic_interval={}, normal_interval={}, bca_interval={}, replicates={})", x.estimate, x.bias, x.standard_error, x.confidence_level, f::interval(x.percentile_interval), f::interval(x.basic_interval), f::interval(x.normal_interval), f::interval(x.bca_interval), f::vec(x.replicates));
  }
};

/// @brief Prints a stats::CNNClassifierOptions (CNN classifier options) as one line, `CNNClassifierOptions(field=value, ...)`.
template <>
struct std::formatter<datamunge::stats::CNNClassifierOptions> : datamunge::format_detail::text_formatter<datamunge::stats::CNNClassifierOptions> {
  /// @brief One-line field listing.
  /// @param x The value to print.
  /// @return `CNNClassifierOptions(...)` with every field; vectors are elided previews.
  static std::string text(const datamunge::stats::CNNClassifierOptions& x) {
    return std::format("CNNClassifierOptions(n_filters={}, kernel_size={}, pool_size={}, dense_hidden={}, learning_rate={}, max_epochs={}, batch_size={}, l2={}, seed={})", x.n_filters, x.kernel_size, x.pool_size, x.dense_hidden, x.learning_rate, x.max_epochs, x.batch_size, x.l2, x.seed);
  }
};

/// @brief Prints a stats::CNNClassifierPrediction (CNN classifier prediction) as one line, `CNNClassifierPrediction(field=value, ...)`.
template <>
struct std::formatter<datamunge::stats::CNNClassifierPrediction> : datamunge::format_detail::text_formatter<datamunge::stats::CNNClassifierPrediction> {
  /// @brief One-line field listing.
  /// @param x The value to print.
  /// @return `CNNClassifierPrediction(...)` with every field; vectors are elided previews.
  static std::string text(const datamunge::stats::CNNClassifierPrediction& x) {
    namespace f = datamunge::format_detail::stats_fmt;
    return std::format("CNNClassifierPrediction(class_label={}, probabilities={})", f::vec(x.class_label), f::rows(x.probabilities));
  }
};

/// @brief Prints a stats::ConformalClassifierOptions (conformal classifier options) as one line, `ConformalClassifierOptions(field=value, ...)`.
template <>
struct std::formatter<datamunge::stats::ConformalClassifierOptions> : datamunge::format_detail::text_formatter<datamunge::stats::ConformalClassifierOptions> {
  /// @brief One-line field listing.
  /// @param x The value to print.
  /// @return `ConformalClassifierOptions(...)` with every field; vectors are elided previews.
  static std::string text(const datamunge::stats::ConformalClassifierOptions& x) {
    namespace f = datamunge::format_detail::stats_fmt;
    return std::format("ConformalClassifierOptions(alpha={}, score={}, class_conditional={}, nonempty={})", x.alpha,
                       f::quoted(x.score), x.class_conditional, x.nonempty);
  }
};

/// @brief Prints a stats::SplitConformalClassifier::Evaluation (conformal classifier evaluation) as one line, `Evaluation(field=value, ...)`.
template <>
struct std::formatter<datamunge::stats::SplitConformalClassifier::Evaluation> : datamunge::format_detail::text_formatter<datamunge::stats::SplitConformalClassifier::Evaluation> {
  /// @brief One-line field listing.
  /// @param x The value to print.
  /// @return `Evaluation(...)` with every field; vectors are elided previews.
  static std::string text(const datamunge::stats::SplitConformalClassifier::Evaluation& x) {
    namespace f = datamunge::format_detail::stats_fmt;
    return std::format("Evaluation(coverage={}, average_set_size={}, empty_rate={}, set_sizes={}, class_coverage={}, size_counts={})",
                       x.coverage, x.average_set_size, x.empty_rate, f::vec(x.set_sizes), f::vec(x.class_coverage),
                       f::vec(x.size_counts));
  }
};

/// @brief Prints a stats::ConformalRegressorOptions (conformal regressor options) as one line, `ConformalRegressorOptions(field=value, ...)`.
template <>
struct std::formatter<datamunge::stats::ConformalRegressorOptions> : datamunge::format_detail::text_formatter<datamunge::stats::ConformalRegressorOptions> {
  /// @brief One-line field listing.
  /// @param x The value to print.
  /// @return `ConformalRegressorOptions(...)` with every field; vectors are elided previews.
  static std::string text(const datamunge::stats::ConformalRegressorOptions& x) {
    return std::format("ConformalRegressorOptions(alpha={})", x.alpha);
  }
};

/// @brief Prints a stats::SplitConformalRegressor::Evaluation (conformal regressor evaluation) as one line, `Evaluation(field=value, ...)`.
template <>
struct std::formatter<datamunge::stats::SplitConformalRegressor::Evaluation> : datamunge::format_detail::text_formatter<datamunge::stats::SplitConformalRegressor::Evaluation> {
  /// @brief One-line field listing.
  /// @param x The value to print.
  /// @return `Evaluation(...)` with every field; vectors are elided previews.
  static std::string text(const datamunge::stats::SplitConformalRegressor::Evaluation& x) {
    namespace f = datamunge::format_detail::stats_fmt;
    return std::format("Evaluation(coverage={}, average_width={}, widths={})", x.coverage, x.average_width, f::vec(x.widths));
  }
};

/// @brief Prints a stats::ConformalCoverage (split-conformal coverage law) as one line of its fields.
template <>
struct std::formatter<datamunge::stats::ConformalCoverage> : datamunge::format_detail::text_formatter<datamunge::stats::ConformalCoverage> {
  /// @brief One-line field listing.
  /// @param x The value to print.
  /// @return `ConformalCoverage(rank=..., mean=..., beta_a=..., beta_b=..., sd=...)`.
  static std::string text(const datamunge::stats::ConformalCoverage& x) {
    return std::format("ConformalCoverage(rank={}, mean={}, beta_a={}, beta_b={}, sd={})", x.rank, x.mean, x.beta_a, x.beta_b, x.sd);
  }
};

/// @brief Prints a stats::DBSCANOptions (DBSCAN options) as one line, `DBSCANOptions(field=value, ...)`.
template <>
struct std::formatter<datamunge::stats::DBSCANOptions> : datamunge::format_detail::text_formatter<datamunge::stats::DBSCANOptions> {
  /// @brief One-line field listing.
  /// @param x The value to print.
  /// @return `DBSCANOptions(...)` with every field; vectors are elided previews.
  static std::string text(const datamunge::stats::DBSCANOptions& x) {
    return std::format("DBSCANOptions(eps={}, min_samples={}, metric={})", x.eps, x.min_samples, x.metric);
  }
};

/// @brief Prints a stats::DecisionTreeClassifierOptions (classification-tree options) as one line, `DecisionTreeClassifierOptions(field=value, ...)`.
template <>
struct std::formatter<datamunge::stats::DecisionTreeClassifierOptions> : datamunge::format_detail::text_formatter<datamunge::stats::DecisionTreeClassifierOptions> {
  /// @brief One-line field listing.
  /// @param x The value to print.
  /// @return `DecisionTreeClassifierOptions(...)` with every field; vectors are elided previews.
  static std::string text(const datamunge::stats::DecisionTreeClassifierOptions& x) {
    return std::format("DecisionTreeClassifierOptions(max_depth={}, min_samples_split={}, min_samples_leaf={}, min_impurity_decrease={}, criterion={}, max_features={}, random_seed={})", x.max_depth, x.min_samples_split, x.min_samples_leaf, x.min_impurity_decrease, x.criterion, x.max_features, x.random_seed);
  }
};

/// @brief Prints a stats::DecisionTreeClassifierPrediction (classification-tree prediction) as one line, `DecisionTreeClassifierPrediction(field=value, ...)`.
template <>
struct std::formatter<datamunge::stats::DecisionTreeClassifierPrediction> : datamunge::format_detail::text_formatter<datamunge::stats::DecisionTreeClassifierPrediction> {
  /// @brief One-line field listing.
  /// @param x The value to print.
  /// @return `DecisionTreeClassifierPrediction(...)` with every field; vectors are elided previews.
  static std::string text(const datamunge::stats::DecisionTreeClassifierPrediction& x) {
    namespace f = datamunge::format_detail::stats_fmt;
    return std::format("DecisionTreeClassifierPrediction(class_label={}, probability={})", f::vec(x.class_label), f::rows(x.probability));
  }
};

/// @brief Prints a stats::DecisionTreeRegressorOptions (regression-tree options) as one line, `DecisionTreeRegressorOptions(field=value, ...)`.
template <>
struct std::formatter<datamunge::stats::DecisionTreeRegressorOptions> : datamunge::format_detail::text_formatter<datamunge::stats::DecisionTreeRegressorOptions> {
  /// @brief One-line field listing.
  /// @param x The value to print.
  /// @return `DecisionTreeRegressorOptions(...)` with every field; vectors are elided previews.
  static std::string text(const datamunge::stats::DecisionTreeRegressorOptions& x) {
    return std::format("DecisionTreeRegressorOptions(max_depth={}, min_samples_split={}, min_samples_leaf={}, min_impurity_decrease={}, max_features={}, random_seed={})", x.max_depth, x.min_samples_split, x.min_samples_leaf, x.min_impurity_decrease, x.max_features, x.random_seed);
  }
};

/// @brief Prints a stats::DiffusionMapsOptions (diffusion-maps options) as one line, `DiffusionMapsOptions(field=value, ...)`.
template <>
struct std::formatter<datamunge::stats::DiffusionMapsOptions> : datamunge::format_detail::text_formatter<datamunge::stats::DiffusionMapsOptions> {
  /// @brief One-line field listing.
  /// @param x The value to print.
  /// @return `DiffusionMapsOptions(...)` with every field; vectors are elided previews.
  static std::string text(const datamunge::stats::DiffusionMapsOptions& x) {
    return std::format("DiffusionMapsOptions(n_components={}, heat_kernel_epsilon={}, alpha={}, diffusion_time={})", x.n_components, x.heat_kernel_epsilon, x.alpha, x.diffusion_time);
  }
};

/// @brief Prints a stats::ElasticNetOptions (elastic-net options) as one line, `ElasticNetOptions(field=value, ...)`.
template <>
struct std::formatter<datamunge::stats::ElasticNetOptions> : datamunge::format_detail::text_formatter<datamunge::stats::ElasticNetOptions> {
  /// @brief One-line field listing.
  /// @param x The value to print.
  /// @return `ElasticNetOptions(...)` with every field; vectors are elided previews.
  static std::string text(const datamunge::stats::ElasticNetOptions& x) {
    return std::format("ElasticNetOptions(alpha={}, lambda={}, n_lambda={}, cv_folds={}, max_iter={}, tol={}, seed={}, standardize={})", x.alpha, x.lambda, x.n_lambda, x.cv_folds, x.max_iter, x.tol, x.seed, x.standardize);
  }
};

/// @brief Prints a stats::RidgeOptions (ridge options) as one line, `RidgeOptions(field=value, ...)`.
template <>
struct std::formatter<datamunge::stats::RidgeOptions> : datamunge::format_detail::text_formatter<datamunge::stats::RidgeOptions> {
  /// @brief One-line field listing.
  /// @param x The value to print.
  /// @return `RidgeOptions(...)` with every field; vectors are elided previews.
  static std::string text(const datamunge::stats::RidgeOptions& x) {
    return std::format("RidgeOptions(lambda={}, n_lambda={}, cv_folds={}, max_iter={}, tol={}, seed={}, standardize={})", x.lambda, x.n_lambda, x.cv_folds, x.max_iter, x.tol, x.seed, x.standardize);
  }
};

/// @brief Prints a stats::LassoOptions (lasso options) as one line, `LassoOptions(field=value, ...)`.
template <>
struct std::formatter<datamunge::stats::LassoOptions> : datamunge::format_detail::text_formatter<datamunge::stats::LassoOptions> {
  /// @brief One-line field listing.
  /// @param x The value to print.
  /// @return `LassoOptions(...)` with every field; vectors are elided previews.
  static std::string text(const datamunge::stats::LassoOptions& x) {
    return std::format("LassoOptions(lambda={}, n_lambda={}, cv_folds={}, max_iter={}, tol={}, seed={}, standardize={})", x.lambda, x.n_lambda, x.cv_folds, x.max_iter, x.tol, x.seed, x.standardize);
  }
};

/// @brief Prints a stats::ExponentialSmoothingOptions (exponential-smoothing options) as one line, `ExponentialSmoothingOptions(field=value, ...)`.
template <>
struct std::formatter<datamunge::stats::ExponentialSmoothingOptions> : datamunge::format_detail::text_formatter<datamunge::stats::ExponentialSmoothingOptions> {
  /// @brief One-line field listing.
  /// @param x The value to print.
  /// @return `ExponentialSmoothingOptions(...)` with every field; vectors are elided previews.
  static std::string text(const datamunge::stats::ExponentialSmoothingOptions& x) {
    return std::format("ExponentialSmoothingOptions(trend={}, seasonal={}, seasonal_period={}, alpha={}, beta={}, gamma={}, phi={}, de_population_size={}, de_max_generations={}, seed={})", x.trend, x.seasonal, x.seasonal_period, x.alpha, x.beta, x.gamma, x.phi, x.de_population_size, x.de_max_generations, x.seed);
  }
};

/// @brief Prints a stats::FactorAnalysisOptions (factor-analysis options) as one line, `FactorAnalysisOptions(field=value, ...)`.
template <>
struct std::formatter<datamunge::stats::FactorAnalysisOptions> : datamunge::format_detail::text_formatter<datamunge::stats::FactorAnalysisOptions> {
  /// @brief One-line field listing.
  /// @param x The value to print.
  /// @return `FactorAnalysisOptions(...)` with every field; vectors are elided previews.
  static std::string text(const datamunge::stats::FactorAnalysisOptions& x) {
    namespace f = datamunge::format_detail::stats_fmt;
    return std::format("FactorAnalysisOptions(n_factors={}, standardize={}, rotation={}, max_iter={}, tol={})", x.n_factors, x.standardize, f::quoted(x.rotation), x.max_iter, x.tol);
  }
};

/// @brief Prints a stats::TermGroup (formula term group) as one line, `TermGroup(field=value, ...)`.
template <>
struct std::formatter<datamunge::stats::TermGroup> : datamunge::format_detail::text_formatter<datamunge::stats::TermGroup> {
  /// @brief One-line field listing.
  /// @param x The value to print.
  /// @return `TermGroup(...)` with every field; vectors are elided previews.
  static std::string text(const datamunge::stats::TermGroup& x) {
    namespace f = datamunge::format_detail::stats_fmt;
    return std::format("TermGroup(label={}, column_indices={})", f::quoted(x.label), f::vec(x.column_indices));
  }
};

/// @brief Prints a stats::GaussianProcessRegressionOptions (Gaussian-process regression options) as one line, `GaussianProcessRegressionOptions(field=value, ...)`.
template <>
struct std::formatter<datamunge::stats::GaussianProcessRegressionOptions> : datamunge::format_detail::text_formatter<datamunge::stats::GaussianProcessRegressionOptions> {
  /// @brief One-line field listing.
  /// @param x The value to print.
  /// @return `GaussianProcessRegressionOptions(...)` with every field; vectors are elided previews.
  static std::string text(const datamunge::stats::GaussianProcessRegressionOptions& x) {
    return std::format("GaussianProcessRegressionOptions(length_scale={}, noise_ratio={}, n_length_scale_grid={}, n_noise_grid={}, standardize={}, jitter_ratio={})", x.length_scale, x.noise_ratio, x.n_length_scale_grid, x.n_noise_grid, x.standardize, x.jitter_ratio);
  }
};

/// @brief Prints a stats::GaussianProcessRegressionPrediction (Gaussian-process prediction) as one line, `GaussianProcessRegressionPrediction(field=value, ...)`.
template <>
struct std::formatter<datamunge::stats::GaussianProcessRegressionPrediction> : datamunge::format_detail::text_formatter<datamunge::stats::GaussianProcessRegressionPrediction> {
  /// @brief One-line field listing.
  /// @param x The value to print.
  /// @return `GaussianProcessRegressionPrediction(...)` with every field; vectors are elided previews.
  static std::string text(const datamunge::stats::GaussianProcessRegressionPrediction& x) {
    namespace f = datamunge::format_detail::stats_fmt;
    return std::format("GaussianProcessRegressionPrediction(fit={}, se_fit={}, lower={}, upper={})", f::vec(x.fit), f::vec(x.se_fit), f::vec(x.lower), f::vec(x.upper));
  }
};

/// @brief Prints a stats::GBMClassifierOptions (gradient-boosting classifier options) as one line, `GBMClassifierOptions(field=value, ...)`.
template <>
struct std::formatter<datamunge::stats::GBMClassifierOptions> : datamunge::format_detail::text_formatter<datamunge::stats::GBMClassifierOptions> {
  /// @brief One-line field listing.
  /// @param x The value to print.
  /// @return `GBMClassifierOptions(...)` with every field; vectors are elided previews.
  static std::string text(const datamunge::stats::GBMClassifierOptions& x) {
    return std::format("GBMClassifierOptions(n_trees={}, learning_rate={}, max_depth={}, min_samples_split={}, min_samples_leaf={}, subsample={}, seed={})", x.n_trees, x.learning_rate, x.max_depth, x.min_samples_split, x.min_samples_leaf, x.subsample, x.seed);
  }
};

/// @brief Prints a stats::GBMClassifierPrediction (gradient-boosting classifier prediction) as one line, `GBMClassifierPrediction(field=value, ...)`.
template <>
struct std::formatter<datamunge::stats::GBMClassifierPrediction> : datamunge::format_detail::text_formatter<datamunge::stats::GBMClassifierPrediction> {
  /// @brief One-line field listing.
  /// @param x The value to print.
  /// @return `GBMClassifierPrediction(...)` with every field; vectors are elided previews.
  static std::string text(const datamunge::stats::GBMClassifierPrediction& x) {
    namespace f = datamunge::format_detail::stats_fmt;
    return std::format("GBMClassifierPrediction(class_label={}, probability={})", f::vec(x.class_label), f::rows(x.probability));
  }
};

/// @brief Prints a stats::GBMRegressorOptions (gradient-boosting regressor options) as one line, `GBMRegressorOptions(field=value, ...)`.
template <>
struct std::formatter<datamunge::stats::GBMRegressorOptions> : datamunge::format_detail::text_formatter<datamunge::stats::GBMRegressorOptions> {
  /// @brief One-line field listing.
  /// @param x The value to print.
  /// @return `GBMRegressorOptions(...)` with every field; vectors are elided previews.
  static std::string text(const datamunge::stats::GBMRegressorOptions& x) {
    return std::format("GBMRegressorOptions(n_trees={}, learning_rate={}, max_depth={}, min_samples_split={}, min_samples_leaf={}, subsample={}, seed={})", x.n_trees, x.learning_rate, x.max_depth, x.min_samples_split, x.min_samples_leaf, x.subsample, x.seed);
  }
};

/// @brief Prints a stats::GLMOptions (GLM options) as one line, `GLMOptions(field=value, ...)`.
template <>
struct std::formatter<datamunge::stats::GLMOptions> : datamunge::format_detail::text_formatter<datamunge::stats::GLMOptions> {
  /// @brief One-line field listing.
  /// @param x The value to print.
  /// @return `GLMOptions(...)` with every field; vectors are elided previews.
  static std::string text(const datamunge::stats::GLMOptions& x) {
    namespace f = datamunge::format_detail::stats_fmt;
    return std::format("GLMOptions(family={}, weights_column={}, max_iter={}, tol={})", x.family, f::optional_text(x.weights_column), x.max_iter, x.tol);
  }
};

/// @brief Prints a stats::GLMPrediction (GLM prediction) as one line, `GLMPrediction(field=value, ...)`.
template <>
struct std::formatter<datamunge::stats::GLMPrediction> : datamunge::format_detail::text_formatter<datamunge::stats::GLMPrediction> {
  /// @brief One-line field listing.
  /// @param x The value to print.
  /// @return `GLMPrediction(...)` with every field; vectors are elided previews.
  static std::string text(const datamunge::stats::GLMPrediction& x) {
    namespace f = datamunge::format_detail::stats_fmt;
    return std::format("GLMPrediction(fit={}, se_fit={}, lower={}, upper={})", f::vec(x.fit), f::vec(x.se_fit), f::vec(x.lower), f::vec(x.upper));
  }
};

/// @brief Prints a stats::GLM::Interval (GLM coefficient confidence interval) as one line, `Interval(field=value, ...)`.
template <>
struct std::formatter<datamunge::stats::GLM::Interval> : datamunge::format_detail::text_formatter<datamunge::stats::GLM::Interval> {
  /// @brief One-line field listing.
  /// @param x The value to print.
  /// @return `Interval(...)` with every field; vectors are elided previews.
  static std::string text(const datamunge::stats::GLM::Interval& x) {
    return std::format("Interval(lower={}, upper={})", x.lower, x.upper);
  }
};

/// @brief Prints a stats::GLMMOptions (GLMM options) as one line, `GLMMOptions(field=value, ...)`.
template <>
struct std::formatter<datamunge::stats::GLMMOptions> : datamunge::format_detail::text_formatter<datamunge::stats::GLMMOptions> {
  /// @brief One-line field listing.
  /// @param x The value to print.
  /// @return `GLMMOptions(...)` with every field; vectors are elided previews.
  static std::string text(const datamunge::stats::GLMMOptions& x) {
    return std::format("GLMMOptions(family={}, max_iterations={}, tol={}, de_population_size={}, de_max_generations={}, theta_bound={}, seed={}, fixed_random_intercept_sd={})", x.family, x.max_iterations, x.tol, x.de_population_size, x.de_max_generations, x.theta_bound, x.seed, x.fixed_random_intercept_sd ? std::format("{}",*x.fixed_random_intercept_sd) : "none");
  }
};

/// @brief Prints a stats::ICAOptions (ICA options) as one line, `ICAOptions(field=value, ...)`.
template <>
struct std::formatter<datamunge::stats::ICAOptions> : datamunge::format_detail::text_formatter<datamunge::stats::ICAOptions> {
  /// @brief One-line field listing.
  /// @param x The value to print.
  /// @return `ICAOptions(...)` with every field; vectors are elided previews.
  static std::string text(const datamunge::stats::ICAOptions& x) {
    namespace f = datamunge::format_detail::stats_fmt;
    return std::format("ICAOptions(n_components={}, contrast={}, max_iter={}, tol={}, seed={})", x.n_components, f::quoted(x.contrast), x.max_iter, x.tol, x.seed);
  }
};

/// @brief Prints a stats::ImageExplanationOptions (image-explainer options) as one line, `ImageExplanationOptions(field=value, ...)`.
template <>
struct std::formatter<datamunge::stats::ImageExplanationOptions> : datamunge::format_detail::text_formatter<datamunge::stats::ImageExplanationOptions> {
  /// @brief One-line field listing.
  /// @param x The value to print.
  /// @return `ImageExplanationOptions(...)` with every field; vectors are elided previews.
  static std::string text(const datamunge::stats::ImageExplanationOptions& x) {
    return std::format("ImageExplanationOptions(patch_rows={}, patch_cols={}, baseline={}, n_samples={}, lime_kernel_width={}, lime_l2={}, seed={})", x.patch_rows, x.patch_cols, x.baseline, x.n_samples, x.lime_kernel_width, x.lime_l2, x.seed);
  }
};

/// @brief Prints a stats::INLAMixedModelOptions (INLA mixed-model options) as one line, `INLAMixedModelOptions(field=value, ...)`.
template <>
struct std::formatter<datamunge::stats::INLAMixedModelOptions> : datamunge::format_detail::text_formatter<datamunge::stats::INLAMixedModelOptions> {
  /// @brief One-line field listing.
  /// @param x The value to print.
  /// @return `INLAMixedModelOptions(...)` with every field; vectors are elided previews.
  static std::string text(const datamunge::stats::INLAMixedModelOptions& x) {
    namespace f = datamunge::format_detail::stats_fmt;
    return std::format("INLAMixedModelOptions(family={}, strategy={}, fixed_effect_prior_sd={}, lambda_diag_epsilon={}, lambda_bound={}, log_sigma_search_radius={}, grid_points_per_dim={}, grid_span={}, mode_population_size={}, mode_max_generations={}, seed={})", x.family, f::strategy_text(x.strategy), x.fixed_effect_prior_sd, x.lambda_diag_epsilon, x.lambda_bound, x.log_sigma_search_radius, x.grid_points_per_dim, x.grid_span, x.mode_population_size, x.mode_max_generations, x.seed);
  }
};

/// @brief Prints a stats::IsomapOptions (Isomap options) as one line, `IsomapOptions(field=value, ...)`.
template <>
struct std::formatter<datamunge::stats::IsomapOptions> : datamunge::format_detail::text_formatter<datamunge::stats::IsomapOptions> {
  /// @brief One-line field listing.
  /// @param x The value to print.
  /// @return `IsomapOptions(...)` with every field; vectors are elided previews.
  static std::string text(const datamunge::stats::IsomapOptions& x) {
    return std::format("IsomapOptions(n_components={}, n_neighbors={}, metric={})", x.n_components, x.n_neighbors, x.metric);
  }
};

/// @brief Prints a stats::JackknifeResult (jackknife result) as one line, `JackknifeResult(field=value, ...)`.
template <>
struct std::formatter<datamunge::stats::JackknifeResult> : datamunge::format_detail::text_formatter<datamunge::stats::JackknifeResult> {
  /// @brief One-line field listing.
  /// @param x The value to print.
  /// @return `JackknifeResult(...)` with every field; vectors are elided previews.
  static std::string text(const datamunge::stats::JackknifeResult& x) {
    namespace f = datamunge::format_detail::stats_fmt;
    return std::format("JackknifeResult(estimate={}, bias={}, bias_corrected={}, standard_error={}, leave_one_out={}, pseudo_values={})", x.estimate, x.bias, x.bias_corrected, x.standard_error, f::vec(x.leave_one_out), f::vec(x.pseudo_values));
  }
};

/// @brief Prints a stats::DeleteDJackknifeOptions (delete-d jackknife options) as one line, `DeleteDJackknifeOptions(field=value, ...)`.
template <>
struct std::formatter<datamunge::stats::DeleteDJackknifeOptions> : datamunge::format_detail::text_formatter<datamunge::stats::DeleteDJackknifeOptions> {
  /// @brief One-line field listing.
  /// @param x The value to print.
  /// @return `DeleteDJackknifeOptions(...)` with every field; vectors are elided previews.
  static std::string text(const datamunge::stats::DeleteDJackknifeOptions& x) {
    return std::format("DeleteDJackknifeOptions(d={}, max_subsets={}, seed={})", x.d, x.max_subsets, x.seed);
  }
};

/// @brief Prints a stats::DeleteDJackknifeResult (delete-d jackknife result) as one line, `DeleteDJackknifeResult(field=value, ...)`.
template <>
struct std::formatter<datamunge::stats::DeleteDJackknifeResult> : datamunge::format_detail::text_formatter<datamunge::stats::DeleteDJackknifeResult> {
  /// @brief One-line field listing.
  /// @param x The value to print.
  /// @return `DeleteDJackknifeResult(...)` with every field; vectors are elided previews.
  static std::string text(const datamunge::stats::DeleteDJackknifeResult& x) {
    return std::format("DeleteDJackknifeResult(estimate={}, standard_error={}, d={}, num_subsets={}, complete={})", x.estimate, x.standard_error, x.d, x.num_subsets, x.complete);
  }
};

/// @brief Prints a stats::KernelPCAOptions (kernel-PCA options) as one line, `KernelPCAOptions(field=value, ...)`.
template <>
struct std::formatter<datamunge::stats::KernelPCAOptions> : datamunge::format_detail::text_formatter<datamunge::stats::KernelPCAOptions> {
  /// @brief One-line field listing.
  /// @param x The value to print.
  /// @return `KernelPCAOptions(...)` with every field; vectors are elided previews.
  static std::string text(const datamunge::stats::KernelPCAOptions& x) {
    namespace f = datamunge::format_detail::stats_fmt;
    return std::format("KernelPCAOptions(n_components={}, kernel={}, gamma={}, degree={}, coef0={})", x.n_components, f::quoted(x.kernel), x.gamma, x.degree, x.coef0);
  }
};

/// @brief Prints a stats::KernelRegressionOptions (kernel-regression options) as one line, `KernelRegressionOptions(field=value, ...)`.
template <>
struct std::formatter<datamunge::stats::KernelRegressionOptions> : datamunge::format_detail::text_formatter<datamunge::stats::KernelRegressionOptions> {
  /// @brief One-line field listing.
  /// @param x The value to print.
  /// @return `KernelRegressionOptions(...)` with every field; vectors are elided previews.
  static std::string text(const datamunge::stats::KernelRegressionOptions& x) {
    namespace f = datamunge::format_detail::stats_fmt;
    return std::format("KernelRegressionOptions(kernel={}, bandwidth={}, n_bandwidth={}, standardize={}, bandwidth_candidates={})", x.kernel, x.bandwidth, x.n_bandwidth, x.standardize, f::vec(x.bandwidth_candidates));
  }
};

/// @brief Prints a stats::KernelRidgeRegressionOptions (kernel ridge regression options) as one line, `KernelRidgeRegressionOptions(field=value, ...)`.
template <>
struct std::formatter<datamunge::stats::KernelRidgeRegressionOptions> : datamunge::format_detail::text_formatter<datamunge::stats::KernelRidgeRegressionOptions> {
  /// @brief One-line field listing.
  /// @param x The value to print.
  /// @return `KernelRidgeRegressionOptions(...)` with every field; vectors are elided previews.
  static std::string text(const datamunge::stats::KernelRidgeRegressionOptions& x) {
    namespace f = datamunge::format_detail::stats_fmt;
    return std::format("KernelRidgeRegressionOptions(kernel={}, lambda={}, gamma={}, degree={}, coef0={}, n_lambda_grid={}, n_gamma_grid={}, standardize={})", f::quoted(x.kernel), x.lambda, x.gamma, x.degree, x.coef0, x.n_lambda_grid, x.n_gamma_grid, x.standardize);
  }
};

/// @brief Prints a stats::KMeansOptions (k-means options) as one line, `KMeansOptions(field=value, ...)`.
template <>
struct std::formatter<datamunge::stats::KMeansOptions> : datamunge::format_detail::text_formatter<datamunge::stats::KMeansOptions> {
  /// @brief One-line field listing.
  /// @param x The value to print.
  /// @return `KMeansOptions(...)` with every field; vectors are elided previews.
  static std::string text(const datamunge::stats::KMeansOptions& x) {
    return std::format("KMeansOptions(n_clusters={}, max_iterations={}, n_init={}, tolerance={}, seed={})", x.n_clusters, x.max_iterations, x.n_init, x.tolerance, x.seed);
  }
};

/// @brief Prints a stats::KNNClassifierOptions (k-NN classifier options) as one line, `KNNClassifierOptions(field=value, ...)`.
template <>
struct std::formatter<datamunge::stats::KNNClassifierOptions> : datamunge::format_detail::text_formatter<datamunge::stats::KNNClassifierOptions> {
  /// @brief One-line field listing.
  /// @param x The value to print.
  /// @return `KNNClassifierOptions(...)` with every field; vectors are elided previews.
  static std::string text(const datamunge::stats::KNNClassifierOptions& x) {
    return std::format("KNNClassifierOptions(k={}, metric={}, weighted={}, standardize={})", x.k, x.metric, x.weighted, x.standardize);
  }
};

/// @brief Prints a stats::KNNClassifierPrediction (k-NN classifier prediction) as one line, `KNNClassifierPrediction(field=value, ...)`.
template <>
struct std::formatter<datamunge::stats::KNNClassifierPrediction> : datamunge::format_detail::text_formatter<datamunge::stats::KNNClassifierPrediction> {
  /// @brief One-line field listing.
  /// @param x The value to print.
  /// @return `KNNClassifierPrediction(...)` with every field; vectors are elided previews.
  static std::string text(const datamunge::stats::KNNClassifierPrediction& x) {
    namespace f = datamunge::format_detail::stats_fmt;
    return std::format("KNNClassifierPrediction(class_label={}, vote_share={})", f::vec(x.class_label), f::rows(x.vote_share));
  }
};

/// @brief Prints a stats::KNNNeighbors (the neighbours of query points) as one line, `KNNNeighbors(field=value, ...)`.
template <>
struct std::formatter<datamunge::stats::KNNNeighbors> : datamunge::format_detail::text_formatter<datamunge::stats::KNNNeighbors> {
  /// @brief One-line field listing.
  /// @param x The value to print.
  /// @return `KNNNeighbors(...)` with every field; vectors are elided previews.
  static std::string text(const datamunge::stats::KNNNeighbors& x) {
    namespace f = datamunge::format_detail::stats_fmt;
    return std::format("KNNNeighbors(index={}, distance={})", datamunge::format_detail::preview(x.index, 3), f::rows(x.distance));
  }
};

/// @brief Prints a stats::KNNRegressorOptions (k-NN regressor options) as one line, `KNNRegressorOptions(field=value, ...)`.
template <>
struct std::formatter<datamunge::stats::KNNRegressorOptions> : datamunge::format_detail::text_formatter<datamunge::stats::KNNRegressorOptions> {
  /// @brief One-line field listing.
  /// @param x The value to print.
  /// @return `KNNRegressorOptions(...)` with every field; vectors are elided previews.
  static std::string text(const datamunge::stats::KNNRegressorOptions& x) {
    return std::format("KNNRegressorOptions(k={}, metric={}, weighted={}, standardize={})", x.k, x.metric, x.weighted, x.standardize);
  }
};

/// @brief Prints a stats::LaplacianEigenmapsOptions (Laplacian-eigenmaps options) as one line, `LaplacianEigenmapsOptions(field=value, ...)`.
template <>
struct std::formatter<datamunge::stats::LaplacianEigenmapsOptions> : datamunge::format_detail::text_formatter<datamunge::stats::LaplacianEigenmapsOptions> {
  /// @brief One-line field listing.
  /// @param x The value to print.
  /// @return `LaplacianEigenmapsOptions(...)` with every field; vectors are elided previews.
  static std::string text(const datamunge::stats::LaplacianEigenmapsOptions& x) {
    return std::format("LaplacianEigenmapsOptions(n_components={}, n_neighbors={}, heat_kernel_t={})", x.n_components, x.n_neighbors, x.heat_kernel_t);
  }
};

/// @brief Prints a stats::LDAOptions (LDA options) as one line, `LDAOptions(field=value, ...)`.
template <>
struct std::formatter<datamunge::stats::LDAOptions> : datamunge::format_detail::text_formatter<datamunge::stats::LDAOptions> {
  /// @brief One-line field listing.
  /// @param x The value to print.
  /// @return `LDAOptions(...)` with every field; vectors are elided previews.
  static std::string text(const datamunge::stats::LDAOptions& x) {
    namespace f = datamunge::format_detail::stats_fmt;
    return std::format("LDAOptions(priors={})", f::optional_vec(x.priors));
  }
};

/// @brief Prints a stats::LDAPrediction (LDA prediction) as one line, `LDAPrediction(field=value, ...)`.
template <>
struct std::formatter<datamunge::stats::LDAPrediction> : datamunge::format_detail::text_formatter<datamunge::stats::LDAPrediction> {
  /// @brief One-line field listing.
  /// @param x The value to print.
  /// @return `LDAPrediction(...)` with every field; vectors are elided previews.
  static std::string text(const datamunge::stats::LDAPrediction& x) {
    namespace f = datamunge::format_detail::stats_fmt;
    return std::format("LDAPrediction(class_label={}, posterior={}, discriminants={})", f::vec(x.class_label), f::rows(x.posterior), f::rows(x.discriminants));
  }
};

/// @brief Prints a stats::LLEOptions (LLE options) as one line, `LLEOptions(field=value, ...)`.
template <>
struct std::formatter<datamunge::stats::LLEOptions> : datamunge::format_detail::text_formatter<datamunge::stats::LLEOptions> {
  /// @brief One-line field listing.
  /// @param x The value to print.
  /// @return `LLEOptions(...)` with every field; vectors are elided previews.
  static std::string text(const datamunge::stats::LLEOptions& x) {
    return std::format("LLEOptions(n_components={}, n_neighbors={}, regularization={}, metric={})", x.n_components, x.n_neighbors, x.regularization, x.metric);
  }
};

/// @brief Prints a stats::LmOptions (linear-model options) as one line, `LmOptions(field=value, ...)`.
template <>
struct std::formatter<datamunge::stats::LmOptions> : datamunge::format_detail::text_formatter<datamunge::stats::LmOptions> {
  /// @brief One-line field listing.
  /// @param x The value to print.
  /// @return `LmOptions(...)` with every field; vectors are elided previews.
  static std::string text(const datamunge::stats::LmOptions& x) {
    namespace f = datamunge::format_detail::stats_fmt;
    return std::format("LmOptions(weights_column={})", f::optional_text(x.weights_column));
  }
};

/// @brief Prints a stats::LmPrediction (linear-model prediction) as one line, `LmPrediction(field=value, ...)`.
template <>
struct std::formatter<datamunge::stats::LmPrediction> : datamunge::format_detail::text_formatter<datamunge::stats::LmPrediction> {
  /// @brief One-line field listing.
  /// @param x The value to print.
  /// @return `LmPrediction(...)` with every field; vectors are elided previews.
  static std::string text(const datamunge::stats::LmPrediction& x) {
    namespace f = datamunge::format_detail::stats_fmt;
    return std::format("LmPrediction(fit={}, se_fit={}, lower={}, upper={})", f::vec(x.fit), f::vec(x.se_fit), f::vec(x.lower), f::vec(x.upper));
  }
};

/// @brief Prints a stats::AnovaRow (ANOVA table row) as one line, `AnovaRow(field=value, ...)`.
template <>
struct std::formatter<datamunge::stats::AnovaRow> : datamunge::format_detail::text_formatter<datamunge::stats::AnovaRow> {
  /// @brief One-line field listing.
  /// @param x The value to print.
  /// @return `AnovaRow(...)` with every field; vectors are elided previews.
  static std::string text(const datamunge::stats::AnovaRow& x) {
    namespace f = datamunge::format_detail::stats_fmt;
    return std::format("AnovaRow(term={}, degrees_of_freedom={}, sum_sq={}, mean_sq={}, f_value={}, p_value={})", f::quoted(x.term), x.degrees_of_freedom, x.sum_sq, x.mean_sq, x.f_value, x.p_value);
  }
};

/// @brief Prints a stats::LM::Interval (linear-model coefficient confidence interval) as one line, `Interval(field=value, ...)`.
template <>
struct std::formatter<datamunge::stats::LM::Interval> : datamunge::format_detail::text_formatter<datamunge::stats::LM::Interval> {
  /// @brief One-line field listing.
  /// @param x The value to print.
  /// @return `Interval(...)` with every field; vectors are elided previews.
  static std::string text(const datamunge::stats::LM::Interval& x) {
    return std::format("Interval(lower={}, upper={})", x.lower, x.upper);
  }
};

/// @brief Prints a stats::NestedAnova (nested-model F-test) as one line, `NestedAnova(field=value, ...)`.
template <>
struct std::formatter<datamunge::stats::NestedAnova> : datamunge::format_detail::text_formatter<datamunge::stats::NestedAnova> {
  /// @brief One-line field listing.
  /// @param x The value to print.
  /// @return `NestedAnova(...)` with every field.
  static std::string text(const datamunge::stats::NestedAnova& x) {
    return std::format("NestedAnova(reduced_df={}, full_df={}, reduced_rss={}, full_rss={}, df={}, sum_sq={}, f_value={}, p_value={})", x.reduced_df, x.full_df, x.reduced_rss, x.full_rss, x.df, x.sum_sq, x.f_value, x.p_value);
  }
};

/// @brief Prints a stats::LMMOptions (LMM options) as one line, `LMMOptions(field=value, ...)`.
template <>
struct std::formatter<datamunge::stats::LMMOptions> : datamunge::format_detail::text_formatter<datamunge::stats::LMMOptions> {
  /// @brief One-line field listing.
  /// @param x The value to print.
  /// @return `LMMOptions(...)` with every field; vectors are elided previews.
  static std::string text(const datamunge::stats::LMMOptions& x) {
    return std::format("LMMOptions(reml={}, de_population_size={}, de_max_generations={}, theta_bound={}, seed={})", x.reml, x.de_population_size, x.de_max_generations, x.theta_bound, x.seed);
  }
};

/// @brief Prints a stats::MDSOptions (MDS options) as one line, `MDSOptions(field=value, ...)`.
template <>
struct std::formatter<datamunge::stats::MDSOptions> : datamunge::format_detail::text_formatter<datamunge::stats::MDSOptions> {
  /// @brief One-line field listing.
  /// @param x The value to print.
  /// @return `MDSOptions(...)` with every field; vectors are elided previews.
  static std::string text(const datamunge::stats::MDSOptions& x) {
    return std::format("MDSOptions(n_components={}, metric={})", x.n_components, x.metric);
  }
};

/// @brief Prints a stats::MLPClassifierOptions (MLP classifier options) as one line, `MLPClassifierOptions(field=value, ...)`.
template <>
struct std::formatter<datamunge::stats::MLPClassifierOptions> : datamunge::format_detail::text_formatter<datamunge::stats::MLPClassifierOptions> {
  /// @brief One-line field listing.
  /// @param x The value to print.
  /// @return `MLPClassifierOptions(...)` with every field; vectors are elided previews.
  static std::string text(const datamunge::stats::MLPClassifierOptions& x) {
    namespace f = datamunge::format_detail::stats_fmt;
    return std::format("MLPClassifierOptions(hidden_layer_sizes={}, activation={}, learning_rate={}, max_epochs={}, batch_size={}, l2={}, seed={}, standardize={})", f::vec(x.hidden_layer_sizes), f::quoted(x.activation), x.learning_rate, x.max_epochs, x.batch_size, x.l2, x.seed, x.standardize);
  }
};

/// @brief Prints a stats::MLPClassifierPrediction (MLP classifier prediction) as one line, `MLPClassifierPrediction(field=value, ...)`.
template <>
struct std::formatter<datamunge::stats::MLPClassifierPrediction> : datamunge::format_detail::text_formatter<datamunge::stats::MLPClassifierPrediction> {
  /// @brief One-line field listing.
  /// @param x The value to print.
  /// @return `MLPClassifierPrediction(...)` with every field; vectors are elided previews.
  static std::string text(const datamunge::stats::MLPClassifierPrediction& x) {
    namespace f = datamunge::format_detail::stats_fmt;
    return std::format("MLPClassifierPrediction(class_label={}, probabilities={})", f::vec(x.class_label), f::rows(x.probabilities));
  }
};

/// @brief Prints a stats::NaiveBayesClassifierOptions (naive Bayes options) as one line, `NaiveBayesClassifierOptions(field=value, ...)`.
template <>
struct std::formatter<datamunge::stats::NaiveBayesClassifierOptions> : datamunge::format_detail::text_formatter<datamunge::stats::NaiveBayesClassifierOptions> {
  /// @brief One-line field listing.
  /// @param x The value to print.
  /// @return `NaiveBayesClassifierOptions(...)` with every field; vectors are elided previews.
  static std::string text(const datamunge::stats::NaiveBayesClassifierOptions& x) {
    namespace f = datamunge::format_detail::stats_fmt;
    return std::format("NaiveBayesClassifierOptions(laplace_smoothing={}, var_smoothing={}, priors={})", x.laplace_smoothing, x.var_smoothing, f::vec(x.priors));
  }
};

/// @brief Prints a stats::NaiveBayesClassifierPrediction (naive Bayes prediction) as one line, `NaiveBayesClassifierPrediction(field=value, ...)`.
template <>
struct std::formatter<datamunge::stats::NaiveBayesClassifierPrediction> : datamunge::format_detail::text_formatter<datamunge::stats::NaiveBayesClassifierPrediction> {
  /// @brief One-line field listing.
  /// @param x The value to print.
  /// @return `NaiveBayesClassifierPrediction(...)` with every field; vectors are elided previews.
  static std::string text(const datamunge::stats::NaiveBayesClassifierPrediction& x) {
    namespace f = datamunge::format_detail::stats_fmt;
    return std::format("NaiveBayesClassifierPrediction(class_label={}, probability={})", f::vec(x.class_label), f::rows(x.probability));
  }
};

/// @brief Prints a stats::PCAOptions (PCA options) as one line, `PCAOptions(field=value, ...)`.
template <>
struct std::formatter<datamunge::stats::PCAOptions> : datamunge::format_detail::text_formatter<datamunge::stats::PCAOptions> {
  /// @brief One-line field listing.
  /// @param x The value to print.
  /// @return `PCAOptions(...)` with every field; vectors are elided previews.
  static std::string text(const datamunge::stats::PCAOptions& x) {
    return std::format("PCAOptions(center={}, scale={})", x.center, x.scale);
  }
};

/// @brief Prints a stats::PermutationTestOptions (permutation-test options) as one line, `PermutationTestOptions(field=value, ...)`.
template <>
struct std::formatter<datamunge::stats::PermutationTestOptions> : datamunge::format_detail::text_formatter<datamunge::stats::PermutationTestOptions> {
  /// @brief One-line field listing.
  /// @param x The value to print.
  /// @return `PermutationTestOptions(...)` with every field; vectors are elided previews.
  static std::string text(const datamunge::stats::PermutationTestOptions& x) {
    return std::format("PermutationTestOptions(num_permutations={}, alternative={}, seed={})", x.num_permutations, x.alternative, x.seed);
  }
};

/// @brief Prints a stats::PermutationTestResult (permutation-test result) as one line, `PermutationTestResult(field=value, ...)`.
template <>
struct std::formatter<datamunge::stats::PermutationTestResult> : datamunge::format_detail::text_formatter<datamunge::stats::PermutationTestResult> {
  /// @brief One-line field listing.
  /// @param x The value to print.
  /// @return `PermutationTestResult(...)` with every field; vectors are elided previews.
  static std::string text(const datamunge::stats::PermutationTestResult& x) {
    namespace f = datamunge::format_detail::stats_fmt;
    return std::format("PermutationTestResult(observed={}, p_value={}, null_mean={}, null_sd={}, num_permutations={}, null_distribution={})", x.observed, x.p_value, x.null_mean, x.null_sd, x.num_permutations, f::vec(x.null_distribution));
  }
};

/// @brief Prints a stats::QuantileRegressionOptions (quantile-regression options) as one line, `QuantileRegressionOptions(field=value, ...)`.
template <>
struct std::formatter<datamunge::stats::QuantileRegressionOptions> : datamunge::format_detail::text_formatter<datamunge::stats::QuantileRegressionOptions> {
  /// @brief One-line field listing.
  /// @param x The value to print.
  /// @return `QuantileRegressionOptions(...)` with every field; vectors are elided previews.
  static std::string text(const datamunge::stats::QuantileRegressionOptions& x) {
    namespace f = datamunge::format_detail::stats_fmt;
    return std::format("QuantileRegressionOptions(quantile={}, rho={}, max_iterations={}, absolute_tolerance={}, relative_tolerance={}, adaptive_rho={}, weights_column={}, method={})", x.quantile, x.rho, x.max_iterations, x.absolute_tolerance, x.relative_tolerance, x.adaptive_rho, f::optional_text(x.weights_column), x.method);
  }
};

/// @brief Prints a stats::QuantileCoefficientTable (quantile-regression coefficient table) as one line, `QuantileCoefficientTable(field=value, ...)`.
template <>
struct std::formatter<datamunge::stats::QuantileCoefficientTable> : datamunge::format_detail::text_formatter<datamunge::stats::QuantileCoefficientTable> {
  /// @brief One-line field listing.
  /// @param x The value to print.
  /// @return `QuantileCoefficientTable(...)` with every field; vectors are elided previews.
  static std::string text(const datamunge::stats::QuantileCoefficientTable& x) {
    namespace f = datamunge::format_detail::stats_fmt;
    return std::format("QuantileCoefficientTable(names={}, estimate={}, std_error={}, t_value={}, p_value={}, lower={}, upper={}, method={}, level={}, bandwidth={}, replications={}, degrees_of_freedom={})", f::vec(x.names), f::vec(x.estimate), f::vec(x.std_error), f::vec(x.t_value), f::vec(x.p_value), f::vec(x.lower), f::vec(x.upper), x.method, x.level, x.bandwidth, x.replications, x.degrees_of_freedom);
  }
};

/// @brief Prints a stats::QuantileProcess (coefficients over a grid of quantile levels) as one line, `QuantileProcess(field=value, ...)`.
template <>
struct std::formatter<datamunge::stats::QuantileProcess> : datamunge::format_detail::text_formatter<datamunge::stats::QuantileProcess> {
  /// @brief One-line field listing.
  /// @param x The value to print.
  /// @return `QuantileProcess(...)` with the levels, names and sizes; the matrices are summarised by shape.
  static std::string text(const datamunge::stats::QuantileProcess& x) {
    namespace f = datamunge::format_detail::stats_fmt;
    return std::format("QuantileProcess(quantiles={}, names={}, coefficients=[{} x {}], level={})", f::vec(x.quantiles), f::vec(x.names), x.coefficients.size(), x.names.size(), x.level);
  }
};

/// @brief Prints a stats::RandomForestClassifierOptions (random-forest classifier options) as one line, `RandomForestClassifierOptions(field=value, ...)`.
template <>
struct std::formatter<datamunge::stats::RandomForestClassifierOptions> : datamunge::format_detail::text_formatter<datamunge::stats::RandomForestClassifierOptions> {
  /// @brief One-line field listing.
  /// @param x The value to print.
  /// @return `RandomForestClassifierOptions(...)` with every field; vectors are elided previews.
  static std::string text(const datamunge::stats::RandomForestClassifierOptions& x) {
    return std::format("RandomForestClassifierOptions(n_trees={}, max_depth={}, min_samples_split={}, min_samples_leaf={}, max_features={}, criterion={}, bootstrap={}, sample_fraction={}, seed={})", x.n_trees, x.max_depth, x.min_samples_split, x.min_samples_leaf, x.max_features, x.criterion, x.bootstrap, x.sample_fraction, x.seed);
  }
};

/// @brief Prints a stats::RandomForestClassifierPrediction (random-forest classifier prediction) as one line, `RandomForestClassifierPrediction(field=value, ...)`.
template <>
struct std::formatter<datamunge::stats::RandomForestClassifierPrediction> : datamunge::format_detail::text_formatter<datamunge::stats::RandomForestClassifierPrediction> {
  /// @brief One-line field listing.
  /// @param x The value to print.
  /// @return `RandomForestClassifierPrediction(...)` with every field; vectors are elided previews.
  static std::string text(const datamunge::stats::RandomForestClassifierPrediction& x) {
    namespace f = datamunge::format_detail::stats_fmt;
    return std::format("RandomForestClassifierPrediction(class_label={}, vote_share={})", f::vec(x.class_label), f::rows(x.vote_share));
  }
};

/// @brief Prints a stats::RandomForestStrength (Breiman's out-of-bag strength and correlation) as one line, `RandomForestStrength(field=value, ...)`.
template <>
struct std::formatter<datamunge::stats::RandomForestStrength> : datamunge::format_detail::text_formatter<datamunge::stats::RandomForestStrength> {
  /// @brief One-line field listing.
  /// @param x The value to print.
  /// @return `RandomForestStrength(...)` with every field.
  static std::string text(const datamunge::stats::RandomForestStrength& x) {
    return std::format("RandomForestStrength(strength={}, margin_variance={}, mean_tree_sd={}, correlation={}, error_bound={}, rows={})", x.strength, x.margin_variance, x.mean_tree_sd, x.correlation, x.error_bound, x.rows);
  }
};

/// @brief Prints a stats::RandomForestPermutationImportance (out-of-bag permutation importance) as one line, `RandomForestPermutationImportance(field=value, ...)`.
template <>
struct std::formatter<datamunge::stats::RandomForestPermutationImportance> : datamunge::format_detail::text_formatter<datamunge::stats::RandomForestPermutationImportance> {
  /// @brief One-line field listing.
  /// @param x The value to print.
  /// @return `RandomForestPermutationImportance(...)` with every field; vectors are elided previews.
  static std::string text(const datamunge::stats::RandomForestPermutationImportance& x) {
    namespace f = datamunge::format_detail::stats_fmt;
    return std::format("RandomForestPermutationImportance(predictor_names={}, importance={}, standard_error={})", f::vec(x.predictor_names), f::vec(x.importance), f::vec(x.standard_error));
  }
};

/// @brief Prints a stats::RandomForestRegressorOptions (random-forest regressor options) as one line, `RandomForestRegressorOptions(field=value, ...)`.
template <>
struct std::formatter<datamunge::stats::RandomForestRegressorOptions> : datamunge::format_detail::text_formatter<datamunge::stats::RandomForestRegressorOptions> {
  /// @brief One-line field listing.
  /// @param x The value to print.
  /// @return `RandomForestRegressorOptions(...)` with every field; vectors are elided previews.
  static std::string text(const datamunge::stats::RandomForestRegressorOptions& x) {
    return std::format("RandomForestRegressorOptions(n_trees={}, max_depth={}, min_samples_split={}, min_samples_leaf={}, max_features={}, bootstrap={}, sample_fraction={}, seed={})", x.n_trees, x.max_depth, x.min_samples_split, x.min_samples_leaf, x.max_features, x.bootstrap, x.sample_fraction, x.seed);
  }
};

/// @brief Prints a stats::SammonMappingOptions (Sammon-mapping options) as one line, `SammonMappingOptions(field=value, ...)`.
template <>
struct std::formatter<datamunge::stats::SammonMappingOptions> : datamunge::format_detail::text_formatter<datamunge::stats::SammonMappingOptions> {
  /// @brief One-line field listing.
  /// @param x The value to print.
  /// @return `SammonMappingOptions(...)` with every field; vectors are elided previews.
  static std::string text(const datamunge::stats::SammonMappingOptions& x) {
    return std::format("SammonMappingOptions(n_components={}, learning_rate={}, max_iterations={}, tolerance={}, metric={}, seed={})", x.n_components, x.learning_rate, x.max_iterations, x.tolerance, x.metric, x.seed);
  }
};

/// @brief Prints a stats::SVMOptions (SVM options) as one line, `SVMOptions(field=value, ...)`.
template <>
struct std::formatter<datamunge::stats::SVMOptions> : datamunge::format_detail::text_formatter<datamunge::stats::SVMOptions> {
  /// @brief One-line field listing.
  /// @param x The value to print.
  /// @return `SVMOptions(...)` with every field; vectors are elided previews.
  static std::string text(const datamunge::stats::SVMOptions& x) {
    return std::format("SVMOptions(kernel={}, cost={}, gamma={}, coef0={}, degree={}, tolerance={}, max_passes={}, scale={})", x.kernel, x.cost, x.gamma, x.coef0, x.degree, x.tolerance, x.max_passes, x.scale);
  }
};

/// @brief Prints a stats::SVMPrediction (SVM prediction) as one line, `SVMPrediction(field=value, ...)`.
template <>
struct std::formatter<datamunge::stats::SVMPrediction> : datamunge::format_detail::text_formatter<datamunge::stats::SVMPrediction> {
  /// @brief One-line field listing.
  /// @param x The value to print.
  /// @return `SVMPrediction(...)` with every field; vectors are elided previews.
  static std::string text(const datamunge::stats::SVMPrediction& x) {
    namespace f = datamunge::format_detail::stats_fmt;
    return std::format("SVMPrediction(class_label={}, votes={})", f::vec(x.class_label), f::rows(x.votes));
  }
};

/// @brief Prints a stats::TSNEOptions (t-SNE options) as one line, `TSNEOptions(field=value, ...)`.
template <>
struct std::formatter<datamunge::stats::TSNEOptions> : datamunge::format_detail::text_formatter<datamunge::stats::TSNEOptions> {
  /// @brief One-line field listing.
  /// @param x The value to print.
  /// @return `TSNEOptions(...)` with every field; vectors are elided previews.
  static std::string text(const datamunge::stats::TSNEOptions& x) {
    return std::format("TSNEOptions(n_components={}, perplexity={}, max_iterations={}, learning_rate={}, early_exaggeration={}, early_exaggeration_iterations={}, initial_momentum={}, final_momentum={}, momentum_switch_iteration={}, metric={}, seed={})", x.n_components, x.perplexity, x.max_iterations, x.learning_rate, x.early_exaggeration, x.early_exaggeration_iterations, x.initial_momentum, x.final_momentum, x.momentum_switch_iteration, x.metric, x.seed);
  }
};

/// @brief Prints a stats::UMAPOptions (UMAP options) as one line, `UMAPOptions(field=value, ...)`.
template <>
struct std::formatter<datamunge::stats::UMAPOptions> : datamunge::format_detail::text_formatter<datamunge::stats::UMAPOptions> {
  /// @brief One-line field listing.
  /// @param x The value to print.
  /// @return `UMAPOptions(...)` with every field; vectors are elided previews.
  static std::string text(const datamunge::stats::UMAPOptions& x) {
    return std::format("UMAPOptions(n_components={}, n_neighbors={}, min_dist={}, max_iterations={}, learning_rate={}, negative_sample_rate={}, metric={}, seed={})", x.n_components, x.n_neighbors, x.min_dist, x.max_iterations, x.learning_rate, x.negative_sample_rate, x.metric, x.seed);
  }
};

/// @brief Prints a stats::UStatisticOptions (U-statistic options) as one line, `UStatisticOptions(field=value, ...)`.
template <>
struct std::formatter<datamunge::stats::UStatisticOptions> : datamunge::format_detail::text_formatter<datamunge::stats::UStatisticOptions> {
  /// @brief One-line field listing.
  /// @param x The value to print.
  /// @return `UStatisticOptions(...)` with every field; vectors are elided previews.
  static std::string text(const datamunge::stats::UStatisticOptions& x) {
    return std::format("UStatisticOptions(max_terms={}, seed={})", x.max_terms, x.seed);
  }
};

/// @brief Prints a stats::UStatisticResult (U-statistic result) as one line, `UStatisticResult(field=value, ...)`.
template <>
struct std::formatter<datamunge::stats::UStatisticResult> : datamunge::format_detail::text_formatter<datamunge::stats::UStatisticResult> {
  /// @brief One-line field listing.
  /// @param x The value to print.
  /// @return `UStatisticResult(...)` with every field; vectors are elided previews.
  static std::string text(const datamunge::stats::UStatisticResult& x) {
    return std::format("UStatisticResult(estimate={}, zeta1={}, variance={}, standard_error={}, degree={}, num_terms={}, complete={})", x.estimate, x.zeta1, x.variance, x.standard_error, x.degree, x.num_terms, x.complete);
  }
};

/// @brief Prints a stats::VStatisticOptions (V-statistic options) as one line, `VStatisticOptions(field=value, ...)`.
template <>
struct std::formatter<datamunge::stats::VStatisticOptions> : datamunge::format_detail::text_formatter<datamunge::stats::VStatisticOptions> {
  /// @brief One-line field listing.
  /// @param x The value to print.
  /// @return `VStatisticOptions(...)` with every field; vectors are elided previews.
  static std::string text(const datamunge::stats::VStatisticOptions& x) {
    return std::format("VStatisticOptions(max_terms={}, seed={})", x.max_terms, x.seed);
  }
};

/// @brief Prints a stats::VStatisticResult (V-statistic result) as one line, `VStatisticResult(field=value, ...)`.
template <>
struct std::formatter<datamunge::stats::VStatisticResult> : datamunge::format_detail::text_formatter<datamunge::stats::VStatisticResult> {
  /// @brief One-line field listing.
  /// @param x The value to print.
  /// @return `VStatisticResult(...)` with every field; vectors are elided previews.
  static std::string text(const datamunge::stats::VStatisticResult& x) {
    return std::format("VStatisticResult(estimate={}, degree={}, num_terms={}, exact={})", x.estimate, x.degree, x.num_terms, x.exact);
  }
};

/// @brief Prints a stats::XGBoostClassifierOptions (XGBoost classifier options) as one line, `XGBoostClassifierOptions(field=value, ...)`.
template <>
struct std::formatter<datamunge::stats::XGBoostClassifierOptions> : datamunge::format_detail::text_formatter<datamunge::stats::XGBoostClassifierOptions> {
  /// @brief One-line field listing.
  /// @param x The value to print.
  /// @return `XGBoostClassifierOptions(...)` with every field; vectors are elided previews.
  static std::string text(const datamunge::stats::XGBoostClassifierOptions& x) {
    return std::format("XGBoostClassifierOptions(n_trees={}, learning_rate={}, max_depth={}, lambda={}, alpha={}, gamma={}, min_child_weight={}, min_samples_leaf={}, subsample={}, colsample_bytree={}, seed={})", x.n_trees, x.learning_rate, x.max_depth, x.lambda, x.alpha, x.gamma, x.min_child_weight, x.min_samples_leaf, x.subsample, x.colsample_bytree, x.seed);
  }
};

/// @brief Prints a stats::XGBoostClassifierPrediction (XGBoost classifier prediction) as one line, `XGBoostClassifierPrediction(field=value, ...)`.
template <>
struct std::formatter<datamunge::stats::XGBoostClassifierPrediction> : datamunge::format_detail::text_formatter<datamunge::stats::XGBoostClassifierPrediction> {
  /// @brief One-line field listing.
  /// @param x The value to print.
  /// @return `XGBoostClassifierPrediction(...)` with every field; vectors are elided previews.
  static std::string text(const datamunge::stats::XGBoostClassifierPrediction& x) {
    namespace f = datamunge::format_detail::stats_fmt;
    return std::format("XGBoostClassifierPrediction(class_label={}, probability={})", f::vec(x.class_label), f::rows(x.probability));
  }
};

/// @brief Prints a stats::XGBoostRegressorOptions (XGBoost regressor options) as one line, `XGBoostRegressorOptions(field=value, ...)`.
template <>
struct std::formatter<datamunge::stats::XGBoostRegressorOptions> : datamunge::format_detail::text_formatter<datamunge::stats::XGBoostRegressorOptions> {
  /// @brief One-line field listing.
  /// @param x The value to print.
  /// @return `XGBoostRegressorOptions(...)` with every field; vectors are elided previews.
  static std::string text(const datamunge::stats::XGBoostRegressorOptions& x) {
    return std::format("XGBoostRegressorOptions(n_trees={}, learning_rate={}, max_depth={}, lambda={}, alpha={}, gamma={}, min_child_weight={}, min_samples_leaf={}, subsample={}, colsample_bytree={}, seed={})", x.n_trees, x.learning_rate, x.max_depth, x.lambda, x.alpha, x.gamma, x.min_child_weight, x.min_samples_leaf, x.subsample, x.colsample_bytree, x.seed);
  }
};

// ---------------------------------------------------------------- hand-written

/// @brief Prints a stats::HypothesisTestResult as one line: method, statistic, degrees of freedom, p-value, estimates, CI.
template <>
struct std::formatter<datamunge::stats::HypothesisTestResult>
    : datamunge::format_detail::text_formatter<datamunge::stats::HypothesisTestResult> {
  /// @brief One-line test report.
  /// @param x The test result.
  /// @return `HypothesisTestResult(method="...", statistic=..., ..., conf_int=[lo, hi] | none, alternative=...)`.
  static std::string text(const datamunge::stats::HypothesisTestResult& x) {
    namespace f = datamunge::format_detail::stats_fmt;
    const std::string ci =
        x.has_conf_int ? f::interval({x.conf_int_lower, x.conf_int_upper}) : std::string("none");
    return std::format(
        "HypothesisTestResult(method={}, statistic={}, parameter1={}, parameter2={}, p_value={}, estimate1={}, "
        "estimate2={}, conf_int={}, alternative={})",
        f::quoted(x.method), x.statistic, x.parameter1, x.parameter2, x.p_value, x.estimate1, x.estimate2, ci,
        x.alternative);
  }
};

/// @brief Prints a stats::ExprNode (a formula expression tree) as infix text, `ExprNode(x * (y + 2))`.
template <>
struct std::formatter<datamunge::stats::ExprNode> : datamunge::format_detail::text_formatter<datamunge::stats::ExprNode> {
  /// @brief The expression in infix form.
  /// @param x The expression node.
  /// @return `ExprNode(<infix expression>)`.
  static std::string text(const datamunge::stats::ExprNode& x) {
    return std::format("ExprNode({})", datamunge::format_detail::stats_fmt::expr(x));
  }
};

/// @brief Prints a stats::ResolvedColumn (one design-matrix column) as one line with its kind, name and non-empty details.
template <>
struct std::formatter<datamunge::stats::ResolvedColumn>
    : datamunge::format_detail::text_formatter<datamunge::stats::ResolvedColumn> {
  /// @brief One-line column description.
  /// @param x The resolved column.
  /// @return `ResolvedColumn(kind=..., name="...", source_column="...", ...)`; empty details are omitted.
  static std::string text(const datamunge::stats::ResolvedColumn& x) {
    namespace f = datamunge::format_detail::stats_fmt;
    std::string out = std::format("ResolvedColumn(kind={}, name={}", x.kind, f::quoted(x.name));
    if (!x.source_column.empty()) out += std::format(", source_column={}", f::quoted(x.source_column));
    if (!x.level.empty()) out += std::format(", level={}", f::quoted(x.level));
    if (!x.function_name.empty()) out += std::format(", function_name={}", f::quoted(x.function_name));
    if (x.poly_degree != 0) out += std::format(", poly_degree={}", x.poly_degree);
    if (!x.bspline_columns.empty()) {
      out += std::format(", bspline_columns={}, bspline_component={}", f::vec(x.bspline_columns), x.bspline_component);
    }
    if (!x.interaction_parts.empty()) {
      std::vector<std::string> names;
      for (const auto& part : x.interaction_parts) names.push_back(part.name);
      out += std::format(", interaction_parts={}", f::vec(names));
    }
    if (x.expression) out += std::format(", expression={}", f::expr(*x.expression));
    return out + ")";
  }
};

/// @brief Prints a stats::DesignMatrixData as its matrix shape plus response/row previews.
template <>
struct std::formatter<datamunge::stats::DesignMatrixData>
    : datamunge::format_detail::text_formatter<datamunge::stats::DesignMatrixData> {
  /// @brief One-line description.
  /// @param x The design data.
  /// @return `DesignMatrixData(X=<rows>x<cols>, y=[...], used_row_indices=[...])`.
  static std::string text(const datamunge::stats::DesignMatrixData& x) {
    namespace f = datamunge::format_detail::stats_fmt;
    return std::format("DesignMatrixData(X={}, y={}, used_row_indices={})", f::shape(x.X), f::vec(x.y),
                       f::vec(x.used_row_indices));
  }
};

/// @brief Prints a stats::ClassificationDesignData as its matrix shape plus label/row previews.
template <>
struct std::formatter<datamunge::stats::ClassificationDesignData>
    : datamunge::format_detail::text_formatter<datamunge::stats::ClassificationDesignData> {
  /// @brief One-line description.
  /// @param x The design data.
  /// @return `ClassificationDesignData(X=<rows>x<cols>, labels=[...], used_row_indices=[...])`.
  static std::string text(const datamunge::stats::ClassificationDesignData& x) {
    namespace f = datamunge::format_detail::stats_fmt;
    return std::format("ClassificationDesignData(X={}, labels={}, used_row_indices={})", f::shape(x.X),
                       f::vec(x.labels), f::vec(x.used_row_indices));
  }
};

/// @brief Prints a stats::DesignInfo (a resolved formula) as response, intercept, coefficient names and term groups.
template <>
struct std::formatter<datamunge::stats::DesignInfo>
    : datamunge::format_detail::text_formatter<datamunge::stats::DesignInfo> {
  /// @brief One-line description.
  /// @param x The design info.
  /// @return `DesignInfo(response_name="...", has_intercept=..., coefficient_names=[...], term_groups=[...], ...)`.
  static std::string text(const datamunge::stats::DesignInfo& x) {
    namespace f = datamunge::format_detail::stats_fmt;
    std::vector<std::string> groups;
    for (const auto& group : x.term_groups) groups.push_back(group.label);
    return std::format(
        "DesignInfo(response_name={}, has_intercept={}, columns={}, coefficient_names={}, term_groups={}, "
        "categorical_columns={})",
        f::quoted(x.response_name), x.has_intercept, x.columns.size(), f::vec(x.coefficient_names), f::vec(groups),
        x.categorical_levels.size());
  }
};

/// @brief Prints a stats::Formula as its source text, e.g. `len ~ supp * dose`.
template <>
struct std::formatter<datamunge::stats::Formula> : datamunge::format_detail::text_formatter<datamunge::stats::Formula> {
  /// @brief The formula text.
  /// @param x The formula.
  /// @return `x.text()`.
  static std::string text(const datamunge::stats::Formula& x) { return x.text(); }
};

/// @brief Prints a fitted stats::ARIMA: order, coefficients and fit statistics (three lines).
template <>
struct std::formatter<datamunge::stats::ARIMA> : datamunge::format_detail::text_formatter<datamunge::stats::ARIMA> {
  /// @brief Order header, coefficients, and sigma²/log-likelihood/AIC/BIC.
  /// @param x The fitted model.
  /// @return E.g. `ARIMA(1,0,0) with mean, n=100` followed by coefficient and fit-statistic lines.
  static std::string text(const datamunge::stats::ARIMA& x) {
    namespace f = datamunge::format_detail::stats_fmt;
    const auto& o = x.options();
    std::string out = std::format("ARIMA({},{},{})", o.p, o.d, o.q);
    if (o.seasonal_period > 0) {
      out += std::format("({},{},{})[{}]", o.seasonal_p, o.seasonal_d, o.seasonal_q, o.seasonal_period);
    }
    out += std::format("{}, n={}\n", o.include_mean ? " with mean" : "", x.observations());
    out += std::format("ar={}, ma={}", f::vec(x.ar_coefficients()), f::vec(x.ma_coefficients()));
    if (o.seasonal_period > 0) {
      out += std::format(", sar={}, sma={}", f::vec(x.seasonal_ar_coefficients()), f::vec(x.seasonal_ma_coefficients()));
    }
    out += std::format(", mean={}\n", x.mean());
    out += std::format("sigma2={}, log_likelihood={}, aic={}, bic={}", x.sigma2(), x.log_likelihood(), x.aic(), x.bic());
    return out;
  }
};

/// @brief Prints a fitted stats::ExponentialSmoothing: components, smoothing parameters and fit statistics (three lines).
template <>
struct std::formatter<datamunge::stats::ExponentialSmoothing>
    : datamunge::format_detail::text_formatter<datamunge::stats::ExponentialSmoothing> {
  /// @brief Component header, fitted smoothing parameters, and SSE/sigma²/log-likelihood/AIC/BIC.
  /// @param x The fitted model.
  /// @return E.g. `ExponentialSmoothing(trend=Additive, seasonal=None, seasonal_period=0), n=40` plus two lines.
  static std::string text(const datamunge::stats::ExponentialSmoothing& x) {
    const auto& o = x.options();
    return std::format(
        "ExponentialSmoothing(trend={}, seasonal={}, seasonal_period={}), n={}\n"
        "alpha={}, beta={}, gamma={}, phi={}\n"
        "sse={}, sigma2={}, log_likelihood={}, aic={}, bic={}",
        o.trend, o.seasonal, o.seasonal_period, x.observations(), x.alpha(), x.beta(), x.gamma(), x.phi(), x.sse(),
        x.sigma2(), x.log_likelihood(), x.aic(), x.bic());
  }
};

/// @brief Prints a calibrated stats::SplitConformalClassifier as its settings and calibrated quantile.
template <>
struct std::formatter<datamunge::stats::SplitConformalClassifier>
    : datamunge::format_detail::text_formatter<datamunge::stats::SplitConformalClassifier> {
  /// @brief One-line description.
  /// @param x The calibrated classifier.
  /// @return `SplitConformalClassifier(alpha=..., score="...", quantile=..., n_calibration=..., classes=[...])`.
  static std::string text(const datamunge::stats::SplitConformalClassifier& x) {
    namespace f = datamunge::format_detail::stats_fmt;
    return std::format("SplitConformalClassifier(alpha={}, score={}, quantile={}, n_calibration={}, classes={})",
                       x.alpha(), f::quoted(x.score()), x.quantile(), x.n_calibration(), f::vec(x.classes()));
  }
};

/// @brief Prints a calibrated stats::SplitConformalRegressor as its settings and calibrated quantile.
template <>
struct std::formatter<datamunge::stats::SplitConformalRegressor>
    : datamunge::format_detail::text_formatter<datamunge::stats::SplitConformalRegressor> {
  /// @brief One-line description.
  /// @param x The calibrated regressor.
  /// @return `SplitConformalRegressor(alpha=..., quantile=..., n_calibration=...)`.
  static std::string text(const datamunge::stats::SplitConformalRegressor& x) {
    return std::format("SplitConformalRegressor(alpha={}, quantile={}, n_calibration={})", x.alpha(), x.quantile(),
                       x.n_calibration());
  }
};

/// @brief Prints a stats::LimeImageExplainer as its patch grid and local surrogate fit.
template <>
struct std::formatter<datamunge::stats::LimeImageExplainer>
    : datamunge::format_detail::text_formatter<datamunge::stats::LimeImageExplainer> {
  /// @brief One-line description.
  /// @param x The explainer.
  /// @return `LimeImageExplainer(patches=<rows>x<cols>, intercept=..., local_r_squared=...)`.
  static std::string text(const datamunge::stats::LimeImageExplainer& x) {
    return std::format("LimeImageExplainer(patches={}, intercept={}, local_r_squared={})",
                       datamunge::format_detail::stats_fmt::shape(x.patch_weights()), x.intercept(),
                       x.local_r_squared());
  }
};

/// @brief Prints a stats::ShapImageExplainer as its patch grid and base/full model values.
template <>
struct std::formatter<datamunge::stats::ShapImageExplainer>
    : datamunge::format_detail::text_formatter<datamunge::stats::ShapImageExplainer> {
  /// @brief One-line description.
  /// @param x The explainer.
  /// @return `ShapImageExplainer(patches=<rows>x<cols>, base_value=..., full_value=...)`.
  static std::string text(const datamunge::stats::ShapImageExplainer& x) {
    return std::format("ShapImageExplainer(patches={}, base_value={}, full_value={})",
                       datamunge::format_detail::stats_fmt::shape(x.patch_shapley()), x.base_value(), x.full_value());
  }
};

/// @brief Prints a stats::Bootstrap as `Bootstrap(num_resamples=B, confidence_level=c, seed=s)`.
template <>
struct std::formatter<datamunge::stats::Bootstrap> : datamunge::format_detail::text_formatter<datamunge::stats::Bootstrap> {
  /// @brief The configured options.
  /// @param x The resampler.
  /// @return The one-line text.
  static std::string text(const datamunge::stats::Bootstrap& x) {
    const auto& o = x.options();
    return std::format("Bootstrap(num_resamples={}, confidence_level={}, seed={})", o.num_resamples,
                       o.confidence_level, o.seed);
  }
};

/// @brief Prints a stats::UStatistic as `UStatistic(degree=m, max_terms=t, seed=s)`.
template <>
struct std::formatter<datamunge::stats::UStatistic>
    : datamunge::format_detail::text_formatter<datamunge::stats::UStatistic> {
  /// @brief The kernel degree and configured options.
  /// @param x The estimator.
  /// @return The one-line text.
  static std::string text(const datamunge::stats::UStatistic& x) {
    return std::format("UStatistic(degree={}, max_terms={}, seed={})", x.degree(), x.options().max_terms,
                       x.options().seed);
  }
};

/// @brief Prints a stats::VStatistic as `VStatistic(degree=m, max_terms=t, seed=s)`.
template <>
struct std::formatter<datamunge::stats::VStatistic>
    : datamunge::format_detail::text_formatter<datamunge::stats::VStatistic> {
  /// @brief The kernel degree and configured options.
  /// @param x The estimator.
  /// @return The one-line text.
  static std::string text(const datamunge::stats::VStatistic& x) {
    return std::format("VStatistic(degree={}, max_terms={}, seed={})", x.degree(), x.options().max_terms,
                       x.options().seed);
  }
};

// ---------------------------------------------------------------- fitted models (summary())

/// @brief Prints a fitted stats::AgglomerativeClustering (hierarchical clustering) as its summary(), the text print_summary() writes.
template <>
struct std::formatter<datamunge::stats::AgglomerativeClustering> : datamunge::format_detail::text_formatter<datamunge::stats::AgglomerativeClustering> {
  /// @brief The model summary.
  /// @param x The fitted model.
  /// @return The same text as `x.summary()`.
  static std::string text(const datamunge::stats::AgglomerativeClustering& x) {
    return x.summary();
  }
};

/// @brief Prints a fitted stats::CNNClassifier (CNN classifier) as its summary(), the text print_summary() writes.
template <>
struct std::formatter<datamunge::stats::CNNClassifier> : datamunge::format_detail::text_formatter<datamunge::stats::CNNClassifier> {
  /// @brief The model summary.
  /// @param x The fitted model.
  /// @return The same text as `x.summary()`.
  static std::string text(const datamunge::stats::CNNClassifier& x) {
    return x.summary();
  }
};

/// @brief Prints a fitted stats::DBSCAN (DBSCAN clustering) as its summary(), the text print_summary() writes.
template <>
struct std::formatter<datamunge::stats::DBSCAN> : datamunge::format_detail::text_formatter<datamunge::stats::DBSCAN> {
  /// @brief The model summary.
  /// @param x The fitted model.
  /// @return The same text as `x.summary()`.
  static std::string text(const datamunge::stats::DBSCAN& x) {
    return x.summary();
  }
};

/// @brief Prints a fitted stats::DecisionTreeClassifier (classification tree) as its summary(), the text print_summary() writes.
template <>
struct std::formatter<datamunge::stats::DecisionTreeClassifier> : datamunge::format_detail::text_formatter<datamunge::stats::DecisionTreeClassifier> {
  /// @brief The model summary.
  /// @param x The fitted model.
  /// @return The same text as `x.summary()`.
  static std::string text(const datamunge::stats::DecisionTreeClassifier& x) {
    return x.summary();
  }
};

/// @brief Prints a fitted stats::DecisionTreeRegressor (regression tree) as its summary(), the text print_summary() writes.
template <>
struct std::formatter<datamunge::stats::DecisionTreeRegressor> : datamunge::format_detail::text_formatter<datamunge::stats::DecisionTreeRegressor> {
  /// @brief The model summary.
  /// @param x The fitted model.
  /// @return The same text as `x.summary()`.
  static std::string text(const datamunge::stats::DecisionTreeRegressor& x) {
    return x.summary();
  }
};

/// @brief Prints a fitted stats::DiffusionMaps (diffusion-map embedding) as its summary(), the text print_summary() writes.
template <>
struct std::formatter<datamunge::stats::DiffusionMaps> : datamunge::format_detail::text_formatter<datamunge::stats::DiffusionMaps> {
  /// @brief The model summary.
  /// @param x The fitted model.
  /// @return The same text as `x.summary()`.
  static std::string text(const datamunge::stats::DiffusionMaps& x) {
    return x.summary();
  }
};

/// @brief Prints a fitted stats::ElasticNet (elastic-net regression) as its summary(), the text print_summary() writes.
template <>
struct std::formatter<datamunge::stats::ElasticNet> : datamunge::format_detail::text_formatter<datamunge::stats::ElasticNet> {
  /// @brief The model summary.
  /// @param x The fitted model.
  /// @return The same text as `x.summary()`.
  static std::string text(const datamunge::stats::ElasticNet& x) {
    return x.summary();
  }
};

/// @brief Prints a fitted stats::Ridge (ridge regression) as its summary(), the text print_summary() writes.
template <>
struct std::formatter<datamunge::stats::Ridge> : datamunge::format_detail::text_formatter<datamunge::stats::Ridge> {
  /// @brief The model summary.
  /// @param x The fitted model.
  /// @return The same text as `x.summary()`.
  static std::string text(const datamunge::stats::Ridge& x) {
    return x.summary();
  }
};

/// @brief Prints a fitted stats::Lasso (lasso regression) as its summary(), the text print_summary() writes.
template <>
struct std::formatter<datamunge::stats::Lasso> : datamunge::format_detail::text_formatter<datamunge::stats::Lasso> {
  /// @brief The model summary.
  /// @param x The fitted model.
  /// @return The same text as `x.summary()`.
  static std::string text(const datamunge::stats::Lasso& x) {
    return x.summary();
  }
};

/// @brief Prints a fitted stats::FactorAnalysis (factor analysis) as its summary(), the text print_summary() writes.
template <>
struct std::formatter<datamunge::stats::FactorAnalysis> : datamunge::format_detail::text_formatter<datamunge::stats::FactorAnalysis> {
  /// @brief The model summary.
  /// @param x The fitted model.
  /// @return The same text as `x.summary()`.
  static std::string text(const datamunge::stats::FactorAnalysis& x) {
    return x.summary();
  }
};

/// @brief Prints a fitted stats::GaussianProcessRegression (Gaussian-process regression) as its summary(), the text print_summary() writes.
template <>
struct std::formatter<datamunge::stats::GaussianProcessRegression> : datamunge::format_detail::text_formatter<datamunge::stats::GaussianProcessRegression> {
  /// @brief The model summary.
  /// @param x The fitted model.
  /// @return The same text as `x.summary()`.
  static std::string text(const datamunge::stats::GaussianProcessRegression& x) {
    return x.summary();
  }
};

/// @brief Prints a fitted stats::GBMClassifier (gradient-boosting classifier) as its summary(), the text print_summary() writes.
template <>
struct std::formatter<datamunge::stats::GBMClassifier> : datamunge::format_detail::text_formatter<datamunge::stats::GBMClassifier> {
  /// @brief The model summary.
  /// @param x The fitted model.
  /// @return The same text as `x.summary()`.
  static std::string text(const datamunge::stats::GBMClassifier& x) {
    return x.summary();
  }
};

/// @brief Prints a fitted stats::GBMRegressor (gradient-boosting regressor) as its summary(), the text print_summary() writes.
template <>
struct std::formatter<datamunge::stats::GBMRegressor> : datamunge::format_detail::text_formatter<datamunge::stats::GBMRegressor> {
  /// @brief The model summary.
  /// @param x The fitted model.
  /// @return The same text as `x.summary()`.
  static std::string text(const datamunge::stats::GBMRegressor& x) {
    return x.summary();
  }
};

/// @brief Prints a fitted stats::GLM (generalized linear model) as its summary(), the text print_summary() writes.
template <>
struct std::formatter<datamunge::stats::GLM> : datamunge::format_detail::text_formatter<datamunge::stats::GLM> {
  /// @brief The model summary.
  /// @param x The fitted model.
  /// @return The same text as `x.summary()`.
  static std::string text(const datamunge::stats::GLM& x) {
    return x.summary();
  }
};

/// @brief Prints a fitted stats::GLMM (generalized linear mixed model) as its summary(), the text print_summary() writes.
template <>
struct std::formatter<datamunge::stats::GLMM> : datamunge::format_detail::text_formatter<datamunge::stats::GLMM> {
  /// @brief The model summary.
  /// @param x The fitted model.
  /// @return The same text as `x.summary()`.
  static std::string text(const datamunge::stats::GLMM& x) {
    return x.summary();
  }
};

/// @brief Prints a fitted stats::ICA (independent component analysis) as its summary(), the text print_summary() writes.
template <>
struct std::formatter<datamunge::stats::ICA> : datamunge::format_detail::text_formatter<datamunge::stats::ICA> {
  /// @brief The model summary.
  /// @param x The fitted model.
  /// @return The same text as `x.summary()`.
  static std::string text(const datamunge::stats::ICA& x) {
    return x.summary();
  }
};

/// @brief Prints a fitted stats::INLAMixedModel (INLA mixed model) as its summary(), the text print_summary() writes.
template <>
struct std::formatter<datamunge::stats::INLAMixedModel> : datamunge::format_detail::text_formatter<datamunge::stats::INLAMixedModel> {
  /// @brief The model summary.
  /// @param x The fitted model.
  /// @return The same text as `x.summary()`.
  static std::string text(const datamunge::stats::INLAMixedModel& x) {
    return x.summary();
  }
};

/// @brief Prints a fitted stats::Isomap (Isomap embedding) as its summary(), the text print_summary() writes.
template <>
struct std::formatter<datamunge::stats::Isomap> : datamunge::format_detail::text_formatter<datamunge::stats::Isomap> {
  /// @brief The model summary.
  /// @param x The fitted model.
  /// @return The same text as `x.summary()`.
  static std::string text(const datamunge::stats::Isomap& x) {
    return x.summary();
  }
};

/// @brief Prints a fitted stats::KernelPCA (kernel PCA) as its summary(), the text print_summary() writes.
template <>
struct std::formatter<datamunge::stats::KernelPCA> : datamunge::format_detail::text_formatter<datamunge::stats::KernelPCA> {
  /// @brief The model summary.
  /// @param x The fitted model.
  /// @return The same text as `x.summary()`.
  static std::string text(const datamunge::stats::KernelPCA& x) {
    return x.summary();
  }
};

/// @brief Prints a fitted stats::KernelRegression (kernel (Nadaraya-Watson) regression) as its summary(), the text print_summary() writes.
template <>
struct std::formatter<datamunge::stats::KernelRegression> : datamunge::format_detail::text_formatter<datamunge::stats::KernelRegression> {
  /// @brief The model summary.
  /// @param x The fitted model.
  /// @return The same text as `x.summary()`.
  static std::string text(const datamunge::stats::KernelRegression& x) {
    return x.summary();
  }
};

/// @brief Prints a fitted stats::KernelRidgeRegression (kernel ridge regression) as its summary(), the text print_summary() writes.
template <>
struct std::formatter<datamunge::stats::KernelRidgeRegression> : datamunge::format_detail::text_formatter<datamunge::stats::KernelRidgeRegression> {
  /// @brief The model summary.
  /// @param x The fitted model.
  /// @return The same text as `x.summary()`.
  static std::string text(const datamunge::stats::KernelRidgeRegression& x) {
    return x.summary();
  }
};

/// @brief Prints a fitted stats::KMeans (k-means clustering) as its summary(), the text print_summary() writes.
template <>
struct std::formatter<datamunge::stats::KMeans> : datamunge::format_detail::text_formatter<datamunge::stats::KMeans> {
  /// @brief The model summary.
  /// @param x The fitted model.
  /// @return The same text as `x.summary()`.
  static std::string text(const datamunge::stats::KMeans& x) {
    return x.summary();
  }
};

/// @brief Prints a fitted stats::KNNClassifier (k-nearest-neighbour classifier) as its summary(), the text print_summary() writes.
template <>
struct std::formatter<datamunge::stats::KNNClassifier> : datamunge::format_detail::text_formatter<datamunge::stats::KNNClassifier> {
  /// @brief The model summary.
  /// @param x The fitted model.
  /// @return The same text as `x.summary()`.
  static std::string text(const datamunge::stats::KNNClassifier& x) {
    return x.summary();
  }
};

/// @brief Prints a fitted stats::KNNRegressor (k-nearest-neighbour regressor) as its summary(), the text print_summary() writes.
template <>
struct std::formatter<datamunge::stats::KNNRegressor> : datamunge::format_detail::text_formatter<datamunge::stats::KNNRegressor> {
  /// @brief The model summary.
  /// @param x The fitted model.
  /// @return The same text as `x.summary()`.
  static std::string text(const datamunge::stats::KNNRegressor& x) {
    return x.summary();
  }
};

/// @brief Prints a fitted stats::LaplacianEigenmaps (Laplacian-eigenmaps embedding) as its summary(), the text print_summary() writes.
template <>
struct std::formatter<datamunge::stats::LaplacianEigenmaps> : datamunge::format_detail::text_formatter<datamunge::stats::LaplacianEigenmaps> {
  /// @brief The model summary.
  /// @param x The fitted model.
  /// @return The same text as `x.summary()`.
  static std::string text(const datamunge::stats::LaplacianEigenmaps& x) {
    return x.summary();
  }
};

/// @brief Prints a fitted stats::LDA (linear discriminant analysis) as its summary(), the text print_summary() writes.
template <>
struct std::formatter<datamunge::stats::LDA> : datamunge::format_detail::text_formatter<datamunge::stats::LDA> {
  /// @brief The model summary.
  /// @param x The fitted model.
  /// @return The same text as `x.summary()`.
  static std::string text(const datamunge::stats::LDA& x) {
    return x.summary();
  }
};

/// @brief Prints a fitted stats::LLE (locally linear embedding) as its summary(), the text print_summary() writes.
template <>
struct std::formatter<datamunge::stats::LLE> : datamunge::format_detail::text_formatter<datamunge::stats::LLE> {
  /// @brief The model summary.
  /// @param x The fitted model.
  /// @return The same text as `x.summary()`.
  static std::string text(const datamunge::stats::LLE& x) {
    return x.summary();
  }
};

/// @brief Prints a fitted stats::LM (linear model) as its summary(), the text print_summary() writes.
template <>
struct std::formatter<datamunge::stats::LM> : datamunge::format_detail::text_formatter<datamunge::stats::LM> {
  /// @brief The model summary.
  /// @param x The fitted model.
  /// @return The same text as `x.summary()`.
  static std::string text(const datamunge::stats::LM& x) {
    return x.summary();
  }
};

/// @brief Prints a fitted stats::LMM (linear mixed model) as its summary(), the text print_summary() writes.
template <>
struct std::formatter<datamunge::stats::LMM> : datamunge::format_detail::text_formatter<datamunge::stats::LMM> {
  /// @brief The model summary.
  /// @param x The fitted model.
  /// @return The same text as `x.summary()`.
  static std::string text(const datamunge::stats::LMM& x) {
    return x.summary();
  }
};

/// @brief Prints a fitted stats::MDS (multidimensional scaling) as its summary(), the text print_summary() writes.
template <>
struct std::formatter<datamunge::stats::MDS> : datamunge::format_detail::text_formatter<datamunge::stats::MDS> {
  /// @brief The model summary.
  /// @param x The fitted model.
  /// @return The same text as `x.summary()`.
  static std::string text(const datamunge::stats::MDS& x) {
    return x.summary();
  }
};

/// @brief Prints a fitted stats::MLPClassifier (multilayer-perceptron classifier) as its summary(), the text print_summary() writes.
template <>
struct std::formatter<datamunge::stats::MLPClassifier> : datamunge::format_detail::text_formatter<datamunge::stats::MLPClassifier> {
  /// @brief The model summary.
  /// @param x The fitted model.
  /// @return The same text as `x.summary()`.
  static std::string text(const datamunge::stats::MLPClassifier& x) {
    return x.summary();
  }
};

/// @brief Prints a fitted stats::NaiveBayesClassifier (naive Bayes classifier) as its summary(), the text print_summary() writes.
template <>
struct std::formatter<datamunge::stats::NaiveBayesClassifier> : datamunge::format_detail::text_formatter<datamunge::stats::NaiveBayesClassifier> {
  /// @brief The model summary.
  /// @param x The fitted model.
  /// @return The same text as `x.summary()`.
  static std::string text(const datamunge::stats::NaiveBayesClassifier& x) {
    return x.summary();
  }
};

/// @brief Prints a fitted stats::PCA (principal component analysis) as its summary(), the text print_summary() writes.
template <>
struct std::formatter<datamunge::stats::PCA> : datamunge::format_detail::text_formatter<datamunge::stats::PCA> {
  /// @brief The model summary.
  /// @param x The fitted model.
  /// @return The same text as `x.summary()`.
  static std::string text(const datamunge::stats::PCA& x) {
    return x.summary();
  }
};

/// @brief Prints a fitted stats::QuantileRegression (quantile regression) as its summary(), the text print_summary() writes.
template <>
struct std::formatter<datamunge::stats::QuantileRegression> : datamunge::format_detail::text_formatter<datamunge::stats::QuantileRegression> {
  /// @brief The model summary.
  /// @param x The fitted model.
  /// @return The same text as `x.summary()`.
  static std::string text(const datamunge::stats::QuantileRegression& x) {
    return x.summary();
  }
};

/// @brief Prints a fitted stats::RandomForestClassifier (random-forest classifier) as its summary(), the text print_summary() writes.
template <>
struct std::formatter<datamunge::stats::RandomForestClassifier> : datamunge::format_detail::text_formatter<datamunge::stats::RandomForestClassifier> {
  /// @brief The model summary.
  /// @param x The fitted model.
  /// @return The same text as `x.summary()`.
  static std::string text(const datamunge::stats::RandomForestClassifier& x) {
    return x.summary();
  }
};

/// @brief Prints a fitted stats::RandomForestRegressor (random-forest regressor) as its summary(), the text print_summary() writes.
template <>
struct std::formatter<datamunge::stats::RandomForestRegressor> : datamunge::format_detail::text_formatter<datamunge::stats::RandomForestRegressor> {
  /// @brief The model summary.
  /// @param x The fitted model.
  /// @return The same text as `x.summary()`.
  static std::string text(const datamunge::stats::RandomForestRegressor& x) {
    return x.summary();
  }
};

/// @brief Prints a fitted stats::SammonMapping (Sammon mapping) as its summary(), the text print_summary() writes.
template <>
struct std::formatter<datamunge::stats::SammonMapping> : datamunge::format_detail::text_formatter<datamunge::stats::SammonMapping> {
  /// @brief The model summary.
  /// @param x The fitted model.
  /// @return The same text as `x.summary()`.
  static std::string text(const datamunge::stats::SammonMapping& x) {
    return x.summary();
  }
};

/// @brief Prints a fitted stats::SVM (support vector machine) as its summary(), the text print_summary() writes.
template <>
struct std::formatter<datamunge::stats::SVM> : datamunge::format_detail::text_formatter<datamunge::stats::SVM> {
  /// @brief The model summary.
  /// @param x The fitted model.
  /// @return The same text as `x.summary()`.
  static std::string text(const datamunge::stats::SVM& x) {
    return x.summary();
  }
};

/// @brief Prints a fitted stats::TSNE (t-SNE embedding) as its summary(), the text print_summary() writes.
template <>
struct std::formatter<datamunge::stats::TSNE> : datamunge::format_detail::text_formatter<datamunge::stats::TSNE> {
  /// @brief The model summary.
  /// @param x The fitted model.
  /// @return The same text as `x.summary()`.
  static std::string text(const datamunge::stats::TSNE& x) {
    return x.summary();
  }
};

/// @brief Prints a fitted stats::UMAP (UMAP embedding) as its summary(), the text print_summary() writes.
template <>
struct std::formatter<datamunge::stats::UMAP> : datamunge::format_detail::text_formatter<datamunge::stats::UMAP> {
  /// @brief The model summary.
  /// @param x The fitted model.
  /// @return The same text as `x.summary()`.
  static std::string text(const datamunge::stats::UMAP& x) {
    return x.summary();
  }
};

/// @brief Prints a fitted stats::XGBoostClassifier (XGBoost classifier) as its summary(), the text print_summary() writes.
template <>
struct std::formatter<datamunge::stats::XGBoostClassifier> : datamunge::format_detail::text_formatter<datamunge::stats::XGBoostClassifier> {
  /// @brief The model summary.
  /// @param x The fitted model.
  /// @return The same text as `x.summary()`.
  static std::string text(const datamunge::stats::XGBoostClassifier& x) {
    return x.summary();
  }
};

/// @brief Prints a fitted stats::XGBoostRegressor (XGBoost regressor) as its summary(), the text print_summary() writes.
template <>
struct std::formatter<datamunge::stats::XGBoostRegressor> : datamunge::format_detail::text_formatter<datamunge::stats::XGBoostRegressor> {
  /// @brief The model summary.
  /// @param x The fitted model.
  /// @return The same text as `x.summary()`.
  static std::string text(const datamunge::stats::XGBoostRegressor& x) {
    return x.summary();
  }
};
