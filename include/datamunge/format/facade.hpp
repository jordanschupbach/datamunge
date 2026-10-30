#pragma once

/// @file facade.hpp
/// @brief std::format / std::print support for the top-level SWIG-facing classes in datamunge.hpp (dm::LM, dm::PCA,
///        dm::DataFrame, dm::Tensor, ...) and the spline regressions in bspline_regression.hpp / pspline_regression.hpp.
/// Included by <datamunge/format.hpp>; requires C++20 <format>.
///
/// Every fitted model facade prints its summary() -- the same text print_summary() writes and
/// the wrapped core model prints. The facade DataFrame and Tensor print via their to_string();
/// plotting facades print a short description and never render.
///
/// @par Example
/// @code
/// const dm::Dual d(2.0, 1.0);
/// std::print("{}\n", d.multiply(d));   // Dual(value=4, derivative=4)
/// @endcode

#include <datamunge/bspline_regression.hpp>
#include <datamunge/datamunge.hpp>
#include <datamunge/format/text_formatter.hpp>
#include <datamunge/pspline_regression.hpp>

#include <format>
#include <string>

/// @brief Prints a fitted dm::LM as its summary (coefficients table, R², F-test), the text print_summary() writes.
template <>
struct std::formatter<datamunge::LM> : datamunge::format_detail::text_formatter<datamunge::LM> {
  /// @brief The model summary.
  /// @param model The fitted model to print.
  /// @return model.summary().
  static std::string text(const datamunge::LM& model) { return model.summary(); }
};

/// @brief Prints a fitted dm::LMM as its summary(), the text print_summary() writes.
template <>
struct std::formatter<datamunge::LMM> : datamunge::format_detail::text_formatter<datamunge::LMM> {
  /// @brief The model summary.
  /// @param model The fitted model to print.
  /// @return model.summary().
  static std::string text(const datamunge::LMM& model) { return model.summary(); }
};

/// @brief Prints a fitted dm::GLMM as its summary(), the text print_summary() writes.
template <>
struct std::formatter<datamunge::GLMM> : datamunge::format_detail::text_formatter<datamunge::GLMM> {
  /// @brief The model summary.
  /// @param model The fitted model to print.
  /// @return model.summary().
  static std::string text(const datamunge::GLMM& model) { return model.summary(); }
};

/// @brief Prints a fitted dm::INLAMixedModel as its summary(), the text print_summary() writes.
template <>
struct std::formatter<datamunge::INLAMixedModel> : datamunge::format_detail::text_formatter<datamunge::INLAMixedModel> {
  /// @brief The model summary.
  /// @param model The fitted model to print.
  /// @return model.summary().
  static std::string text(const datamunge::INLAMixedModel& model) { return model.summary(); }
};

/// @brief Prints a fitted dm::LDA as its summary(), the text print_summary() writes.
template <>
struct std::formatter<datamunge::LDA> : datamunge::format_detail::text_formatter<datamunge::LDA> {
  /// @brief The model summary.
  /// @param model The fitted model to print.
  /// @return model.summary().
  static std::string text(const datamunge::LDA& model) { return model.summary(); }
};

/// @brief Prints a fitted dm::SVM as its summary(), the text print_summary() writes.
template <>
struct std::formatter<datamunge::SVM> : datamunge::format_detail::text_formatter<datamunge::SVM> {
  /// @brief The model summary.
  /// @param model The fitted model to print.
  /// @return model.summary().
  static std::string text(const datamunge::SVM& model) { return model.summary(); }
};

/// @brief Prints a fitted dm::DecisionTreeClassifier as its summary(), the text print_summary() writes.
template <>
struct std::formatter<datamunge::DecisionTreeClassifier> : datamunge::format_detail::text_formatter<datamunge::DecisionTreeClassifier> {
  /// @brief The model summary.
  /// @param model The fitted model to print.
  /// @return model.summary().
  static std::string text(const datamunge::DecisionTreeClassifier& model) { return model.summary(); }
};

/// @brief Prints a fitted dm::DecisionTreeRegressor as its summary(), the text print_summary() writes.
template <>
struct std::formatter<datamunge::DecisionTreeRegressor> : datamunge::format_detail::text_formatter<datamunge::DecisionTreeRegressor> {
  /// @brief The model summary.
  /// @param model The fitted model to print.
  /// @return model.summary().
  static std::string text(const datamunge::DecisionTreeRegressor& model) { return model.summary(); }
};

/// @brief Prints a fitted dm::RandomForestClassifier as its summary(), the text print_summary() writes.
template <>
struct std::formatter<datamunge::RandomForestClassifier> : datamunge::format_detail::text_formatter<datamunge::RandomForestClassifier> {
  /// @brief The model summary.
  /// @param model The fitted model to print.
  /// @return model.summary().
  static std::string text(const datamunge::RandomForestClassifier& model) { return model.summary(); }
};

/// @brief Prints a fitted dm::RandomForestRegressor as its summary(), the text print_summary() writes.
template <>
struct std::formatter<datamunge::RandomForestRegressor> : datamunge::format_detail::text_formatter<datamunge::RandomForestRegressor> {
  /// @brief The model summary.
  /// @param model The fitted model to print.
  /// @return model.summary().
  static std::string text(const datamunge::RandomForestRegressor& model) { return model.summary(); }
};

/// @brief Prints a fitted dm::ElasticNet as its summary(), the text print_summary() writes.
template <>
struct std::formatter<datamunge::ElasticNet> : datamunge::format_detail::text_formatter<datamunge::ElasticNet> {
  /// @brief The model summary.
  /// @param model The fitted model to print.
  /// @return model.summary().
  static std::string text(const datamunge::ElasticNet& model) { return model.summary(); }
};

/// @brief Prints a fitted dm::Ridge as its summary(), the text print_summary() writes.
template <>
struct std::formatter<datamunge::Ridge> : datamunge::format_detail::text_formatter<datamunge::Ridge> {
  /// @brief The model summary.
  /// @param model The fitted model to print.
  /// @return model.summary().
  static std::string text(const datamunge::Ridge& model) { return model.summary(); }
};

/// @brief Prints a fitted dm::Lasso as its summary(), the text print_summary() writes.
template <>
struct std::formatter<datamunge::Lasso> : datamunge::format_detail::text_formatter<datamunge::Lasso> {
  /// @brief The model summary.
  /// @param model The fitted model to print.
  /// @return model.summary().
  static std::string text(const datamunge::Lasso& model) { return model.summary(); }
};

/// @brief Prints a fitted dm::KNNClassifier as its summary(), the text print_summary() writes.
template <>
struct std::formatter<datamunge::KNNClassifier> : datamunge::format_detail::text_formatter<datamunge::KNNClassifier> {
  /// @brief The model summary.
  /// @param model The fitted model to print.
  /// @return model.summary().
  static std::string text(const datamunge::KNNClassifier& model) { return model.summary(); }
};

/// @brief Prints a fitted dm::KNNRegressor as its summary(), the text print_summary() writes.
template <>
struct std::formatter<datamunge::KNNRegressor> : datamunge::format_detail::text_formatter<datamunge::KNNRegressor> {
  /// @brief The model summary.
  /// @param model The fitted model to print.
  /// @return model.summary().
  static std::string text(const datamunge::KNNRegressor& model) { return model.summary(); }
};

/// @brief Prints a fitted dm::KMeans as its summary(), the text print_summary() writes.
template <>
struct std::formatter<datamunge::KMeans> : datamunge::format_detail::text_formatter<datamunge::KMeans> {
  /// @brief The model summary.
  /// @param model The fitted model to print.
  /// @return model.summary().
  static std::string text(const datamunge::KMeans& model) { return model.summary(); }
};

/// @brief Prints a fitted dm::PCA as its summary(), the text print_summary() writes.
template <>
struct std::formatter<datamunge::PCA> : datamunge::format_detail::text_formatter<datamunge::PCA> {
  /// @brief The model summary.
  /// @param model The fitted model to print.
  /// @return model.summary().
  static std::string text(const datamunge::PCA& model) { return model.summary(); }
};

/// @brief Prints a fitted dm::MDS as its summary(), the text print_summary() writes.
template <>
struct std::formatter<datamunge::MDS> : datamunge::format_detail::text_formatter<datamunge::MDS> {
  /// @brief The model summary.
  /// @param model The fitted model to print.
  /// @return model.summary().
  static std::string text(const datamunge::MDS& model) { return model.summary(); }
};

/// @brief Prints a fitted dm::Isomap as its summary(), the text print_summary() writes.
template <>
struct std::formatter<datamunge::Isomap> : datamunge::format_detail::text_formatter<datamunge::Isomap> {
  /// @brief The model summary.
  /// @param model The fitted model to print.
  /// @return model.summary().
  static std::string text(const datamunge::Isomap& model) { return model.summary(); }
};

/// @brief Prints a fitted dm::LLE as its summary(), the text print_summary() writes.
template <>
struct std::formatter<datamunge::LLE> : datamunge::format_detail::text_formatter<datamunge::LLE> {
  /// @brief The model summary.
  /// @param model The fitted model to print.
  /// @return model.summary().
  static std::string text(const datamunge::LLE& model) { return model.summary(); }
};

/// @brief Prints a fitted dm::TSNE as its summary(), the text print_summary() writes.
template <>
struct std::formatter<datamunge::TSNE> : datamunge::format_detail::text_formatter<datamunge::TSNE> {
  /// @brief The model summary.
  /// @param model The fitted model to print.
  /// @return model.summary().
  static std::string text(const datamunge::TSNE& model) { return model.summary(); }
};

/// @brief Prints a fitted dm::LaplacianEigenmaps as its summary(), the text print_summary() writes.
template <>
struct std::formatter<datamunge::LaplacianEigenmaps> : datamunge::format_detail::text_formatter<datamunge::LaplacianEigenmaps> {
  /// @brief The model summary.
  /// @param model The fitted model to print.
  /// @return model.summary().
  static std::string text(const datamunge::LaplacianEigenmaps& model) { return model.summary(); }
};

/// @brief Prints a fitted dm::DiffusionMaps as its summary(), the text print_summary() writes.
template <>
struct std::formatter<datamunge::DiffusionMaps> : datamunge::format_detail::text_formatter<datamunge::DiffusionMaps> {
  /// @brief The model summary.
  /// @param model The fitted model to print.
  /// @return model.summary().
  static std::string text(const datamunge::DiffusionMaps& model) { return model.summary(); }
};

/// @brief Prints a fitted dm::KernelPCA as its summary(), the text print_summary() writes.
template <>
struct std::formatter<datamunge::KernelPCA> : datamunge::format_detail::text_formatter<datamunge::KernelPCA> {
  /// @brief The model summary.
  /// @param model The fitted model to print.
  /// @return model.summary().
  static std::string text(const datamunge::KernelPCA& model) { return model.summary(); }
};

/// @brief Prints a fitted dm::SammonMapping as its summary(), the text print_summary() writes.
template <>
struct std::formatter<datamunge::SammonMapping> : datamunge::format_detail::text_formatter<datamunge::SammonMapping> {
  /// @brief The model summary.
  /// @param model The fitted model to print.
  /// @return model.summary().
  static std::string text(const datamunge::SammonMapping& model) { return model.summary(); }
};

/// @brief Prints a fitted dm::UMAP as its summary(), the text print_summary() writes.
template <>
struct std::formatter<datamunge::UMAP> : datamunge::format_detail::text_formatter<datamunge::UMAP> {
  /// @brief The model summary.
  /// @param model The fitted model to print.
  /// @return model.summary().
  static std::string text(const datamunge::UMAP& model) { return model.summary(); }
};

/// @brief Prints a fitted dm::AgglomerativeClustering as its summary(), the text print_summary() writes.
template <>
struct std::formatter<datamunge::AgglomerativeClustering> : datamunge::format_detail::text_formatter<datamunge::AgglomerativeClustering> {
  /// @brief The model summary.
  /// @param model The fitted model to print.
  /// @return model.summary().
  static std::string text(const datamunge::AgglomerativeClustering& model) { return model.summary(); }
};

/// @brief Prints a fitted dm::DBSCAN as its summary(), the text print_summary() writes.
template <>
struct std::formatter<datamunge::DBSCAN> : datamunge::format_detail::text_formatter<datamunge::DBSCAN> {
  /// @brief The model summary.
  /// @param model The fitted model to print.
  /// @return model.summary().
  static std::string text(const datamunge::DBSCAN& model) { return model.summary(); }
};

/// @brief Prints a fitted dm::GBMClassifier as its summary(), the text print_summary() writes.
template <>
struct std::formatter<datamunge::GBMClassifier> : datamunge::format_detail::text_formatter<datamunge::GBMClassifier> {
  /// @brief The model summary.
  /// @param model The fitted model to print.
  /// @return model.summary().
  static std::string text(const datamunge::GBMClassifier& model) { return model.summary(); }
};

/// @brief Prints a fitted dm::GBMRegressor as its summary(), the text print_summary() writes.
template <>
struct std::formatter<datamunge::GBMRegressor> : datamunge::format_detail::text_formatter<datamunge::GBMRegressor> {
  /// @brief The model summary.
  /// @param model The fitted model to print.
  /// @return model.summary().
  static std::string text(const datamunge::GBMRegressor& model) { return model.summary(); }
};

/// @brief Prints a fitted dm::XGBoostClassifier as its summary(), the text print_summary() writes.
template <>
struct std::formatter<datamunge::XGBoostClassifier> : datamunge::format_detail::text_formatter<datamunge::XGBoostClassifier> {
  /// @brief The model summary.
  /// @param model The fitted model to print.
  /// @return model.summary().
  static std::string text(const datamunge::XGBoostClassifier& model) { return model.summary(); }
};

/// @brief Prints a fitted dm::XGBoostRegressor as its summary(), the text print_summary() writes.
template <>
struct std::formatter<datamunge::XGBoostRegressor> : datamunge::format_detail::text_formatter<datamunge::XGBoostRegressor> {
  /// @brief The model summary.
  /// @param model The fitted model to print.
  /// @return model.summary().
  static std::string text(const datamunge::XGBoostRegressor& model) { return model.summary(); }
};

/// @brief Prints a fitted dm::KernelRegression as its summary(), the text print_summary() writes.
template <>
struct std::formatter<datamunge::KernelRegression> : datamunge::format_detail::text_formatter<datamunge::KernelRegression> {
  /// @brief The model summary.
  /// @param model The fitted model to print.
  /// @return model.summary().
  static std::string text(const datamunge::KernelRegression& model) { return model.summary(); }
};

/// @brief Prints a fitted dm::GaussianProcessRegression as its summary(), the text print_summary() writes.
template <>
struct std::formatter<datamunge::GaussianProcessRegression> : datamunge::format_detail::text_formatter<datamunge::GaussianProcessRegression> {
  /// @brief The model summary.
  /// @param model The fitted model to print.
  /// @return model.summary().
  static std::string text(const datamunge::GaussianProcessRegression& model) { return model.summary(); }
};

/// @brief Prints a fitted dm::NaiveBayesClassifier as its summary(), the text print_summary() writes.
template <>
struct std::formatter<datamunge::NaiveBayesClassifier> : datamunge::format_detail::text_formatter<datamunge::NaiveBayesClassifier> {
  /// @brief The model summary.
  /// @param model The fitted model to print.
  /// @return model.summary().
  static std::string text(const datamunge::NaiveBayesClassifier& model) { return model.summary(); }
};

/// @brief Prints a fitted dm::GLM as its summary(), the text print_summary() writes.
template <>
struct std::formatter<datamunge::GLM> : datamunge::format_detail::text_formatter<datamunge::GLM> {
  /// @brief The model summary.
  /// @param model The fitted model to print.
  /// @return model.summary().
  static std::string text(const datamunge::GLM& model) { return model.summary(); }
};

/// @brief Prints the facade dm::DataFrame via its to_string() (R-style table, first 10 rows).
template <>
struct std::formatter<datamunge::DataFrame> : datamunge::format_detail::text_formatter<datamunge::DataFrame> {
  /// @brief The frame as an R-print-style table, capped at 10 rows.
  /// @param frame The data frame to print.
  /// @return frame.to_string().
  static std::string text(const datamunge::DataFrame& frame) { return frame.to_string(); }
};

/// @brief Prints the facade dm::Tensor via its to_string() (shape, dtype, first 100 elements).
template <>
struct std::formatter<datamunge::Tensor> : datamunge::format_detail::text_formatter<datamunge::Tensor> {
  /// @brief The tensor's to_string() text, capped at 100 elements.
  /// @param tensor The tensor to print.
  /// @return tensor.to_string().
  static std::string text(const datamunge::Tensor& tensor) { return tensor.to_string(); }
};

/// @brief Prints the facade dm::ShapeLayer as "ShapeLayer(type=.., shapes=.., bounds=[xmin, ymin, xmax, ymax])".
template <>
struct std::formatter<datamunge::ShapeLayer> : datamunge::format_detail::text_formatter<datamunge::ShapeLayer> {
  /// @brief The layer's shape type, record count, and bounding box.
  /// @param layer The layer to print.
  /// @return The one-line description.
  static std::string text(const datamunge::ShapeLayer& layer) {
    return std::format("ShapeLayer(type={}, shapes={}, bounds={})", layer.shape_type(), layer.size(), layer.bounds());
  }
};

/// @brief Prints the facade dm::GGPlot as "GGPlot rows=n x=".." y=".. layers=[..] theme=t" plus any color, facet
///        and labs() text; never renders.
template <>
struct std::formatter<datamunge::GGPlot> : datamunge::format_detail::text_formatter<datamunge::GGPlot> {
  /// @brief The data size, mapped columns, layer geoms and theme.
  /// @param g The plot to print.
  /// @return The one-line description.
  static std::string text(const datamunge::GGPlot& g) {
    std::string out = std::format("GGPlot rows={} x={:?}", g.rows(), g.x_column());
    if (!g.y_column().empty()) out += std::format(" y={:?}", g.y_column());
    if (!g.color_column().empty()) out += std::format(" color={:?}", g.color_column());
    out += std::format(" layers={} theme={}", datamunge::format_detail::preview(g.layer_kinds(), 8), g.theme_name());
    if (!g.facet_column().empty()) out += std::format(" facet={:?}", g.facet_column());
    if (!g.title_text().empty()) out += std::format(" title={:?}", g.title_text());
    if (!g.x_label().empty()) out += std::format(" xlab={:?}", g.x_label());
    if (!g.y_label().empty()) out += std::format(" ylab={:?}", g.y_label());
    return out;
  }
};

/// @brief Prints the facade dm::Splom as "Splom WxH title=.. variables=[..]" plus its group_by() groups; never renders.
template <>
struct std::formatter<datamunge::Splom> : datamunge::format_detail::text_formatter<datamunge::Splom> {
  /// @brief The plotted variables and color groups.
  /// @param s The scatterplot matrix to print.
  /// @return The one-line description.
  static std::string text(const datamunge::Splom& s) {
    std::string out = std::format("Splom {}x{}", s.width(), s.height());
    if (!s.title_text().empty()) out += std::format(" title={:?}", s.title_text());
    out += std::format(" variables={}", datamunge::format_detail::preview(s.variables(), 8));
    const auto groups = s.groups();
    if (!groups.empty()) out += std::format(" groups={}", datamunge::format_detail::preview(groups, 8));
    return out;
  }
};

/// @brief Prints a dm::Dual as "Dual(value=v, derivative=d)".
template <>
struct std::formatter<datamunge::Dual> : datamunge::format_detail::text_formatter<datamunge::Dual> {
  /// @brief The value and its tangent.
  /// @param d The dual number to print.
  /// @return The one-line description.
  static std::string text(const datamunge::Dual& d) {
    return std::format("Dual(value={}, derivative={})", d.value(), d.derivative());
  }
};

/// @brief Prints a dm::HyperDual as "HyperDual(value=.., eps1=.., eps2=.., eps1eps2=..)".
template <>
struct std::formatter<datamunge::HyperDual> : datamunge::format_detail::text_formatter<datamunge::HyperDual> {
  /// @brief The value and its three infinitesimal parts.
  /// @param h The hyper-dual number to print.
  /// @return The one-line description.
  static std::string text(const datamunge::HyperDual& h) {
    return std::format("HyperDual(value={}, eps1={}, eps2={}, eps1eps2={})", h.value(), h.eps1(), h.eps2(),
                       h.eps1eps2());
  }
};

/// @brief Prints a dm::Tape as "Tape(size=n)".
template <>
struct std::formatter<datamunge::Tape> : datamunge::format_detail::text_formatter<datamunge::Tape> {
  /// @brief The number of recorded nodes.
  /// @param tape The tape to print.
  /// @return "Tape(size=n)".
  static std::string text(const datamunge::Tape& tape) { return std::format("Tape(size={})", tape.size()); }
};

/// @brief Prints a dm::Var as "Var(value=v, index=i)".
template <>
struct std::formatter<datamunge::Var> : datamunge::format_detail::text_formatter<datamunge::Var> {
  /// @brief The node's value and its index on the tape.
  /// @param v The variable to print.
  /// @return "Var(value=v, index=i)".
  static std::string text(const datamunge::Var& v) { return std::format("Var(value={}, index={})", v.value(), v.index()); }
};

/// @brief Prints a dm::BSplineRegression as its basis, fit size and R².
template <>
struct std::formatter<datamunge::BSplineRegression>
    : datamunge::format_detail::text_formatter<datamunge::BSplineRegression> {
  /// @brief "BSplineRegression(n=.., basis_size=.., degree=.., domain=[a, b], r_squared=..)".
  /// @param fit The fitted regression to print.
  /// @return The one-line description.
  static std::string text(const datamunge::BSplineRegression& fit) {
    const auto [lo, hi] = fit.domain();
    return std::format("BSplineRegression(n={}, basis_size={}, degree={}, domain=[{}, {}], r_squared={})",
                       fit.fitted().size(), fit.basis_size(), fit.degree(), lo, hi, fit.r_squared());
  }
};

/// @brief Prints a dm::PSplineRegression as its basis, penalty, effective df, GCV and R².
template <>
struct std::formatter<datamunge::PSplineRegression>
    : datamunge::format_detail::text_formatter<datamunge::PSplineRegression> {
  /// @brief "PSplineRegression(n=.., basis_size=.., degree=.., penalty_order=.., lambda=.., effective_df=.., gcv=..,
  ///        domain=[a, b], r_squared=..)".
  /// @param fit The fitted regression to print.
  /// @return The one-line description.
  static std::string text(const datamunge::PSplineRegression& fit) {
    const auto [lo, hi] = fit.domain();
    return std::format(
        "PSplineRegression(n={}, basis_size={}, degree={}, penalty_order={}, lambda={}, effective_df={}, gcv={}, "
        "domain=[{}, {}], r_squared={})",
        fit.fitted().size(), fit.basis_size(), fit.degree(), fit.penalty_order(), fit.lambda(), fit.effective_df(),
        fit.gcv(), lo, hi, fit.r_squared());
  }
};
