#pragma once

#include <datamunge/dstruct/dataframe.hpp>
#include <datamunge/fda/bspline.hpp>
#include <datamunge/linalg/dense_matrix.hpp>

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <unordered_map>
#include <vector>

namespace datamunge::stats {

/// @brief Categorical coding for a formula factor with levels in lexical order.
/// @par Example
/// @code
/// const auto coding = dm::stats::FactorContrast::Sum;
/// std::println("{}", coding == dm::stats::FactorContrast::Sum); // true
/// @endcode
/// @see LmOptions
enum class FactorContrast {
    Treatment, ///< The first level is the reference and has an all-zero code.
    Sum,       ///< The last level receives minus one in every contrast column.
};

// ---- Arithmetic expression AST used for I(...) terms ----

enum class ExprOp { Literal, Column, Add, Subtract, Multiply, Divide, Power, Negate };

struct ExprNode {
    ExprOp                       op;
    double                       literal{0.0};
    std::string                  column;
    std::shared_ptr<ExprNode>    left;
    std::shared_ptr<ExprNode>    right;

    double evaluate(const dstruct::DataFrame& data, std::size_t row) const;
};

// ---- Resolved design columns ----

enum class ResolvedColumnKind { Numeric, CategoricalDummy, Function, BSpline, Interaction, Expression };

struct ResolvedColumn {
    ResolvedColumnKind           kind{ResolvedColumnKind::Numeric};
    std::string                  name;             // coefficient / column display name
    std::string                  source_column;     // Numeric / CategoricalDummy / Function
    std::string                  level;             // CategoricalDummy
    std::string                  function_name;     // Function: log, log10, log2, sqrt, exp, abs, poly
    int                          poly_degree{0};    // Function == "poly": this term's power (1..degree)
    std::vector<std::string>     bspline_columns;    // BSpline input columns
    std::shared_ptr<fda::BSpline> bspline;           // BSpline basis fixed at model fit time
    std::size_t                  bspline_component{0};
    std::vector<ResolvedColumn>  interaction_parts; // Interaction
    std::shared_ptr<ExprNode>    expression;        // Expression
    std::optional<std::string>  negative_level; ///< Last level of a sum-coded categorical column.

    // Returns false (row should be dropped) if a required value is null.
    bool evaluate(const dstruct::DataFrame& data, std::size_t row, double& out) const;
};

struct TermGroup {
    std::string              label;
    std::vector<std::size_t> column_indices; // indices into DesignInfo::columns / coefficient_names (excluding intercept)
    std::vector<std::string> factors; ///< Sorted constituent formula expressions for term containment.
};

struct DesignMatrixData {
    linalg::DenseMatrix<double> X;
    std::vector<double>         y;               // empty when the frame has no response column
    std::vector<std::size_t>    used_row_indices; // rows of the source DataFrame kept after dropping nulls
};

struct ClassificationDesignData {
    linalg::DenseMatrix<double> X;               // predictor columns only, never includes an intercept
    std::vector<std::string>    labels;           // response class label per kept row
    std::vector<std::size_t>    used_row_indices; // rows of the source DataFrame kept after dropping nulls
};

struct DesignInfo {
    std::string                                              response_name;
    bool                                                      has_intercept{true};
    std::vector<ResolvedColumn>                               columns;       // excludes the intercept
    std::vector<std::string>                                  coefficient_names; // includes "(Intercept)" when present
    std::vector<TermGroup>                                    term_groups;
    std::unordered_map<std::string, std::vector<std::string>> categorical_levels; // source column -> sorted levels (incl. reference)

    [[nodiscard]] DesignMatrixData build_matrix(const dstruct::DataFrame& data, bool require_response) const;

    // Like build_matrix, but for a categorical (string) response: never
    // includes an intercept column, and drops rows with a null response in
    // addition to rows with null predictors. Used by classifiers (LDA).
    [[nodiscard]] ClassificationDesignData build_classification_matrix(const dstruct::DataFrame& data) const;

    // Throws std::runtime_error if any categorical predictor column in
    // `data` contains a level that was not seen when the model was fit.
    // `caller` is used only to prefix the error message (e.g. "LM::predict").
    void validate_categorical_levels(const dstruct::DataFrame& data, const std::string& caller) const;
};

// Parses and resolves R-style model formulas ("y ~ x1 + x2", "y ~ .", "y ~
// x1 * x2 - 1", "y ~ log(x1) + poly(x2, 2) + bs(x1) + bs(x1, x2) + I(x1^2)") against a DataFrame's
// schema. Supported grammar:
//   - '+' / '-' to add/remove terms, '.' for "all other columns"
//   - '1' / '0' / '-1' to force/remove the intercept
//   - ':' for interactions, '*' for a*b == a + b + a:b (crossing)
//   - '(' ... ')' grouping
//   - named single-argument functions: log, log10, log2, sqrt, exp, abs
//   - poly(x, degree) for raw polynomial terms
//   - bs(x1, ...): cubic, six-basis-function-per-axis B-spline tensor-product terms
//     (up to four numeric inputs); one reference column is omitted to avoid collinearity
//     with the formula intercept
//   - I(expr) for a literal arithmetic expression (+, -, *, /, ^, parens)
// String/categorical columns are dummy-encoded with a dropped reference
// level (the alphabetically-first level), matching R's default treatment
// contrasts.
enum class ResponseKind { Numeric, Categorical };

/// @brief A parsed model formula such as `"y ~ x1 + x2"`, R's `formula` object.
///
/// Parses (and validates the syntax of) the text on construction; columns are
/// resolved later against a data frame by each model. Converts implicitly from
/// a string, so model constructors taking a Formula also accept `"y ~ x"`.
///
/// @par Example
/// @code
/// const auto f = dm::formula("len ~ supp * dose");
/// const auto model = dm::stats::LM(f, dm::datasets::tooth_growth());
/// std::print("{} | response: {}\n", f, f.response_name());  // len ~ supp * dose | response: len
/// @endcode
///
/// @see formula()
class Formula {
 public:
    /// @brief Parses a formula from its text (implicit, so a std::string can stand in for a Formula).
    /// @param formula_text  The formula, e.g. `"y ~ x1 + log(x2)"`.
    /// @throws std::invalid_argument  If the text has no `~` or a malformed term.
    Formula(std::string formula_text);  // NOLINT(google-explicit-constructor): implicit by design

    /// @brief Parses a formula from a string literal (implicit, so `"y ~ x"` can stand in for a Formula).
    /// @param formula_text  The formula, e.g. `"y ~ x1 + log(x2)"`.
    /// @throws std::invalid_argument  If the text has no `~` or a malformed term.
    Formula(const char* formula_text)  // NOLINT(google-explicit-constructor): implicit by design
        : Formula(std::string(formula_text)) {}

    [[nodiscard]] const std::string& text() const { return text_; }
    [[nodiscard]] const std::string& response_name() const { return response_name_; }

    // `response_kind` controls what the LHS column is validated against:
    // Numeric for regression-style models (LM), Categorical (string) for
    // classifiers (LDA). Any '1'/'0'/'-1' intercept terms are parsed but
    // only meaningful for Numeric responses — classification designs never
    // include an intercept column (see build_classification_matrix).
    // `force_no_intercept` additionally suppresses the intercept for a
    // Numeric response regardless of the formula text, for models (like
    // DecisionTreeRegressor) where a constant column is never meaningful.
    [[nodiscard]] DesignInfo resolve(const dstruct::DataFrame& data, ResponseKind response_kind = ResponseKind::Numeric,
                                     bool force_no_intercept = false) const;

    /// @brief Resolves a numerical-response formula with explicit coding for selected factors.
    /// @param data Frame containing the formula's response and predictors.
    /// @param contrasts Factor name to coding. Unlisted factors use treatment coding.
    /// @return Resolved design and term metadata used for fitting and prediction.
    /// @throws std::invalid_argument Invalid formula data, contrast method, or contrast factor.
    /// @par Example
    /// @code
    /// dm::dstruct::DataFrame data;
    /// data.add_column("y", std::vector<double>{1, 2});
    /// data.add_column("g", std::vector<std::string>{"A", "B"});
    /// const auto design = dm::stats::Formula("y ~ g").resolve_with_contrasts(data, {{"g", dm::stats::FactorContrast::Sum}});
    /// std::println("{:.0f}", design.build_matrix(data, true).X(1, 1)); // -1
    /// @endcode
    [[nodiscard]] DesignInfo resolve_with_contrasts(const dstruct::DataFrame& data,
        const std::unordered_map<std::string, FactorContrast>& contrasts) const;

 private:
    /// @brief Shared formula resolution for numerical and categorical responses.
    [[nodiscard]] DesignInfo resolve_impl(const dstruct::DataFrame& data, ResponseKind response_kind,
        bool force_no_intercept, const std::unordered_map<std::string, FactorContrast>& contrasts) const;

    struct RawTerm {
        std::string factors; // ':'-joined factor text, still unresolved
    };

    std::string              text_;
    std::string              response_name_;
    bool                     intercept_requested_{true};
    std::vector<std::string> raw_terms_; // ':'-joined factor strings, '.' kept literal for later expansion
    std::vector<std::string> removed_terms_; // terms subtracted with '-', also excluded from the expansion of '.'

    void parse();
};

/// @brief Builds a model formula, like R's `formula("y ~ x")` / `y ~ x`.
///
/// Pass the result to any formula-based model in R's `(formula, data)` order.
///
/// @param text  The formula text, e.g. `"len ~ supp * dose"`.
/// @return The parsed formula.
/// @throws std::invalid_argument  If the text has no `~` or a malformed term.
///
/// @par Example
/// @code
/// const auto model = dm::stats::LM(dm::formula("len ~ supp * dose"), dm::datasets::tooth_growth());
/// std::print("{}\n", model.coefficient_names().size());  // 4
/// @endcode
///
/// @par References
/// - R Core Team. "formula: Model Formulae". https://stat.ethz.ch/R-manual/R-devel/library/stats/html/formula.html
inline Formula formula(std::string text) { return Formula(std::move(text)); }

} // namespace datamunge::stats

/// @brief Core library APIs and their algorithm, data, and numerical submodules.
namespace datamunge {
/// @brief `dm::formula("y ~ x")`, the top-level spelling of stats::formula().
using stats::formula;
} // namespace datamunge
