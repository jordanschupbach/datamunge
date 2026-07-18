#pragma once

#include <datamunge/dstruct/dataframe.hpp>
#include <datamunge/linalg/dense_matrix.hpp>

#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace datamunge::stats {

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

enum class ResolvedColumnKind { Numeric, CategoricalDummy, Function, Interaction, Expression };

struct ResolvedColumn {
    ResolvedColumnKind           kind{ResolvedColumnKind::Numeric};
    std::string                  name;             // coefficient / column display name
    std::string                  source_column;     // Numeric / CategoricalDummy / Function
    std::string                  level;             // CategoricalDummy
    std::string                  function_name;     // Function: log, log10, log2, sqrt, exp, abs, poly
    int                          poly_degree{0};    // Function == "poly": this term's power (1..degree)
    std::vector<ResolvedColumn>  interaction_parts; // Interaction
    std::shared_ptr<ExprNode>    expression;        // Expression

    // Returns false (row should be dropped) if a required value is null.
    bool evaluate(const dstruct::DataFrame& data, std::size_t row, double& out) const;
};

struct TermGroup {
    std::string              label;
    std::vector<std::size_t> column_indices; // indices into DesignInfo::columns / coefficient_names (excluding intercept)
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
// x1 * x2 - 1", "y ~ log(x1) + poly(x2, 2) + I(x1^2)") against a DataFrame's
// schema. Supported grammar:
//   - '+' / '-' to add/remove terms, '.' for "all other columns"
//   - '1' / '0' / '-1' to force/remove the intercept
//   - ':' for interactions, '*' for a*b == a + b + a:b (crossing)
//   - '(' ... ')' grouping
//   - named single-argument functions: log, log10, log2, sqrt, exp, abs
//   - poly(x, degree) for raw polynomial terms
//   - I(expr) for a literal arithmetic expression (+, -, *, /, ^, parens)
// String/categorical columns are dummy-encoded with a dropped reference
// level (the alphabetically-first level), matching R's default treatment
// contrasts.
enum class ResponseKind { Numeric, Categorical };

class Formula {
 public:
    explicit Formula(std::string formula_text);

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

 private:
    struct RawTerm {
        std::string factors; // ':'-joined factor text, still unresolved
    };

    std::string              text_;
    std::string              response_name_;
    bool                     intercept_requested_{true};
    std::vector<std::string> raw_terms_; // ':'-joined factor strings, '.' kept literal for later expansion

    void parse();
};

} // namespace datamunge::stats
