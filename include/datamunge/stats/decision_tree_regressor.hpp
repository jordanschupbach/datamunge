#pragma once

#include <datamunge/dstruct/dataframe.hpp>
#include <datamunge/linalg/dense_matrix.hpp>
#include <datamunge/plot/plot.hpp>
#include <datamunge/random/random.hpp>
#include <datamunge/stats/formula.hpp>

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

namespace datamunge::stats {

struct DecisionTreeRegressorOptions {
    std::size_t max_depth{5};
    std::size_t min_samples_split{2};
    std::size_t min_samples_leaf{1};
    double      min_impurity_decrease{0.0}; // required SSE reduction to accept a split
    // 0 = consider every predictor at each split (ordinary CART). When
    // nonzero, each split considers a random subset of this many predictors
    // (drawn fresh per node, seeded by random_seed) — this is the mechanism
    // RandomForestRegressor uses to decorrelate its trees.
    std::size_t   max_features{0};
    std::uint64_t random_seed{0};
};

// CART-style binary decision tree regressor, fit from a DataFrame and an
// R-style formula whose left-hand side is a numeric column:
//
//   DecisionTreeRegressor model(iris, "Petal.Length ~ Sepal.Length + Sepal.Width");
//   model.print_summary();
//   auto preds = model.predict(newdata);
//
// Recursively splits on a single predictor and threshold at each node,
// choosing the split that most reduces sum-of-squared-error; each leaf
// predicts the mean of its training targets. As with LM, any intercept
// term in the formula would normally apply, but it is never meaningful for
// a tree so it is always suppressed. Rows with a null predictor or null
// response are dropped before fitting.
class DecisionTreeRegressor {
 public:
    DecisionTreeRegressor(const dstruct::DataFrame& data, const std::string& formula,
                          DecisionTreeRegressorOptions options = {});

    [[nodiscard]] const std::string&              formula_text() const { return formula_.text(); }
    [[nodiscard]] const std::vector<std::string>& predictor_names() const { return predictor_names_; }
    [[nodiscard]] std::size_t                     observations() const { return observations_; }

    [[nodiscard]] std::size_t node_count() const { return nodes_.size(); }
    [[nodiscard]] std::size_t leaf_count() const;
    [[nodiscard]] std::size_t depth() const;

    // SSE-decrease-based importance per predictor, normalized to sum to 1.
    [[nodiscard]] std::vector<double> feature_importance() const;

    [[nodiscard]] const std::vector<double>& fitted_values() const { return fitted_values_; }
    [[nodiscard]] double                     r_squared() const;
    [[nodiscard]] double                     rmse() const;

    [[nodiscard]] std::string summary() const;
    void                      print_summary(std::ostream& os) const;
    void                      print_summary() const;

    [[nodiscard]] std::vector<double> predict(const dstruct::DataFrame& newdata) const;

    [[nodiscard]] plot::ScatterPlot plot_predicted_vs_actual() const;
    [[nodiscard]] plot::ScatterPlot plot_residuals_vs_fitted() const;

 private:
    struct Node {
        bool        is_leaf{true};
        std::size_t feature_index{0};
        double      threshold{0.0};
        std::size_t left{0};
        std::size_t right{0};
        std::size_t depth{0};
        std::size_t n_samples{0};
        double      sse{0.0};
        double      mean_value{0.0};
    };

    void        fit(const dstruct::DataFrame& data, DecisionTreeRegressorOptions options);
    std::size_t build_node(const linalg::DenseMatrix<double>& X, const std::vector<double>& y,
                           std::vector<std::size_t> rows, std::size_t depth);
    std::size_t leaf_for(const std::vector<double>& x) const;
    void        dump_node(std::ostream& os, std::size_t node_index, std::size_t indent) const;
    std::vector<std::size_t> feature_candidates(std::size_t p);

    Formula                      formula_;
    DesignInfo                   design_;
    DecisionTreeRegressorOptions options_;
    random::SplitMix64           rng_{0};

    std::vector<std::string> predictor_names_;
    std::vector<Node>        nodes_;

    std::vector<double> fitted_values_;
    std::vector<double> training_y_;
    std::size_t          observations_{0};
};

} // namespace datamunge::stats
