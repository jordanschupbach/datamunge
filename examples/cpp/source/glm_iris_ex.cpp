#include <datamunge/datasets/datasets.hpp>
#include <datamunge/dstruct/dstruct.hpp>
#include <datamunge/stats/stats.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>

using datamunge::dstruct::DataFrame;
using datamunge::stats::GLM;
using datamunge::stats::GLMFamily;
using datamunge::stats::GLMOptions;
using datamunge::stats::GLMPredictionInterval;

int main() {
    const auto iris = datamunge::datasets::iris();

    // Logistic regression: versicolor vs virginica only -- setosa is
    // perfectly separable from the other two on these predictors, which
    // sends logistic regression's coefficients toward +/-infinity (a
    // genuine degeneracy of the method, not a bug) -- so this is the
    // well-behaved binary split for a demo.
    std::vector<double> is_virginica, petal_length, petal_width;
    for (std::size_t i = 0; i < iris.nrows(); ++i) {
        const auto species = iris.string_at("Species", i);
        if (species != "versicolor" && species != "virginica") continue;
        is_virginica.push_back(species == "virginica" ? 1.0 : 0.0);
        petal_length.push_back(iris.double_at("Petal.Length", i));
        petal_width.push_back(iris.double_at("Petal.Width", i));
    }
    DataFrame sub;
    sub.add_column("Petal.Length", petal_length);
    sub.add_column("Petal.Width", petal_width);
    sub.add_column("is_virginica", is_virginica);

    std::cout << "=================== Logistic regression (binomial, logit link) ===================\n";
    GLMOptions logit_options;
    logit_options.family = GLMFamily::Binomial;
    GLM logit(sub, "is_virginica ~ Petal.Length + Petal.Width", logit_options);
    logit.print_summary(std::cout);

    std::size_t correct = 0;
    const auto  fitted   = logit.fitted_values();
    for (std::size_t i = 0; i < is_virginica.size(); ++i)
        if ((fitted[i] >= 0.5) == (is_virginica[i] >= 0.5)) ++correct;
    std::cout << "\nResubstitution accuracy at 0.5 threshold: "
               << (100.0 * static_cast<double>(correct) / static_cast<double>(is_virginica.size())) << "%\n";

    logit.save_diagnostic_plots("glm_logistic_iris");
    std::cout << "\nSaved glm_logistic_iris_{residuals_vs_fitted,normal_qq,scale_location,residuals_vs_leverage}.svg\n";

    // Predicted-probability curve across Petal.Length, with Petal.Width
    // held at its mean -- the classic sigmoid shape of a fitted logistic
    // regression, with a 95% confidence band.
    double width_mean = 0.0;
    for (const double w : petal_width) width_mean += w;
    width_mean /= static_cast<double>(petal_width.size());

    constexpr std::size_t kGridN = 100;
    double pl_min = *std::min_element(petal_length.begin(), petal_length.end()) - 0.3;
    double pl_max = *std::max_element(petal_length.begin(), petal_length.end()) + 0.3;
    std::vector<double> grid_x(kGridN), grid_width(kGridN, width_mean);
    for (std::size_t i = 0; i < kGridN; ++i)
        grid_x[i] = pl_min + (pl_max - pl_min) * static_cast<double>(i) / static_cast<double>(kGridN - 1);
    DataFrame grid;
    grid.add_column("Petal.Length", grid_x);
    grid.add_column("Petal.Width", grid_width);
    const auto curve = logit.predict(grid, GLMPredictionInterval::Confidence);

    auto plot = datamunge::plot::ScatterPlot::create();
    std::vector<double> obs_x0, obs_y0, obs_x1, obs_y1;
    for (std::size_t i = 0; i < is_virginica.size(); ++i) {
        (is_virginica[i] > 0.5 ? obs_x1 : obs_x0).push_back(petal_length[i]);
        (is_virginica[i] > 0.5 ? obs_y1 : obs_y0).push_back(is_virginica[i]);
    }
    plot.points(obs_x0, obs_y0, "versicolor (0)", {37, 99, 235}, 5.0);
    plot.points(obs_x1, obs_y1, "virginica (1)", {220, 38, 38}, 5.0);
    plot.line(grid_x, curve.lower, "lower 95%", {252, 165, 165}, 1.5);
    plot.line(grid_x, curve.upper, "upper 95%", {252, 165, 165}, 1.5);
    plot.line(grid_x, curve.fit, "P(virginica)", {124, 58, 237}, 2.5);
    plot.title("Logistic Regression Fit").x_label("Petal.Length").y_label("P(virginica)");
    plot.save("glm_logistic_iris_curve.svg");
    std::cout << "Saved glm_logistic_iris_curve.svg\n";

    // Poisson regression, for contrast: same IRLS engine, different family/link.
    std::cout << "\n=================== Poisson regression (log link) ===================\n";
    auto count_data = iris;
    std::vector<double> count(count_data.nrows());
    for (std::size_t i = 0; i < count_data.nrows(); ++i)
        count[i] = std::nearbyint(count_data.double_at("Sepal.Length", i));
    count_data.add_column("count", count);

    GLMOptions poisson_options;
    poisson_options.family = GLMFamily::Poisson;
    GLM poisson(count_data, "count ~ Sepal.Width + Petal.Length", poisson_options);
    poisson.print_summary(std::cout);

    return 0;
}
