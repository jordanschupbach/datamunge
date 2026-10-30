#include <datamunge/datasets/datasets.hpp>
#include <datamunge/dstruct/dstruct.hpp>
#include <datamunge/stats/stats.hpp>

#include <iomanip>
#include <iostream>

using datamunge::dstruct::DataFrame;
using datamunge::stats::GaussianProcessRegression;

namespace {
constexpr const char* kFormula = "Petal.Length ~ Petal.Width";
}

int main() {
    const auto iris = datamunge::datasets::iris();
    std::cout << "iris: " << iris.nrows() << " rows x " << iris.ncols() << " cols\n";
    std::cout << "formula: " << kFormula << "\n\n";

    // Both the length scale and the noise ratio are auto-selected by
    // maximizing the exact log marginal likelihood.
    GaussianProcessRegression model(iris, kFormula);
    model.print_summary(std::cout);

    model.plot_fit(iris).save("gpr_iris_fit.svg");
    model.plot_length_scale_profile().save("gpr_iris_length_scale_profile.svg");
    std::cout << "\nSaved gpr_iris_fit.svg and gpr_iris_length_scale_profile.svg\n";

    // Unlike every other regressor built this session, a GP gives a
    // genuine posterior confidence interval at every point -- including
    // far outside the training data, where it should widen substantially
    // as the model's uncertainty grows.
    DataFrame query;
    query.add_column("Petal.Width", std::vector<double>{0.2, 1.3, 2.5, 10.0});
    const auto detail = model.predict_detail(query);
    std::cout << "\nPredictions with 95% confidence intervals:\n";
    std::cout << std::setw(14) << "Petal.Width" << std::setw(10) << "fit" << std::setw(10) << "se" << std::setw(10)
               << "lwr" << std::setw(10) << "upr" << "\n";
    for (std::size_t i = 0; i < detail.fit.size(); ++i) {
        std::cout << std::setw(14) << query.double_at("Petal.Width", i) << std::setw(10) << detail.fit[i]
                   << std::setw(10) << detail.se_fit[i] << std::setw(10) << detail.lower[i] << std::setw(10)
                   << detail.upper[i] << "\n";
    }
    std::cout << "(Petal.Width=10.0 is far outside the training range [0.1, 2.5] -- note how much wider its\n"
                 " interval is than the in-range predictions.)\n";

    return 0;
}
