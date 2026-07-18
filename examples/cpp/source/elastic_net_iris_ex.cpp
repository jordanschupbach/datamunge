#include <datamunge/datasets/datasets.hpp>
#include <datamunge/stats/stats.hpp>

#include <iostream>

using datamunge::stats::ElasticNet;
using datamunge::stats::ElasticNetOptions;
using datamunge::stats::Lasso;
using datamunge::stats::Ridge;

namespace {
constexpr const char* kFormula = "Petal.Length ~ Sepal.Length + Sepal.Width + Petal.Width + Sepal.Length:Petal.Width";
}

int main() {
    const auto iris = datamunge::datasets::iris();
    std::cout << "iris: " << iris.nrows() << " rows x " << iris.ncols() << " cols\n";
    std::cout << "formula: " << kFormula << "\n\n";

    std::cout << "=================== Ridge ===================\n";
    Ridge ridge(iris, kFormula);
    ridge.print_summary(std::cout);

    std::cout << "\n=================== Lasso ===================\n";
    Lasso lasso(iris, kFormula);
    lasso.print_summary(std::cout);

    std::cout << "\n================= Elastic Net =================\n";
    ElasticNetOptions en_options;
    en_options.alpha = 0.5;
    ElasticNet elastic(iris, kFormula, en_options);
    elastic.print_summary(std::cout);

    std::cout << "\nSaved figures showing how each model's coefficients respond to the "
                 "regularization strength, and the cross-validation curve used to pick it:\n";

    ridge.plot_coefficient_path().save("elastic_net_ridge_path.svg");
    ridge.plot_cv_curve().save("elastic_net_ridge_cv.svg");
    std::cout << "  ridge:       elastic_net_ridge_path.svg, elastic_net_ridge_cv.svg\n";

    lasso.plot_coefficient_path().save("elastic_net_lasso_path.svg");
    lasso.plot_cv_curve().save("elastic_net_lasso_cv.svg");
    std::cout << "  lasso:       elastic_net_lasso_path.svg, elastic_net_lasso_cv.svg\n";

    elastic.plot_coefficient_path().save("elastic_net_elasticnet_path.svg");
    elastic.plot_cv_curve().save("elastic_net_elasticnet_cv.svg");
    std::cout << "  elastic net: elastic_net_elasticnet_path.svg, elastic_net_elasticnet_cv.svg\n";

    return 0;
}
