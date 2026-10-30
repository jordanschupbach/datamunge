#include <datamunge/datasets/datasets.hpp>
#include <datamunge/dstruct/dstruct.hpp>
#include <datamunge/stats/anova_test.hpp>
#include <datamunge/stats/chi_squared_test.hpp>
#include <datamunge/stats/correlation_test.hpp>
#include <datamunge/stats/fisher_exact_test.hpp>
#include <datamunge/stats/ks_test.hpp>
#include <datamunge/stats/normality_test.hpp>
#include <datamunge/stats/proportion_test.hpp>
#include <datamunge/stats/t_test.hpp>
#include <datamunge/stats/variance_test.hpp>
#include <datamunge/stats/wilcoxon_test.hpp>

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace datamunge::stats;

namespace {

const char* alternative_name(Alternative a) {
    switch (a) {
        case Alternative::Less: return "less";
        case Alternative::Greater: return "greater";
        default: return "two.sided";
    }
}

void print_result(const std::string& label, const HypothesisTestResult& r) {
    std::cout << label << ": statistic=" << r.statistic;
    if (r.parameter1 != 0.0) std::cout << ", df1=" << r.parameter1;
    if (r.parameter2 != 0.0) std::cout << ", df2=" << r.parameter2;
    std::cout << ", p=" << r.p_value << " (" << alternative_name(r.alternative) << ")";
    if (r.has_conf_int) std::cout << ", CI=[" << r.conf_int_lower << ", " << r.conf_int_upper << "]";
    std::cout << " -- " << r.method << "\n";
}

std::vector<double> column_for_species(const datamunge::dstruct::DataFrame& iris, const std::string& column,
                                        const std::string& species) {
    std::vector<double> out;
    for (std::size_t i = 0; i < iris.nrows(); ++i) {
        if (iris.string_at("Species", i) == species) out.push_back(iris.double_at(column, i));
    }
    return out;
}

} // namespace

int main() {
    std::cout << std::fixed << std::setprecision(4);
    const auto iris = datamunge::datasets::iris();

    const auto setosa_petal = column_for_species(iris, "Petal.Length", "setosa");
    const auto versicolor_petal = column_for_species(iris, "Petal.Length", "versicolor");
    const auto virginica_petal = column_for_species(iris, "Petal.Length", "virginica");

    std::cout << "=================== t-tests: petal length, setosa vs. versicolor ===================\n";
    print_result("Welch two-sample t-test", t_test_two_sample(setosa_petal, versicolor_petal));
    print_result("Wilcoxon rank-sum test", wilcoxon_rank_sum_test(setosa_petal, versicolor_petal));

    std::cout << "\n=================== One-way ANOVA / Kruskal-Wallis across all 3 species ===================\n";
    std::vector<double> all_petal = setosa_petal;
    all_petal.insert(all_petal.end(), versicolor_petal.begin(), versicolor_petal.end());
    all_petal.insert(all_petal.end(), virginica_petal.begin(), virginica_petal.end());
    const std::vector<std::size_t> sizes{setosa_petal.size(), versicolor_petal.size(), virginica_petal.size()};
    print_result("One-way ANOVA", one_way_anova(all_petal, sizes));
    print_result("Kruskal-Wallis", kruskal_wallis_test(all_petal, sizes));

    std::cout << "\n=================== Correlation: sepal length vs. petal length ===================\n";
    std::vector<double> sepal_length, petal_length;
    for (std::size_t i = 0; i < iris.nrows(); ++i) {
        sepal_length.push_back(iris.double_at("Sepal.Length", i));
        petal_length.push_back(iris.double_at("Petal.Length", i));
    }
    print_result("Pearson correlation", pearson_correlation_test(sepal_length, petal_length));
    print_result("Spearman correlation", spearman_correlation_test(sepal_length, petal_length));

    std::cout << "\n=================== F-test: petal length variance, setosa vs. virginica ===================\n";
    print_result("F test", f_test_variance(setosa_petal, virginica_petal));

    std::cout << "\n=================== Normality: is sepal length normally distributed within setosa? "
                 "===================\n";
    const auto setosa_sepal = column_for_species(iris, "Sepal.Length", "setosa");
    print_result("Shapiro-Francia", shapiro_francia_test(setosa_sepal));
    print_result("KS vs. fitted normal", ks_test_one_sample_normal(setosa_sepal, 5.006, 0.3525));

    std::cout << "\n=================== Chi-squared / Fisher: is petal length \"long\" independent of species? "
                 "===================\n";
    // Discretize: "long" petal (above the overall median) vs. species, for setosa vs.
    // versicolor only (a habitat-style 2x2 table this small comparison naturally forms).
    double median_all = 0.0;
    {
        std::vector<double> sorted = all_petal;
        std::sort(sorted.begin(), sorted.end());
        median_all = sorted[sorted.size() / 2];
    }
    std::size_t setosa_long = 0, setosa_short = 0, versicolor_long = 0, versicolor_short = 0;
    for (const double v : setosa_petal) (v > median_all ? setosa_long : setosa_short)++;
    for (const double v : versicolor_petal) (v > median_all ? versicolor_long : versicolor_short)++;
    std::cout << "table: setosa=[" << setosa_long << "," << setosa_short << "] versicolor=[" << versicolor_long
              << "," << versicolor_short << "]\n";
    const std::vector<double> table{static_cast<double>(setosa_long), static_cast<double>(setosa_short),
                                     static_cast<double>(versicolor_long), static_cast<double>(versicolor_short)};
    print_result("Chi-squared independence", chi_squared_test_independence(table, 2, 2));
    print_result("Fisher's exact test", fisher_exact_test_2x2(setosa_long, setosa_short, versicolor_long, versicolor_short));

    std::cout << "\n=================== Proportion / binomial: fraction of \"long\" petals overall "
                 "===================\n";
    std::size_t long_count = 0;
    for (const double v : all_petal) {
        if (v > median_all) ++long_count;
    }
    print_result("One-sample proportion test (vs 0.5)", proportion_test_one_sample(long_count, all_petal.size(), 0.5));
    print_result("Exact binomial test (vs 0.5)", binomial_test(long_count, all_petal.size(), 0.5));

    return 0;
}
