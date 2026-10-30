#include <datamunge/datasets/datasets.hpp>
#include <datamunge/dstruct/dstruct.hpp>
#include <datamunge/stats/glmm.hpp>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>

using datamunge::dstruct::DataFrame;
using datamunge::stats::GLMM;
using datamunge::stats::GLMMFamily;
using datamunge::stats::GLMMOptions;

int main() {
    std::cout << std::fixed << std::setprecision(4);

    std::cout << "=================== Binomial (logistic) mixed model on a real dataset (penguins) "
                 "===================\n";
    // Predicting sex from body mass, grouped by island: unlike species (which is almost
    // perfectly confounded with island in this dataset -- Gentoo penguins occur on Biscoe
    // island only), sex is not tied to location, so this is a well-behaved fit.
    const auto penguins = datamunge::datasets::penguins();
    std::vector<double> is_male, body_mass;
    std::vector<std::string> island;
    for (std::size_t i = 0; i < penguins.nrows(); ++i) {
        if (penguins.is_null("sex", i) || penguins.is_null("body_mass_g", i) || penguins.is_null("island", i)) continue;
        is_male.push_back(penguins.string_at("sex", i) == "male" ? 1.0 : 0.0);
        body_mass.push_back(penguins.double_at("body_mass_g", i));
        island.push_back(penguins.string_at("island", i));
    }
    DataFrame sex_df;
    sex_df.add_column("is_male", is_male);
    sex_df.add_column("body_mass_g", body_mass);
    sex_df.add_column("island", island);

    GLMMOptions binomial_options;
    binomial_options.family = GLMMFamily::Binomial;
    GLMM sex_model(sex_df, "is_male ~ body_mass_g + (1 | island)", binomial_options);
    sex_model.print_summary();

    std::cout << "\n=================== Poisson mixed model on simulated multi-site count data "
                 "===================\n";
    // 25 stores, ~20 days each: daily visit counts depend on a promo intensity score, but
    // both the baseline traffic and the promo's effectiveness vary by store.
    std::mt19937_64 rng(4242);
    constexpr int n_stores = 25;
    std::normal_distribution<double> store_intercept(0.0, 0.4); // true random-intercept SD (log scale)
    std::uniform_real_distribution<double> promo_dist(0.0, 3.0);

    std::vector<double> store_effect(n_stores);
    for (auto& e : store_effect) e = store_intercept(rng);

    constexpr double true_intercept = 2.0, true_slope = 0.3;
    std::vector<double> store, promo, visits;
    std::uniform_int_distribution<int> days_dist(15, 25);
    for (int s = 0; s < n_stores; ++s) {
        const int n_days = days_dist(rng);
        for (int d = 0; d < n_days; ++d) {
            const double promo_intensity = promo_dist(rng);
            const double lambda = std::exp(true_intercept + store_effect[s] + true_slope * promo_intensity);
            std::poisson_distribution<int> pois(lambda);
            store.push_back(static_cast<double>(s));
            promo.push_back(promo_intensity);
            visits.push_back(static_cast<double>(pois(rng)));
        }
    }
    DataFrame df;
    df.add_column("store", store);
    df.add_column("promo", promo);
    df.add_column("visits", visits);

    GLMMOptions poisson_options;
    poisson_options.family = GLMMFamily::Poisson;
    GLMM store_model(df, "visits ~ promo + (1 | store)", poisson_options);
    store_model.print_summary();

    std::cout << "\nTrue generating values: intercept=" << true_intercept << ", slope=" << true_slope
              << ", random-intercept SD (log scale)=0.4\n";

    std::cout << "\n--- BLUPs for a few stores ---\n";
    for (const std::size_t idx : {std::size_t{0}, std::size_t{1}, std::size_t{2}}) {
        std::cout << "store " << store_model.group_labels()[idx] << ": intercept shift=" << store_model.random_effects()[idx][0]
                  << "\n";
    }

    std::cout << "\n--- Prediction: population-level vs. store-adjusted ---\n";
    DataFrame newdata_population;
    newdata_population.add_column("promo", std::vector<double>{1.5});
    DataFrame newdata_store0;
    newdata_store0.add_column("promo", std::vector<double>{1.5});
    newdata_store0.add_column("store", std::vector<double>{0.0});
    std::cout << "promo=1.5, unseen store:   " << store_model.predict(newdata_population)[0] << " expected visits (fixed effects only)\n";
    std::cout << "promo=1.5, store 0 (known): " << store_model.predict(newdata_store0)[0]
              << " expected visits (fixed effects + store 0's BLUP)\n";

    return 0;
}
