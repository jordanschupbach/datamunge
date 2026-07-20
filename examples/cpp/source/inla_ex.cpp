#include <datamunge/datasets/datasets.hpp>
#include <datamunge/dstruct/dstruct.hpp>
#include <datamunge/stats/inla_mixed_model.hpp>
#include <datamunge/stats/p_adjust.hpp>

#include <iomanip>
#include <iostream>

using datamunge::dstruct::DataFrame;
using datamunge::stats::INLAMixedModel;
using datamunge::stats::INLAMixedModelFamily;
using datamunge::stats::INLAMixedModelOptions;
using datamunge::stats::p_adjust;
using datamunge::stats::PAdjustMethod;

int main() {
    std::cout << std::fixed << std::setprecision(4);

    std::cout << "=================== INLAMixedModel: Bayesian mixed model via Integrated Nested Laplace "
                 "Approximation ===================\n";
    // Same model glmm_ex.cpp fits by penalized quasi-likelihood: predict sex from body mass,
    // with a random intercept per island. INLA instead returns a genuine posterior (including
    // for the random-effect variance itself), not just a point estimate + asymptotic SE.
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

    INLAMixedModelOptions options;
    options.family = INLAMixedModelFamily::Binomial;
    INLAMixedModel model(sex_df, "is_male ~ body_mass_g + (1 | island)", options);
    model.print_summary();

    std::cout << "\nlog marginal likelihood: " << model.log_marginal_likelihood() << "\n";
    std::cout << "random-effect (island) standard deviation: " << model.random_effect_std_devs()[0] << "\n\n";

    DataFrame newdata;
    newdata.add_column("body_mass_g", std::vector<double>{3500.0, 4500.0, 5500.0});
    newdata.add_column("island", std::vector<std::string>{"Biscoe", "Dream", "Torgersen"});
    const auto predictions = model.predict(newdata);
    std::cout << "P(male) predictions:\n";
    for (std::size_t i = 0; i < predictions.size(); ++i)
        std::cout << "  body_mass_g=" << newdata.double_at("body_mass_g", i) << ", island=" << newdata.string_at("island", i)
                   << " -> " << predictions[i] << "\n";

    std::cout << "\n=================== p_adjust: multiple-comparison correction ===================\n";
    // 10 p-values, 3 of which are genuinely significant (< 0.01); the rest are noise.
    const std::vector<double> raw_p = {0.001, 0.004, 0.009, 0.03, 0.12, 0.18, 0.35, 0.51, 0.72, 0.98};
    std::cout << std::setw(10) << "raw" << std::setw(12) << "bonferroni" << std::setw(10) << "holm" << std::setw(10) << "BH"
              << "\n";
    const auto bonferroni = p_adjust(raw_p, PAdjustMethod::Bonferroni);
    const auto holm = p_adjust(raw_p, PAdjustMethod::Holm);
    const auto bh = p_adjust(raw_p, PAdjustMethod::BH);
    for (std::size_t i = 0; i < raw_p.size(); ++i)
        std::cout << std::setw(10) << raw_p[i] << std::setw(12) << bonferroni[i] << std::setw(10) << holm[i] << std::setw(10)
                   << bh[i] << "\n";

    return 0;
}
