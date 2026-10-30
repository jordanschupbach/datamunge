#include <datamunge/datasets/datasets.hpp>
#include <datamunge/dstruct/dstruct.hpp>
#include <datamunge/stats/lmm.hpp>

#include <iomanip>
#include <iostream>
#include <random>

using datamunge::dstruct::DataFrame;
using datamunge::stats::LMM;

int main() {
    std::cout << std::fixed << std::setprecision(4);

    std::cout << "=================== Random intercept on a real dataset (penguins) ===================\n";
    const auto penguins = datamunge::datasets::penguins();
    LMM species_model(penguins, "body_mass_g ~ flipper_length_mm + bill_length_mm + (1 | species)");
    species_model.print_summary();

    std::cout << "\n=================== Random intercept + slope on a simulated multi-school dataset "
                 "===================\n";
    // 30 schools, ~25 students each: score depends on study_hours, but both the baseline
    // score and the return on an extra hour of study vary by school -- the textbook case
    // for a random-intercept-and-slope model.
    std::mt19937_64 rng(2024);
    constexpr int n_schools = 30;
    std::normal_distribution<double> intercept_effect(0.0, 6.0);  // true random-intercept SD
    std::normal_distribution<double> slope_effect(0.0, 1.2);      // true random-slope SD
    std::normal_distribution<double> noise(0.0, 4.0);             // true residual SD
    std::uniform_real_distribution<double> hours_dist(0.0, 10.0);

    std::vector<double> school_intercept(n_schools), school_slope(n_schools);
    for (int s = 0; s < n_schools; ++s) {
        school_intercept[s] = intercept_effect(rng);
        school_slope[s] = slope_effect(rng);
    }

    constexpr double true_intercept = 60.0, true_slope = 3.0;
    std::vector<double> school, study_hours, score;
    std::uniform_int_distribution<int> group_size_dist(15, 35);
    for (int s = 0; s < n_schools; ++s) {
        const int n_students = group_size_dist(rng);
        for (int i = 0; i < n_students; ++i) {
            const double hours = hours_dist(rng);
            const double s_val = true_intercept + school_intercept[s] +
                                  (true_slope + school_slope[s]) * hours + noise(rng);
            school.push_back(static_cast<double>(s));
            study_hours.push_back(hours);
            score.push_back(s_val);
        }
    }
    DataFrame df;
    df.add_column("school", school);
    df.add_column("study_hours", study_hours);
    df.add_column("score", score);

    LMM model(df, "score ~ study_hours + (1 + study_hours | school)");
    model.print_summary();

    std::cout << "\nTrue generating values: intercept=" << true_intercept << ", slope=" << true_slope
              << ", random-intercept SD=6.0, random-slope SD=1.2, residual SD=4.0\n";

    std::cout << "\n--- Best Linear Unbiased Predictors (BLUPs) for a few schools ---\n";
    for (const std::size_t idx : {std::size_t{0}, std::size_t{1}, std::size_t{2}}) {
        const auto& re = model.random_effects()[idx];
        std::cout << "school " << model.group_labels()[idx] << ": intercept shift=" << re[0]
                  << ", slope shift=" << re[1] << "\n";
    }

    std::cout << "\n--- Prediction: population-level vs. school-adjusted ---\n";
    DataFrame newdata_population;
    newdata_population.add_column("study_hours", std::vector<double>{5.0});
    DataFrame newdata_school0;
    newdata_school0.add_column("study_hours", std::vector<double>{5.0});
    newdata_school0.add_column("school", std::vector<double>{0.0});
    std::cout << "5 study hours, unseen school:      " << model.predict(newdata_population)[0]
              << " (fixed effects only)\n";
    std::cout << "5 study hours, school 0 (known):    " << model.predict(newdata_school0)[0]
              << " (fixed effects + school 0's BLUP)\n";

    return 0;
}
