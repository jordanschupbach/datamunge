#include <datamunge/bayes/bayes.hpp>
#include <datamunge/datasets/datasets.hpp>
#include <datamunge/dstruct/dstruct.hpp>
#include <datamunge/stats/glm.hpp>

#include <cmath>
#include <iomanip>
#include <iostream>

using datamunge::autodiff::Tape;
using datamunge::autodiff::Var;
using datamunge::bayes::AutodiffModel;
using datamunge::bayes::HMC;
using datamunge::bayes::HMCOptions;
using datamunge::bayes::MAP;
using datamunge::bayes::NUTS;
using datamunge::bayes::NUTSOptions;

using datamunge::bayes::bernoulli_logit_lpmf;
using datamunge::bayes::normal_lpdf;

namespace {

void print_vec(const std::vector<double>& v) {
    std::cout << "[";
    for (std::size_t i = 0; i < v.size(); ++i) std::cout << (i ? ", " : "") << v[i];
    std::cout << "]";
}

double mean_of(const std::vector<std::vector<double>>& samples, std::size_t dim) {
    double s = 0.0;
    for (const auto& row : samples) s += row[dim];
    return s / static_cast<double>(samples.size());
}

double sd_of(const std::vector<std::vector<double>>& samples, std::size_t dim, double mean) {
    double s = 0.0;
    for (const auto& row : samples) {
        const double d = row[dim] - mean;
        s += d * d;
    }
    return std::sqrt(s / static_cast<double>(samples.size() - 1));
}

} // namespace

int main() {
    std::cout << std::fixed << std::setprecision(4);

    std::cout << "=================== Normal-Normal conjugate model: MAP, HMC, NUTS vs. the exact posterior "
                 "===================\n";
    // y_i ~ Normal(mu, sigma) iid, mu ~ Normal(mu0, tau0) -- a model with a known, closed-form
    // posterior, so every inference method here can be checked directly against the truth.
    const std::vector<double> y{2.1, 1.8, 2.5, 2.0, 1.9, 2.3, 2.2, 1.7, 2.4, 1.95};
    constexpr double sigma = 1.0, mu0 = 0.0, tau0 = 5.0;

    double sum_y = 0.0;
    for (const double v : y) sum_y += v;
    const double n = static_cast<double>(y.size());
    const double precision_post = 1.0 / (tau0 * tau0) + n / (sigma * sigma);
    const double exact_mean = (mu0 / (tau0 * tau0) + sum_y / (sigma * sigma)) / precision_post;
    const double exact_sd = std::sqrt(1.0 / precision_post);
    std::cout << "Exact posterior: N(" << exact_mean << ", " << exact_sd << "^2)\n\n";

    AutodiffModel conjugate_model([&](Tape&, const std::vector<Var>& params) {
        Var lp = normal_lpdf(params[0], mu0, tau0);
        for (const double yi : y) lp = lp + normal_lpdf(yi, params[0], sigma);
        return lp;
    });

    {
        std::vector<double> coords{0.0};
        const double log_post = MAP().optimize(conjugate_model, coords);
        std::cout << "MAP:  mu = " << coords[0] << " (log-posterior = " << log_post << ")\n";
    }
    {
        HMCOptions options;
        options.num_warmup = 1000;
        options.num_samples = 4000;
        options.num_leapfrog_steps = 15;
        options.initial_step_size = 0.3;
        const auto result = HMC(options).sample(conjugate_model, {0.0});
        const double m = mean_of(result.samples, 0);
        const double s = sd_of(result.samples, 0, m);
        std::cout << "HMC:  mu ~ N(" << m << ", " << s << "^2), accept rate = " << result.accept_rate
                  << ", step size = " << result.final_step_size << "\n";
    }
    {
        NUTSOptions options;
        options.num_warmup = 1000;
        options.num_samples = 4000;
        options.initial_step_size = 0.3;
        const auto result = NUTS(options).sample(conjugate_model, {0.0});
        const double m = mean_of(result.samples, 0);
        const double s = sd_of(result.samples, 0, m);
        std::cout << "NUTS: mu ~ N(" << m << ", " << s << "^2), accept rate = " << result.accept_rate
                  << ", step size = " << result.final_step_size << ", divergences = " << result.num_divergences << "\n";
    }

    std::cout << "\n=================== Bayesian logistic regression vs. GLM's MLE (iris) ===================\n";
    const auto iris = datamunge::datasets::iris();
    std::vector<double> is_virginica, petal_length;
    for (std::size_t i = 0; i < iris.nrows(); ++i) {
        const auto species = iris.string_at("Species", i);
        if (species != "versicolor" && species != "virginica") continue;
        is_virginica.push_back(species == "virginica" ? 1.0 : 0.0);
        petal_length.push_back(iris.double_at("Petal.Length", i));
    }
    datamunge::dstruct::DataFrame df;
    df.add_column("Petal.Length", petal_length);
    df.add_column("is_virginica", is_virginica);
    datamunge::stats::GLMOptions glm_options;
    glm_options.family = datamunge::stats::GLMFamily::Binomial;
    const datamunge::stats::GLM glm(df, "is_virginica ~ Petal.Length", glm_options);
    std::cout << "GLM MLE:        ";
    print_vec(glm.coefficients());
    std::cout << "\n";

    AutodiffModel logistic_model([&](Tape&, const std::vector<Var>& params) {
        Var lp = normal_lpdf(params[0], 0.0, 10.0) + normal_lpdf(params[1], 0.0, 10.0);
        for (std::size_t i = 0; i < petal_length.size(); ++i) {
            const Var eta = params[0] + params[1] * petal_length[i];
            lp = lp + bernoulli_logit_lpmf(is_virginica[i], eta);
        }
        return lp;
    });
    {
        std::vector<double> coords{0.0, 0.0};
        MAP().optimize(logistic_model, coords);
        std::cout << "Bayes MAP:      ";
        print_vec(coords);
        std::cout << " (weak Normal(0, 10) priors)\n";
    }
    {
        NUTSOptions options;
        options.num_warmup = 1000;
        options.num_samples = 3000;
        options.initial_step_size = 0.05;
        const auto result = NUTS(options).sample(logistic_model, {0.0, 0.0});
        std::cout << "Bayes NUTS mean: [" << mean_of(result.samples, 0) << ", " << mean_of(result.samples, 1)
                  << "] (posterior mean, accept rate = " << result.accept_rate << ")\n";
    }
    std::cout << "(Petal.Length nearly separates these two species, so the unregularized MLE inflates toward the\n"
                 " separating boundary; the weak Normal(0, 10) prior visibly pulls the Bayesian estimate back --\n"
                 " a real, expected difference, not a bug.)\n";

    return 0;
}
