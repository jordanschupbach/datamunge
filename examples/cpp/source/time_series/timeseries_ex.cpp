#include <datamunge/stats/arima.hpp>
#include <datamunge/stats/exponential_smoothing.hpp>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

using datamunge::stats::ARIMA;
using datamunge::stats::ARIMAOptions;
using datamunge::stats::ExponentialSmoothing;
using datamunge::stats::ExponentialSmoothingOptions;
using datamunge::stats::SeasonalType;
using datamunge::stats::TrendType;

namespace {

void print_vector(const std::string& label, const std::vector<double>& v) {
    std::cout << label << ": ";
    for (const double x : v) std::cout << x << " ";
    std::cout << "\n";
}

} // namespace

int main() {
    std::cout << std::fixed << std::setprecision(4);

    std::cout << "=================== ARIMA(1,1,1) on a simulated random walk with drift ===================\n";
    {
        std::mt19937_64 rng(42);
        std::normal_distribution<double> noise(0.0, 1.0);
        std::vector<double> y(150);
        double level = 100.0;
        double prev_shock = 0.0;
        for (std::size_t i = 0; i < y.size(); ++i) {
            const double shock = noise(rng);
            level += 0.3 + 0.4 * prev_shock + shock; // integrated ARMA(0,1) increments with drift
            y[i] = level;
            prev_shock = shock;
        }

        ARIMAOptions options;
        options.p = 1;
        options.d = 1;
        options.q = 1;
        options.de_population_size = 80;
        options.de_max_generations = 400;
        const ARIMA model(y, options);

        std::cout << "AR coefficient: " << model.ar_coefficients()[0] << "\n";
        std::cout << "MA coefficient: " << model.ma_coefficients()[0] << "\n";
        std::cout << "sigma^2: " << model.sigma2() << ", AIC: " << model.aic() << ", BIC: " << model.bic() << "\n";

        const auto [forecast, se] = model.forecast_with_intervals(6);
        print_vector("6-step forecast", forecast);
        print_vector("forecast std. errors", se);
    }

    std::cout << "\n=================== SARIMA(1,0,0)(1,1,0)_12 on a seasonal series ===================\n";
    {
        std::mt19937_64 rng(7);
        std::normal_distribution<double> noise(0.0, 1.0);
        std::vector<double> s(120);
        double prev = 0.0;
        for (std::size_t i = 0; i < s.size(); ++i) {
            prev = 0.5 * prev + noise(rng);
            s[i] = 20.0 + 0.2 * static_cast<double>(i) + 5.0 * std::sin(2.0 * 3.14159265358979 * static_cast<double>(i) / 12.0) + prev;
        }

        ARIMAOptions options;
        options.p = 1;
        options.seasonal_p = 1;
        options.seasonal_d = 1;
        options.seasonal_period = 12;
        options.de_population_size = 100;
        options.de_max_generations = 500;
        const ARIMA model(s, options);

        std::cout << "AR coefficient: " << model.ar_coefficients()[0]
                   << ", seasonal AR coefficient: " << model.seasonal_ar_coefficients()[0] << "\n";
        print_vector("12-step forecast", model.forecast(12));
    }

    std::cout << "\n=================== Exponential smoothing: Holt-Winters on retail-style seasonal data "
                 "===================\n";
    {
        // A synthetic monthly series with an upward trend and a repeating 12-month seasonal
        // shape, roughly evoking retail sales with a holiday spike.
        const std::vector<double> seasonal_shape{0.8, 0.75, 0.9, 0.95, 1.0, 1.05, 1.1, 1.05, 1.0, 1.1, 1.3, 1.6};
        std::mt19937_64 rng(11);
        std::normal_distribution<double> noise(0.0, 3.0);
        std::vector<double> y(48);
        for (std::size_t i = 0; i < y.size(); ++i) {
            const double level = 100.0 + 2.0 * static_cast<double>(i);
            y[i] = level * seasonal_shape[i % 12] + noise(rng);
        }

        ExponentialSmoothingOptions options;
        options.trend = TrendType::Additive;
        options.seasonal = SeasonalType::Multiplicative;
        options.seasonal_period = 12;
        options.de_population_size = 60;
        options.de_max_generations = 300;
        const ExponentialSmoothing model(y, options);

        std::cout << "alpha=" << model.alpha() << " beta=" << model.beta() << " gamma=" << model.gamma() << "\n";
        std::cout << "sigma^2: " << model.sigma2() << ", AIC: " << model.aic() << "\n";
        print_vector("12-month forecast", model.forecast(12));
    }

    std::cout << "\n=================== Simple exponential smoothing vs. Holt's linear trend "
                 "===================\n";
    {
        // A short, noisy-but-flat series: simple exponential smoothing should suffice, and a
        // fitted trend model should recognize there is essentially none to extrapolate.
        const std::vector<double> flat{50.2, 49.8, 50.5, 49.6, 50.1, 50.3, 49.9, 50.0, 50.4, 49.7};

        ExponentialSmoothingOptions ses_options;
        const ExponentialSmoothing ses(flat, ses_options);
        std::cout << "SES alpha=" << ses.alpha() << "\n";
        print_vector("SES 5-step forecast", ses.forecast(5));

        ExponentialSmoothingOptions holt_options;
        holt_options.trend = TrendType::Additive;
        const ExponentialSmoothing holt(flat, holt_options);
        std::cout << "Holt alpha=" << holt.alpha() << " beta=" << holt.beta() << "\n";
        print_vector("Holt 5-step forecast", holt.forecast(5));
    }

    return 0;
}
