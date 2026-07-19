module app;

import std.stdio : writeln;
import std.random : Random, uniform;
import std.math : sqrt, log, cos, sin, PI;
import std.conv : to;
import std.array : join;
import std.algorithm : map;
import datamunge;

DVector dv(double[] t) {
  auto v = new DVector();
  foreach (x; t) v.push_back(x);
  return v;
}

double gauss(ref Random rng) {
  double u1 = uniform(0.0, 1.0, rng);
  double u2 = uniform(0.0, 1.0, rng);
  return sqrt(-2.0 * log(u1)) * cos(2.0 * PI * u2);
}

string dvStr(DVector v) {
  string[] parts;
  for (size_t i = 0; i < v.size(); i++) parts ~= to!string(v[i]);
  return parts.join(", ");
}

void main() {
  writeln("=================== ARIMA(1,1,1) on a simulated random walk with drift ===================");
  auto rng = Random(42);
  double[] y;
  double level = 100.0;
  double prev_shock = 0.0;
  for (int i = 0; i < 150; i++) {
    double shock = gauss(rng);
    level += 0.3 + 0.4 * prev_shock + shock;
    y ~= level;
    prev_shock = shock;
  }

  auto options = new ARIMAOptions();
  options.p = 1;
  options.d = 1;
  options.q = 1;
  options.de_population_size = 80;
  options.de_max_generations = 400;
  auto model = new ARIMA(dv(y), options);

  writeln("AR coefficient: ", model.ar_coefficients()[0]);
  writeln("MA coefficient: ", model.ma_coefficients()[0]);
  writeln("sigma^2: ", model.sigma2(), ", AIC: ", model.aic(), ", BIC: ", model.bic());

  auto pair = model.forecast_with_intervals(6);
  writeln("6-step forecast: ", dvStr(pair.first()));
  writeln("forecast std. errors: ", dvStr(pair.second()));

  writeln("\n=================== SARIMA(1,0,0)(1,1,0)_12 on a seasonal series ===================");
  auto rng2 = Random(7);
  double[] s;
  double prev = 0.0;
  for (int i = 0; i < 120; i++) {
    prev = 0.5 * prev + gauss(rng2);
    s ~= 20.0 + 0.2 * i + 5.0 * sin(2.0 * PI * i / 12.0) + prev;
  }

  auto options2 = new ARIMAOptions();
  options2.p = 1;
  options2.seasonal_p = 1;
  options2.seasonal_d = 1;
  options2.seasonal_period = 12;
  options2.de_population_size = 100;
  options2.de_max_generations = 500;
  auto model2 = new ARIMA(dv(s), options2);
  writeln("AR coefficient: ", model2.ar_coefficients()[0], ", seasonal AR coefficient: ", model2.seasonal_ar_coefficients()[0]);
  writeln("12-step forecast: ", dvStr(model2.forecast(12)));

  writeln("\n=================== Exponential smoothing: Holt-Winters on retail-style seasonal data ===================");
  double[] seasonal_shape = [0.8, 0.75, 0.9, 0.95, 1.0, 1.05, 1.1, 1.05, 1.0, 1.1, 1.3, 1.6];
  auto rng3 = Random(11);
  double[] y2;
  for (int i = 0; i < 48; i++) {
    double lvl = 100.0 + 2.0 * i;
    y2 ~= lvl * seasonal_shape[i % 12] + gauss(rng3) * 3.0;
  }

  auto es_options = new ExponentialSmoothingOptions();
  es_options.trend = TrendType.Additive;
  es_options.seasonal = SeasonalType.Multiplicative;
  es_options.seasonal_period = 12;
  es_options.de_population_size = 60;
  es_options.de_max_generations = 300;
  auto es_model = new ExponentialSmoothing(dv(y2), es_options);
  writeln("alpha=", es_model.alpha(), " beta=", es_model.beta(), " gamma=", es_model.gamma());
  writeln("sigma^2: ", es_model.sigma2(), ", AIC: ", es_model.aic());
  writeln("12-month forecast: ", dvStr(es_model.forecast(12)));

  writeln("\n=================== Simple exponential smoothing vs. Holt's linear trend ===================");
  double[] flat = [50.2, 49.8, 50.5, 49.6, 50.1, 50.3, 49.9, 50.0, 50.4, 49.7];

  auto ses = new ExponentialSmoothing(dv(flat), new ExponentialSmoothingOptions());
  writeln("SES alpha= ", ses.alpha());
  writeln("SES 5-step forecast: ", dvStr(ses.forecast(5)));

  auto holt_options = new ExponentialSmoothingOptions();
  holt_options.trend = TrendType.Additive;
  auto holt = new ExponentialSmoothing(dv(flat), holt_options);
  writeln("Holt alpha=", holt.alpha(), " beta=", holt.beta());
  writeln("Holt 5-step forecast: ", dvStr(holt.forecast(5)));
}
