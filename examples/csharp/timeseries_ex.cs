using System;
using System.Collections.Generic;
using System.Linq;

class Program {
  static Random rng = new Random();

  static double Randn() {
    double u1 = rng.NextDouble();
    double u2 = rng.NextDouble();
    return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
  }

  static string CellJoin(DVector v) {
    var parts = new List<string>();
    for (int i = 0; i < v.Count; i++) parts.Add(v[i].ToString());
    return string.Join(", ", parts);
  }

  static void Main() {
    Console.WriteLine("=================== ARIMA(1,1,1) on a simulated random walk with drift ===================");
    var y = new List<double>();
    double level = 100.0;
    double prevShock = 0.0;
    for (int i = 0; i < 150; i++) {
      double shock = Randn();
      level = level + 0.3 + 0.4 * prevShock + shock;
      y.Add(level);
      prevShock = shock;
    }

    var options = new ARIMAOptions();
    options.p = 1;
    options.d = 1;
    options.q = 1;
    options.de_population_size = 80;
    options.de_max_generations = 400;
    var model = new ARIMA(new DVector(y), options);

    var ar = model.ar_coefficients();
    var ma = model.ma_coefficients();
    Console.WriteLine($"AR coefficient: {ar[0]}");
    Console.WriteLine($"MA coefficient: {ma[0]}");
    Console.WriteLine($"sigma^2: {model.sigma2()}, AIC: {model.aic()}, BIC: {model.bic()}");

    var pair = model.forecast_with_intervals(6);
    Console.WriteLine($"6-step forecast: {CellJoin(pair.first)}");
    Console.WriteLine($"forecast std. errors: {CellJoin(pair.second)}");

    Console.WriteLine("\n=================== SARIMA(1,0,0)(1,1,0)_12 on a seasonal series ===================");
    var s = new List<double>();
    double prev = 0.0;
    for (int i = 0; i < 120; i++) {
      prev = 0.5 * prev + Randn();
      s.Add(20.0 + 0.2 * i + 5.0 * Math.Sin((2.0 * Math.PI * i) / 12.0) + prev);
    }

    var options2 = new ARIMAOptions();
    options2.p = 1;
    options2.seasonal_p = 1;
    options2.seasonal_d = 1;
    options2.seasonal_period = 12;
    options2.de_population_size = 100;
    options2.de_max_generations = 500;
    var model2 = new ARIMA(new DVector(s), options2);
    var ar2 = model2.ar_coefficients();
    var sar2 = model2.seasonal_ar_coefficients();
    Console.WriteLine($"AR coefficient: {ar2[0]}, seasonal AR coefficient: {sar2[0]}");
    Console.WriteLine($"12-step forecast: {CellJoin(model2.forecast(12))}");

    Console.WriteLine("\n=================== Exponential smoothing: Holt-Winters on retail-style seasonal data ===================");
    var seasonalShape = new double[] { 0.8, 0.75, 0.9, 0.95, 1.0, 1.05, 1.1, 1.05, 1.0, 1.1, 1.3, 1.6 };
    var y2 = new List<double>();
    for (int i = 0; i < 48; i++) {
      double lvl = 100.0 + 2.0 * i;
      y2.Add(lvl * seasonalShape[i % 12] + Randn() * 3.0);
    }

    var esOptions = new ExponentialSmoothingOptions();
    esOptions.trend = TrendType.Additive;
    esOptions.seasonal = SeasonalType.Multiplicative;
    esOptions.seasonal_period = 12;
    esOptions.de_population_size = 60;
    esOptions.de_max_generations = 300;
    var esModel = new ExponentialSmoothing(new DVector(y2), esOptions);
    Console.WriteLine($"alpha={esModel.alpha()} beta={esModel.beta()} gamma={esModel.gamma()}");
    Console.WriteLine($"sigma^2: {esModel.sigma2()}, AIC: {esModel.aic()}");
    Console.WriteLine($"12-month forecast: {CellJoin(esModel.forecast(12))}");

    Console.WriteLine("\n=================== Simple exponential smoothing vs. Holt's linear trend ===================");
    var flat = new double[] { 50.2, 49.8, 50.5, 49.6, 50.1, 50.3, 49.9, 50.0, 50.4, 49.7 };

    var ses = new ExponentialSmoothing(new DVector(flat), new ExponentialSmoothingOptions());
    Console.WriteLine($"SES alpha= {ses.alpha()}");
    Console.WriteLine($"SES 5-step forecast: {CellJoin(ses.forecast(5))}");

    var holtOptions = new ExponentialSmoothingOptions();
    holtOptions.trend = TrendType.Additive;
    var holt = new ExponentialSmoothing(new DVector(flat), holtOptions);
    Console.WriteLine($"Holt alpha={holt.alpha()} beta={holt.beta()}");
    Console.WriteLine($"Holt 5-step forecast: {CellJoin(holt.forecast(5))}");
  }
}
