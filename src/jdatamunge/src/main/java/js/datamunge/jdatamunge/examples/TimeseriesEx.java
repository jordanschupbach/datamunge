package js.datamunge.jdatamunge.examples;

import js.datamunge.jdatamunge.DVector;
import js.datamunge.jdatamunge.ARIMA;
import js.datamunge.jdatamunge.ARIMAOptions;
import js.datamunge.jdatamunge.ExponentialSmoothing;
import js.datamunge.jdatamunge.ExponentialSmoothingOptions;
import js.datamunge.jdatamunge.TrendType;
import js.datamunge.jdatamunge.SeasonalType;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class TimeseriesEx {
  static {
    System.loadLibrary("datamunge_jni");
  }

  static Random rng = new Random();

  static double randn() {
    return rng.nextGaussian();
  }

  static String cellJoin(DVector v) {
    List<String> parts = new ArrayList<>();
    for (int i = 0; i < v.size(); i++) parts.add(String.valueOf(v.get(i)));
    return String.join(", ", parts);
  }

  public static void run() {
    System.out.println("=================== ARIMA(1,1,1) on a simulated random walk with drift ===================");
    List<Double> y = new ArrayList<>();
    double level = 100.0;
    double prevShock = 0.0;
    for (int i = 0; i < 150; i++) {
      double shock = randn();
      level = level + 0.3 + 0.4 * prevShock + shock;
      y.add(level);
      prevShock = shock;
    }

    var options = new ARIMAOptions();
    options.setP(1);
    options.setD(1);
    options.setQ(1);
    options.setDe_population_size(80);
    options.setDe_max_generations(400);
    var model = new ARIMA(new DVector(y), options);

    var ar = model.ar_coefficients();
    var ma = model.ma_coefficients();
    System.out.println("AR coefficient: " + ar.get(0));
    System.out.println("MA coefficient: " + ma.get(0));
    System.out.println("sigma^2: " + model.sigma2() + ", AIC: " + model.aic() + ", BIC: " + model.bic());

    var pair = model.forecast_with_intervals(6);
    System.out.println("6-step forecast: " + cellJoin(pair.getFirst()));
    System.out.println("forecast std. errors: " + cellJoin(pair.getSecond()));

    System.out.println("\n=================== SARIMA(1,0,0)(1,1,0)_12 on a seasonal series ===================");
    List<Double> s = new ArrayList<>();
    double prev = 0.0;
    for (int i = 0; i < 120; i++) {
      prev = 0.5 * prev + randn();
      s.add(20.0 + 0.2 * i + 5.0 * Math.sin((2.0 * Math.PI * i) / 12.0) + prev);
    }

    var options2 = new ARIMAOptions();
    options2.setP(1);
    options2.setSeasonal_p(1);
    options2.setSeasonal_d(1);
    options2.setSeasonal_period(12);
    options2.setDe_population_size(100);
    options2.setDe_max_generations(500);
    var model2 = new ARIMA(new DVector(s), options2);
    var ar2 = model2.ar_coefficients();
    var sar2 = model2.seasonal_ar_coefficients();
    System.out.println("AR coefficient: " + ar2.get(0) + ", seasonal AR coefficient: " + sar2.get(0));
    System.out.println("12-step forecast: " + cellJoin(model2.forecast(12)));

    System.out.println("\n=================== Exponential smoothing: Holt-Winters on retail-style seasonal data ===================");
    double[] seasonalShape = {0.8, 0.75, 0.9, 0.95, 1.0, 1.05, 1.1, 1.05, 1.0, 1.1, 1.3, 1.6};
    List<Double> y2 = new ArrayList<>();
    for (int i = 0; i < 48; i++) {
      double lvl = 100.0 + 2.0 * i;
      y2.add(lvl * seasonalShape[i % 12] + randn() * 3.0);
    }

    var esOptions = new ExponentialSmoothingOptions();
    esOptions.setTrend(TrendType.Additive);
    esOptions.setSeasonal(SeasonalType.Multiplicative);
    esOptions.setSeasonal_period(12);
    esOptions.setDe_population_size(60);
    esOptions.setDe_max_generations(300);
    var esModel = new ExponentialSmoothing(new DVector(y2), esOptions);
    System.out.println("alpha=" + esModel.alpha() + " beta=" + esModel.beta() + " gamma=" + esModel.gamma());
    System.out.println("sigma^2: " + esModel.sigma2() + ", AIC: " + esModel.aic());
    System.out.println("12-month forecast: " + cellJoin(esModel.forecast(12)));

    System.out.println("\n=================== Simple exponential smoothing vs. Holt's linear trend ===================");
    double[] flat = {50.2, 49.8, 50.5, 49.6, 50.1, 50.3, 49.9, 50.0, 50.4, 49.7};

    var ses = new ExponentialSmoothing(new DVector(flat), new ExponentialSmoothingOptions());
    System.out.println("SES alpha= " + ses.alpha());
    System.out.println("SES 5-step forecast: " + cellJoin(ses.forecast(5)));

    var holtOptions = new ExponentialSmoothingOptions();
    holtOptions.setTrend(TrendType.Additive);
    var holt = new ExponentialSmoothing(new DVector(flat), holtOptions);
    System.out.println("Holt alpha=" + holt.alpha() + " beta=" + holt.beta());
    System.out.println("Holt 5-step forecast: " + cellJoin(holt.forecast(5)));
  }
}
