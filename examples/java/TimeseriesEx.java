// CMD: just TARGET=timeseries_ex run-java
// Time series: ARIMA, seasonal ARIMA and exponential smoothing (SES, Holt, Holt-Winters).
import datamunge.ARIMA;
import datamunge.ARIMAOptions;
import datamunge.ExponentialSmoothing;
import datamunge.ExponentialSmoothingOptions;
import datamunge.SeasonalType;
import datamunge.TrendType;
import java.util.Arrays;
import java.util.Random;
import java.util.stream.Collectors;

public class TimeseriesEx {
  static Random rng;

  // (0, 1]: avoids log(0) in the Box-Muller transform.
  static double uniform() {
    return 1 - rng.nextDouble();
  }

  // A standard normal draw by Box-Muller; reseed by assigning a new rng.
  static double gauss() {
    return Math.sqrt(-2 * Math.log(uniform())) * Math.cos(2 * Math.PI * uniform());
  }

  static String joined(double[] values) {
    return Arrays.stream(values).mapToObj(Double::toString).collect(Collectors.joining(", "));
  }

  public static void main(String[] args) {
    System.out.println(
        "=================== ARIMA(1,1,1) on a simulated random walk with drift"
            + " ===================");
    rng = new Random(42);
    double level = 100.0;
    double prevShock = 0.0;
    double[] y = new double[150];
    for (int i = 0; i < y.length; i++) {
      double shock = gauss();
      level += 0.3 + 0.4 * prevShock + shock; // integrated ARMA(0,1) increments with drift
      prevShock = shock;
      y[i] = level;
    }

    var options = new ARIMAOptions();
    options.setP(1);
    options.setD(1);
    options.setQ(1);
    options.setDePopulationSize(80);
    options.setDeMaxGenerations(400);
    var model = new ARIMA(y, options);

    System.out.println("AR coefficient: " + model.arCoefficients()[0]);
    System.out.println("MA coefficient: " + model.maCoefficients()[0]);
    System.out.println(
        "sigma^2: " + model.sigma2() + ", AIC: " + model.aic() + ", BIC: " + model.bic());

    var forecast = model.forecastWithIntervals(6);
    System.out.println("6-step forecast: " + joined(forecast.first()));
    System.out.println("forecast std. errors: " + joined(forecast.second()));

    System.out.println(
        "\n=================== SARIMA(1,0,0)(1,1,0)_12 on a seasonal series ===================");
    rng = new Random(7);
    double prev = 0.0;
    double[] s = new double[120];
    for (int i = 0; i < s.length; i++) {
      prev = 0.5 * prev + gauss();
      s[i] = 20 + 0.2 * i + 5 * Math.sin(2 * Math.PI * i / 12) + prev;
    }

    var options2 = new ARIMAOptions();
    options2.setP(1);
    options2.setSeasonalP(1);
    options2.setSeasonalD(1);
    options2.setSeasonalPeriod(12);
    options2.setDePopulationSize(100);
    options2.setDeMaxGenerations(500);
    var model2 = new ARIMA(s, options2);
    System.out.println(
        "AR coefficient: "
            + model2.arCoefficients()[0]
            + ", seasonal AR coefficient: "
            + model2.seasonalArCoefficients()[0]);
    System.out.println("12-step forecast: " + joined(model2.forecast(12)));

    System.out.println(
        "\n=================== Exponential smoothing: Holt-Winters on retail-style seasonal data"
            + " ===================");
    double[] seasonalShape = {0.8, 0.75, 0.9, 0.95, 1.0, 1.05, 1.1, 1.05, 1.0, 1.1, 1.3, 1.6};
    rng = new Random(11);
    double[] y2 = new double[48];
    for (int i = 0; i < y2.length; i++) {
      y2[i] = (100 + 2 * i) * seasonalShape[i % 12] + gauss() * 3;
    }

    var esOptions = new ExponentialSmoothingOptions();
    esOptions.setTrend(TrendType.Additive);
    esOptions.setSeasonal(SeasonalType.Multiplicative);
    esOptions.setSeasonalPeriod(12);
    esOptions.setDePopulationSize(60);
    esOptions.setDeMaxGenerations(300);
    var esModel = new ExponentialSmoothing(y2, esOptions);
    System.out.println(
        "alpha=" + esModel.alpha() + " beta=" + esModel.beta() + " gamma=" + esModel.gamma());
    System.out.println("sigma^2: " + esModel.sigma2() + ", AIC: " + esModel.aic());
    System.out.println("12-month forecast: " + joined(esModel.forecast(12)));

    System.out.println(
        "\n=================== Simple exponential smoothing vs. Holt's linear trend"
            + " ===================");
    double[] flat = {50.2, 49.8, 50.5, 49.6, 50.1, 50.3, 49.9, 50.0, 50.4, 49.7};

    var ses = new ExponentialSmoothing(flat, new ExponentialSmoothingOptions());
    System.out.println("SES alpha= " + ses.alpha());
    System.out.println("SES 5-step forecast: " + joined(ses.forecast(5)));

    var holtOptions = new ExponentialSmoothingOptions();
    holtOptions.setTrend(TrendType.Additive);
    var holt = new ExponentialSmoothing(flat, holtOptions);
    System.out.println("Holt alpha=" + holt.alpha() + " beta=" + holt.beta());
    System.out.println("Holt 5-step forecast: " + joined(holt.forecast(5)));
  }
}
