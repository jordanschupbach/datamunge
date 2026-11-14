// CMD: just TARGET=timeseries_ex run-csharp
#:project ../../src/datamungecsharp/Datamunge.csproj
// Time series: ARIMA, seasonal ARIMA and exponential smoothing (SES, Holt, Holt-Winters).
using Datamunge;

var rng = new Random(42);
void Seed(int n) => rng = new Random(n);
double Uniform() => 1 - rng.NextDouble();  // (0, 1]: avoids log(0) in the Box-Muller transform.
double Gauss() => Math.Sqrt(-2 * Math.Log(Uniform())) * Math.Cos(2 * Math.PI * Uniform());
static string Joined(IEnumerable<double> v) => string.Join(", ", v);

Console.WriteLine("=================== ARIMA(1,1,1) on a simulated random walk with drift ===================");
Seed(42);
List<double> y = [];
{
    double level = 100.0, prevShock = 0.0;
    for (var i = 0; i < 150; i++)
    {
        var shock = Gauss();
        level += 0.3 + 0.4 * prevShock + shock;  // integrated ARMA(0,1) increments with drift
        y.Add(level);
        prevShock = shock;
    }
}

var model = new ARIMA(y, new ARIMAOptions { P = 1, D = 1, Q = 1, DePopulationSize = 80, DeMaxGenerations = 400 });

Console.WriteLine($"AR coefficient: {model.ArCoefficients()[0]}");
Console.WriteLine($"MA coefficient: {model.MaCoefficients()[0]}");
Console.WriteLine($"sigma^2: {model.Sigma2()}, AIC: {model.Aic()}, BIC: {model.Bic()}");

var (forecastValues, stdErrors) = model.ForecastWithIntervals(6);
Console.WriteLine($"6-step forecast: {Joined(forecastValues)}");
Console.WriteLine($"forecast std. errors: {Joined(stdErrors)}");

Console.WriteLine("\n=================== SARIMA(1,0,0)(1,1,0)_12 on a seasonal series ===================");
Seed(7);
List<double> s = [];
{
    var prev = 0.0;
    for (var i = 0; i < 120; i++)
    {
        prev = 0.5 * prev + Gauss();
        s.Add(20 + 0.2 * i + 5 * Math.Sin(2 * Math.PI * i / 12) + prev);
    }
}

var model2 = new ARIMA(s, new ARIMAOptions
{
    P = 1,
    SeasonalP = 1,
    SeasonalD = 1,
    SeasonalPeriod = 12,
    DePopulationSize = 100,
    DeMaxGenerations = 500,
});
Console.WriteLine($"AR coefficient: {model2.ArCoefficients()[0]}, seasonal AR coefficient: {model2.SeasonalArCoefficients()[0]}");
Console.WriteLine($"12-step forecast: {Joined(model2.Forecast(12))}");

Console.WriteLine("\n=================== Exponential smoothing: Holt-Winters on retail-style seasonal data ===================");
double[] seasonalShape = [0.8, 0.75, 0.9, 0.95, 1.0, 1.05, 1.1, 1.05, 1.0, 1.1, 1.3, 1.6];
Seed(11);
var y2 = Enumerable.Range(0, 48).Select(i => (100 + 2 * i) * seasonalShape[i % 12] + Gauss() * 3).ToList();

var esModel = new ExponentialSmoothing(y2, new ExponentialSmoothingOptions
{
    Trend = TrendType.Additive,
    Seasonal = SeasonalType.Multiplicative,
    SeasonalPeriod = 12,
    DePopulationSize = 60,
    DeMaxGenerations = 300,
});
Console.WriteLine($"alpha={esModel.Alpha()} beta={esModel.Beta()} gamma={esModel.Gamma()}");
Console.WriteLine($"sigma^2: {esModel.Sigma2()}, AIC: {esModel.Aic()}");
Console.WriteLine($"12-month forecast: {Joined(esModel.Forecast(12))}");

Console.WriteLine("\n=================== Simple exponential smoothing vs. Holt's linear trend ===================");
double[] flat = [50.2, 49.8, 50.5, 49.6, 50.1, 50.3, 49.9, 50.0, 50.4, 49.7];

var ses = new ExponentialSmoothing(flat, new ExponentialSmoothingOptions());
Console.WriteLine($"SES alpha= {ses.Alpha()}");
Console.WriteLine($"SES 5-step forecast: {Joined(ses.Forecast(5))}");

var holt = new ExponentialSmoothing(flat, new ExponentialSmoothingOptions { Trend = TrendType.Additive });
Console.WriteLine($"Holt alpha={holt.Alpha()} beta={holt.Beta()}");
Console.WriteLine($"Holt 5-step forecast: {Joined(holt.Forecast(5))}");
