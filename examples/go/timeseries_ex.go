// CMD: just TARGET=timeseries_ex run-go

//go:build ignore

// Time series: ARIMA, seasonal ARIMA and exponential smoothing (SES, Holt, Holt-Winters).
package main

import (
	"datamunge"
	"fmt"
	"math"
	"math/rand"
	"strings"
)

func check(err error) {
	if err != nil {
		panic(err)
	}
}

// Go's own generator, so the simulated data (and the fits below) differ from other languages'.
var rng *rand.Rand

// (0, 1]: avoids log(0) in the Box-Muller transform.
func uniform() float64 {
	return 1 - rng.Float64()
}

// A standard normal draw by Box-Muller; reseed by assigning a new rng.
func gauss() float64 {
	return math.Sqrt(-2*math.Log(uniform())) * math.Cos(2*math.Pi*uniform())
}

func joined(values []float64) string {
	parts := make([]string, len(values))
	for i, v := range values {
		parts[i] = fmt.Sprint(v)
	}
	return strings.Join(parts, ", ")
}

func main() {
	fmt.Println("=================== ARIMA(1,1,1) on a simulated random walk with drift ===================")
	rng = rand.New(rand.NewSource(42))
	level := 100.0
	prevShock := 0.0
	y := make([]float64, 150)
	for i := range y {
		shock := gauss()
		level += 0.3 + 0.4*prevShock + shock // integrated ARMA(0,1) increments with drift
		prevShock = shock
		y[i] = level
	}

	options := datamunge.Must(datamunge.NewARIMAOptions())
	defer options.Close()
	check(options.SetP(1))
	check(options.SetD(1))
	check(options.SetQ(1))
	check(options.SetDePopulationSize(80))
	check(options.SetDeMaxGenerations(400))
	model := datamunge.Must(datamunge.NewARIMAWithOptions(y, options))
	defer model.Close()

	fmt.Printf("AR coefficient: %v\n", datamunge.Must(model.ArCoefficients())[0])
	fmt.Printf("MA coefficient: %v\n", datamunge.Must(model.MaCoefficients())[0])
	fmt.Printf("sigma^2: %v, AIC: %v, BIC: %v\n",
		datamunge.Must(model.Sigma2()), datamunge.Must(model.Aic()), datamunge.Must(model.Bic()))

	forecast := datamunge.Must(model.ForecastWithIntervals(6))
	fmt.Println("6-step forecast: " + joined(forecast.First))
	fmt.Println("forecast std. errors: " + joined(forecast.Second))

	fmt.Println("\n=================== SARIMA(1,0,0)(1,1,0)_12 on a seasonal series ===================")
	rng = rand.New(rand.NewSource(7))
	prev := 0.0
	s := make([]float64, 120)
	for i := range s {
		prev = 0.5*prev + gauss()
		s[i] = 20 + 0.2*float64(i) + 5*math.Sin(2*math.Pi*float64(i)/12) + prev
	}

	options2 := datamunge.Must(datamunge.NewARIMAOptions())
	defer options2.Close()
	check(options2.SetP(1))
	check(options2.SetSeasonalP(1))
	check(options2.SetSeasonalD(1))
	check(options2.SetSeasonalPeriod(12))
	check(options2.SetDePopulationSize(100))
	check(options2.SetDeMaxGenerations(500))
	model2 := datamunge.Must(datamunge.NewARIMAWithOptions(s, options2))
	defer model2.Close()
	fmt.Printf("AR coefficient: %v, seasonal AR coefficient: %v\n",
		datamunge.Must(model2.ArCoefficients())[0], datamunge.Must(model2.SeasonalArCoefficients())[0])
	fmt.Println("12-step forecast: " + joined(datamunge.Must(model2.Forecast(12))))

	fmt.Println("\n=================== Exponential smoothing: Holt-Winters on retail-style seasonal data ===================")
	seasonalShape := []float64{0.8, 0.75, 0.9, 0.95, 1.0, 1.05, 1.1, 1.05, 1.0, 1.1, 1.3, 1.6}
	rng = rand.New(rand.NewSource(11))
	y2 := make([]float64, 48)
	for i := range y2 {
		y2[i] = (100+2*float64(i))*seasonalShape[i%12] + gauss()*3
	}

	esOptions := datamunge.Must(datamunge.NewExponentialSmoothingOptions())
	defer esOptions.Close()
	check(esOptions.SetTrend(datamunge.TrendTypeAdditive))
	check(esOptions.SetSeasonal(datamunge.SeasonalTypeMultiplicative))
	check(esOptions.SetSeasonalPeriod(12))
	check(esOptions.SetDePopulationSize(60))
	check(esOptions.SetDeMaxGenerations(300))
	esModel := datamunge.Must(datamunge.NewExponentialSmoothingWithOptions(y2, esOptions))
	defer esModel.Close()
	fmt.Printf("alpha=%v beta=%v gamma=%v\n",
		datamunge.Must(esModel.Alpha()), datamunge.Must(esModel.Beta()), datamunge.Must(esModel.Gamma()))
	fmt.Printf("sigma^2: %v, AIC: %v\n", datamunge.Must(esModel.Sigma2()), datamunge.Must(esModel.Aic()))
	fmt.Println("12-month forecast: " + joined(datamunge.Must(esModel.Forecast(12))))

	fmt.Println("\n=================== Simple exponential smoothing vs. Holt's linear trend ===================")
	flat := []float64{50.2, 49.8, 50.5, 49.6, 50.1, 50.3, 49.9, 50.0, 50.4, 49.7}

	ses := datamunge.Must(datamunge.NewExponentialSmoothing(flat))
	defer ses.Close()
	fmt.Printf("SES alpha= %v\n", datamunge.Must(ses.Alpha()))
	fmt.Println("SES 5-step forecast: " + joined(datamunge.Must(ses.Forecast(5))))

	holtOptions := datamunge.Must(datamunge.NewExponentialSmoothingOptions())
	defer holtOptions.Close()
	check(holtOptions.SetTrend(datamunge.TrendTypeAdditive))
	holt := datamunge.Must(datamunge.NewExponentialSmoothingWithOptions(flat, holtOptions))
	defer holt.Close()
	fmt.Printf("Holt alpha=%v beta=%v\n", datamunge.Must(holt.Alpha()), datamunge.Must(holt.Beta()))
	fmt.Println("Holt 5-step forecast: " + joined(datamunge.Must(holt.Forecast(5))))
}
