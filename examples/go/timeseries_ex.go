package main

import (
	"datamunge"
	"fmt"
	"math"
	"math/rand"
	"strings"
)

func dvector(values []float64) datamunge.DVector {
	out := datamunge.NewDVector(int64(len(values)))
	for i, v := range values {
		out.Set(i, v)
	}
	return out
}

func randn() float64 {
	u1 := rand.Float64()
	u2 := rand.Float64()
	return math.Sqrt(-2.0*math.Log(u1)) * math.Cos(2.0*math.Pi*u2)
}

func cellJoin(v datamunge.DVector) string {
	parts := make([]string, v.Size())
	for i := 0; i < int(v.Size()); i++ {
		parts[i] = fmt.Sprintf("%g", v.Get(i))
	}
	return strings.Join(parts, ", ")
}

func main() {
	fmt.Println("=================== ARIMA(1,1,1) on a simulated random walk with drift ===================")
	y := []float64{}
	level := 100.0
	prevShock := 0.0
	for i := 0; i < 150; i++ {
		shock := randn()
		level = level + 0.3 + 0.4*prevShock + shock
		y = append(y, level)
		prevShock = shock
	}

	options := datamunge.NewARIMAOptions()
	options.SetP(1)
	options.SetD(1)
	options.SetQ(1)
	options.SetDe_population_size(80)
	options.SetDe_max_generations(400)
	model := datamunge.NewARIMA(dvector(y), options)

	ar := model.Ar_coefficients()
	ma := model.Ma_coefficients()
	fmt.Printf("AR coefficient: %g\n", ar.Get(0))
	fmt.Printf("MA coefficient: %g\n", ma.Get(0))
	fmt.Printf("sigma^2: %g, AIC: %g, BIC: %g\n", model.Sigma2(), model.Aic(), model.Bic())

	pair := model.Forecast_with_intervals(6)
	fmt.Printf("6-step forecast: %s\n", cellJoin(pair.GetFirst()))
	fmt.Printf("forecast std. errors: %s\n", cellJoin(pair.GetSecond()))

	fmt.Println("\n=================== SARIMA(1,0,0)(1,1,0)_12 on a seasonal series ===================")
	s := []float64{}
	prev := 0.0
	for i := 0; i < 120; i++ {
		prev = 0.5*prev + randn()
		s = append(s, 20.0+0.2*float64(i)+5.0*math.Sin((2.0*math.Pi*float64(i))/12.0)+prev)
	}

	options2 := datamunge.NewARIMAOptions()
	options2.SetP(1)
	options2.SetSeasonal_p(1)
	options2.SetSeasonal_d(1)
	options2.SetSeasonal_period(12)
	options2.SetDe_population_size(100)
	options2.SetDe_max_generations(500)
	model2 := datamunge.NewARIMA(dvector(s), options2)
	ar2 := model2.Ar_coefficients()
	sar2 := model2.Seasonal_ar_coefficients()
	fmt.Printf("AR coefficient: %g, seasonal AR coefficient: %g\n", ar2.Get(0), sar2.Get(0))
	fmt.Printf("12-step forecast: %s\n", cellJoin(model2.Forecast(int64(12))))

	fmt.Println("\n=================== Exponential smoothing: Holt-Winters on retail-style seasonal data ===================")
	seasonalShape := []float64{0.8, 0.75, 0.9, 0.95, 1.0, 1.05, 1.1, 1.05, 1.0, 1.1, 1.3, 1.6}
	y2 := []float64{}
	for i := 0; i < 48; i++ {
		lvl := 100.0 + 2.0*float64(i)
		y2 = append(y2, lvl*seasonalShape[i%12]+randn()*3.0)
	}

	esOptions := datamunge.NewExponentialSmoothingOptions()
	esOptions.SetTrend(datamunge.TrendType_Additive)
	esOptions.SetSeasonal(datamunge.SeasonalType_Multiplicative)
	esOptions.SetSeasonal_period(12)
	esOptions.SetDe_population_size(60)
	esOptions.SetDe_max_generations(300)
	esModel := datamunge.NewExponentialSmoothing(dvector(y2), esOptions)
	fmt.Printf("alpha=%g beta=%g gamma=%g\n", esModel.Alpha(), esModel.Beta(), esModel.Gamma())
	fmt.Printf("sigma^2: %g, AIC: %g\n", esModel.Sigma2(), esModel.Aic())
	fmt.Printf("12-month forecast: %s\n", cellJoin(esModel.Forecast(int64(12))))

	fmt.Println("\n=================== Simple exponential smoothing vs. Holt's linear trend ===================")
	flat := []float64{50.2, 49.8, 50.5, 49.6, 50.1, 50.3, 49.9, 50.0, 50.4, 49.7}

	ses := datamunge.NewExponentialSmoothing(dvector(flat), datamunge.NewExponentialSmoothingOptions())
	fmt.Printf("SES alpha= %g\n", ses.Alpha())
	fmt.Printf("SES 5-step forecast: %s\n", cellJoin(ses.Forecast(int64(5))))

	holtOptions := datamunge.NewExponentialSmoothingOptions()
	holtOptions.SetTrend(datamunge.TrendType_Additive)
	holt := datamunge.NewExponentialSmoothing(dvector(flat), holtOptions)
	fmt.Printf("Holt alpha=%g beta=%g\n", holt.Alpha(), holt.Beta())
	fmt.Printf("Holt 5-step forecast: %s\n", cellJoin(holt.Forecast(int64(5))))
}
