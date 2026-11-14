// CMD: just TARGET=glmm_ex run-go

//go:build ignore

// Generalized linear mixed models: a binomial GLMM on penguins and a Poisson GLMM on simulated store data.
package main

import (
	"datamunge"
	"datamunge/data"
	"fmt"
	"math"
	"math/rand"
)

func check(err error) {
	if err != nil {
		panic(err)
	}
}

// Go's own generator, so the simulated data (and the fit below) differ from other languages'.
var rng = rand.New(rand.NewSource(4242))

// Box-Muller; 1 - u keeps log() away from 0.
func gauss() float64 {
	u1 := 1 - rng.Float64()
	u2 := rng.Float64()
	return math.Sqrt(-2*math.Log(u1)) * math.Cos(2*math.Pi*u2)
}

// Knuth's Poisson sampler.
func rpois(lambda float64) int {
	threshold := math.Exp(-lambda)
	k := 0
	p := 1.0
	for {
		k++
		p *= rng.Float64()
		if p <= threshold {
			break
		}
	}
	return k - 1
}

func main() {
	fmt.Println("=================== Binomial (logistic) mixed model on a real dataset (penguins) ===================")
	// Predicting sex from body mass, grouped by island: unlike species (which is almost perfectly
	// confounded with island in this dataset -- Gentoo penguins occur on Biscoe island only), sex
	// is not tied to location, so this is a well-behaved fit.
	penguins := datamunge.Must(data.Penguins())
	defer penguins.Close()
	var isMale, bodyMass []float64
	var island []string
	for i := 0; i < datamunge.Must(penguins.Nrows()); i++ {
		if datamunge.Must(penguins.IsNull("sex", i)) ||
			datamunge.Must(penguins.IsNull("body_mass_g", i)) ||
			datamunge.Must(penguins.IsNull("island", i)) {
			continue
		}
		male := 0.0
		if datamunge.Must(penguins.StringAt("sex", i)) == "male" {
			male = 1.0
		}
		isMale = append(isMale, male)
		bodyMass = append(bodyMass, datamunge.Must(penguins.NumericAt("body_mass_g", i)))
		island = append(island, datamunge.Must(penguins.StringAt("island", i)))
	}
	sexDf := datamunge.Must(datamunge.NewDataFrame())
	defer sexDf.Close()
	check(sexDf.AddNumericColumn("is_male", isMale))
	check(sexDf.AddNumericColumn("body_mass_g", bodyMass))
	check(sexDf.AddStringColumn("island", island))
	sexModel := datamunge.Must(datamunge.NewGLMMWithFamily(sexDf, "is_male ~ body_mass_g + (1 | island)", "binomial"))
	defer sexModel.Close()
	check(sexModel.PrintSummary())

	fmt.Println("\n=================== Poisson mixed model on simulated multi-site count data ===================")
	// 25 stores, ~20 days each: daily visit counts depend on a promo intensity score, but both
	// the baseline traffic and the promo's effectiveness vary by store.
	nStores := 25
	storeEffect := make([]float64, nStores)
	for i := range storeEffect {
		storeEffect[i] = gauss() * 0.4
	}
	trueIntercept := 2.0
	trueSlope := 0.3

	var store, promo, visits []float64
	for s := 0; s < nStores; s++ {
		nDays := 15 + rng.Intn(11)
		for day := 0; day < nDays; day++ {
			promoIntensity := rng.Float64() * 3
			lambda := math.Exp(trueIntercept + storeEffect[s] + trueSlope*promoIntensity)
			store = append(store, float64(s))
			promo = append(promo, promoIntensity)
			visits = append(visits, float64(rpois(lambda)))
		}
	}
	df := datamunge.Must(datamunge.NewDataFrame())
	defer df.Close()
	check(df.AddNumericColumn("store", store))
	check(df.AddNumericColumn("promo", promo))
	check(df.AddNumericColumn("visits", visits))
	storeModel := datamunge.Must(datamunge.NewGLMMWithFamily(df, "visits ~ promo + (1 | store)", "poisson"))
	defer storeModel.Close()

	check(storeModel.PrintSummary())

	fmt.Printf("\nTrue generating values: intercept=%v, slope=%v, random-intercept SD (log scale)=0.4\n",
		trueIntercept, trueSlope)

	fmt.Println("\n--- BLUPs for a few stores ---")
	labels := datamunge.Must(storeModel.GroupLabels())
	for idx := 0; idx < 3; idx++ {
		fmt.Printf("store %s: intercept shift=%v\n", labels[idx], datamunge.Must(storeModel.RandomEffectsForGroup(idx))[0])
	}

	fmt.Println("\n--- Prediction: population-level vs. store-adjusted ---")
	population := datamunge.Must(datamunge.NewDataFrame())
	defer population.Close()
	store0 := datamunge.Must(datamunge.NewDataFrame())
	defer store0.Close()
	check(population.AddNumericColumn("promo", []float64{1.5}))
	check(store0.AddNumericColumn("promo", []float64{1.5}))
	check(store0.AddNumericColumn("store", []float64{0.0}))
	fmt.Printf("promo=1.5, unseen store:   %v expected visits (fixed effects only)\n",
		datamunge.Must(storeModel.Predict(population))[0])
	fmt.Printf("promo=1.5, store 0 (known): %v expected visits (fixed effects + store 0's BLUP)\n",
		datamunge.Must(storeModel.Predict(store0))[0])
}
