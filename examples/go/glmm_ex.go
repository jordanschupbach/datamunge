package main

import (
	"datamunge"
	"fmt"
	"math"
	"math/rand"
)

func dvector(values []float64) datamunge.DVector {
	out := datamunge.NewDVector(int64(len(values)))
	for i, v := range values {
		out.Set(i, v)
	}
	return out
}

func svector(values []string) datamunge.SVector {
	out := datamunge.NewSVector(int64(len(values)))
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

func main() {
	fmt.Println("=================== Binomial (logistic) mixed model on a real dataset (penguins) ===================")
	penguins := datamunge.DataFramePenguins()
	isMale := []float64{}
	bodyMass := []float64{}
	island := []string{}
	n := penguins.Nrows()
	for i := int64(0); i < n; i++ {
		if !penguins.Is_null("sex", i) && !penguins.Is_null("body_mass_g", i) && !penguins.Is_null("island", i) {
			if penguins.String_at("sex", i) == "male" {
				isMale = append(isMale, 1.0)
			} else {
				isMale = append(isMale, 0.0)
			}
			bodyMass = append(bodyMass, penguins.Numeric_at("body_mass_g", i))
			island = append(island, penguins.String_at("island", i))
		}
	}

	sexDf := datamunge.NewDataFrame()
	sexDf.Add_numeric_column("is_male", dvector(isMale))
	sexDf.Add_numeric_column("body_mass_g", dvector(bodyMass))
	sexDf.Add_string_column("island", svector(island))

	sexModel := datamunge.NewGLMM(sexDf, "is_male ~ body_mass_g + (1 | island)", "binomial")
	sexModel.Print_summary()

	fmt.Println("\n=================== Poisson mixed model on simulated multi-site count data ===================")
	nStores := 25
	storeEffect := make([]float64, nStores)
	for i := 0; i < nStores; i++ {
		storeEffect[i] = randn() * 0.4
	}

	trueIntercept := 2.0
	trueSlope := 0.3
	store := []float64{}
	promo := []float64{}
	visits := []float64{}
	for s := 0; s < nStores; s++ {
		nDays := 15 + int(rand.Float64()*11)
		for d := 0; d < nDays; d++ {
			promoIntensity := rand.Float64() * 3.0
			lam := math.Exp(trueIntercept + storeEffect[s] + trueSlope*promoIntensity)
			lThresh := math.Exp(-lam)
			k := 0
			p := 1.0
			for {
				k++
				p *= rand.Float64()
				if p <= lThresh {
					break
				}
			}
			store = append(store, float64(s))
			promo = append(promo, promoIntensity)
			visits = append(visits, float64(k-1))
		}
	}

	df := datamunge.NewDataFrame()
	df.Add_numeric_column("store", dvector(store))
	df.Add_numeric_column("promo", dvector(promo))
	df.Add_numeric_column("visits", dvector(visits))

	storeModel := datamunge.NewGLMM(df, "visits ~ promo + (1 | store)", "poisson")
	storeModel.Print_summary()

	fmt.Printf("\nTrue generating values: intercept=%g, slope=%g, random-intercept SD (log scale)=0.4\n", trueIntercept, trueSlope)

	fmt.Println("\n--- BLUPs for a few stores ---")
	groupLabels := storeModel.Group_labels()
	for idx := 0; idx < 3; idx++ {
		re := storeModel.Random_effects_for_group(int64(idx))
		fmt.Printf("store %s: intercept shift=%g\n", groupLabels.Get(idx), re.Get(0))
	}

	fmt.Println("\n--- Prediction: population-level vs. store-adjusted ---")
	newdataPopulation := datamunge.NewDataFrame()
	newdataPopulation.Add_numeric_column("promo", dvector([]float64{1.5}))
	newdataStore0 := datamunge.NewDataFrame()
	newdataStore0.Add_numeric_column("promo", dvector([]float64{1.5}))
	newdataStore0.Add_numeric_column("store", dvector([]float64{0.0}))
	predPop := storeModel.Predict(newdataPopulation)
	predS0 := storeModel.Predict(newdataStore0)
	fmt.Printf("promo=1.5, unseen store:   %g expected visits (fixed effects only)\n", predPop.Get(0))
	fmt.Printf("promo=1.5, store 0 (known): %g expected visits (fixed effects + store 0's BLUP)\n", predS0.Get(0))
}
