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

func randn() float64 {
	u1 := rand.Float64()
	u2 := rand.Float64()
	return math.Sqrt(-2.0*math.Log(u1)) * math.Cos(2.0*math.Pi*u2)
}

func main() {
	fmt.Println("=================== Random intercept on a real dataset (penguins) ===================")
	penguins := datamunge.DataFramePenguins()
	speciesModel := datamunge.NewLMM(penguins, "body_mass_g ~ flipper_length_mm + bill_length_mm + (1 | species)")
	speciesModel.Print_summary()

	fmt.Println("\n=================== Random intercept + slope on a simulated multi-school dataset ===================")
	nSchools := 30
	schoolIntercept := make([]float64, nSchools)
	schoolSlope := make([]float64, nSchools)
	for i := 0; i < nSchools; i++ {
		schoolIntercept[i] = randn() * 6.0
		schoolSlope[i] = randn() * 1.2
	}

	trueIntercept := 60.0
	trueSlope := 3.0
	school := []float64{}
	studyHours := []float64{}
	score := []float64{}
	for s := 0; s < nSchools; s++ {
		nStudents := 15 + int(rand.Float64()*21)
		for j := 0; j < nStudents; j++ {
			hours := rand.Float64() * 10.0
			noise := randn() * 4.0
			sVal := trueIntercept + schoolIntercept[s] + (trueSlope+schoolSlope[s])*hours + noise
			school = append(school, float64(s))
			studyHours = append(studyHours, hours)
			score = append(score, sVal)
		}
	}

	df := datamunge.NewDataFrame()
	df.Add_numeric_column("school", dvector(school))
	df.Add_numeric_column("study_hours", dvector(studyHours))
	df.Add_numeric_column("score", dvector(score))

	model := datamunge.NewLMM(df, "score ~ study_hours + (1 + study_hours | school)")
	model.Print_summary()

	fmt.Printf("\nTrue generating values: intercept=%g, slope=%g, random-intercept SD=6.0, random-slope SD=1.2, residual SD=4.0\n", trueIntercept, trueSlope)

	fmt.Println("\n--- Best Linear Unbiased Predictors (BLUPs) for a few schools ---")
	groupLabels := model.Group_labels()
	for idx := 0; idx < 3; idx++ {
		re := model.Random_effects_for_group(int64(idx))
		fmt.Printf("school %s: intercept shift=%g, slope shift=%g\n", groupLabels.Get(idx), re.Get(0), re.Get(1))
	}

	fmt.Println("\n--- Prediction: population-level vs. school-adjusted ---")
	newdataPopulation := datamunge.NewDataFrame()
	newdataPopulation.Add_numeric_column("study_hours", dvector([]float64{5.0}))
	newdataSchool0 := datamunge.NewDataFrame()
	newdataSchool0.Add_numeric_column("study_hours", dvector([]float64{5.0}))
	newdataSchool0.Add_numeric_column("school", dvector([]float64{0.0}))
	predPop := model.Predict(newdataPopulation)
	predS0 := model.Predict(newdataSchool0)
	fmt.Printf("5 study hours, unseen school:      %g (fixed effects only)\n", predPop.Get(0))
	fmt.Printf("5 study hours, school 0 (known):    %g (fixed effects + school 0's BLUP)\n", predS0.Get(0))
}
