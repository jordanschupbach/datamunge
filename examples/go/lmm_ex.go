// CMD: just TARGET=lmm_ex run-go

//go:build ignore

// Linear mixed models: random intercept (penguins) and random intercept + slope (simulated schools).
package main

import (
	"datamunge"
	"datamunge/data"
	"fmt"
)

func check(err error) {
	if err != nil {
		panic(err)
	}
}

// One draw from Normal(0, sd) using datamunge's generator (seeded below), so the simulated data
// (and the fit below) differ from Java's, which uses java.util.Random.
func gauss(sd float64) float64 {
	return datamunge.Must(datamunge.Rnorm(0.0, sd, 1))[0]
}

// One draw from Uniform[lower, upper).
func uniform(lower, upper float64) float64 {
	return datamunge.Must(datamunge.Runif(lower, upper, 1))[0]
}

func main() {
	fmt.Println("=================== Random intercept on a real dataset (penguins) ===================")
	penguins := datamunge.Must(data.Penguins())
	defer penguins.Close()
	penguinModel := datamunge.Must(datamunge.NewLMM(penguins, "body_mass_g ~ flipper_length_mm + bill_length_mm + (1 | species)"))
	defer penguinModel.Close()
	check(penguinModel.PrintSummary())

	fmt.Println("\n=================== Random intercept + slope on a simulated multi-school dataset" +
		" ===================")
	// 30 schools, ~25 students each: score depends on study_hours, but both the baseline score
	// and the return on an extra hour of study vary by school -- the textbook case for a
	// random-intercept-and-slope model.
	check(datamunge.SetSeed(2024))
	nSchools := 30
	schoolIntercept := datamunge.Must(datamunge.Rnorm(0.0, 6.0, nSchools))
	schoolSlope := datamunge.Must(datamunge.Rnorm(0.0, 1.2, nSchools))
	trueIntercept := 60.0
	trueSlope := 3.0

	var school, studyHours, score []float64
	for s := 0; s < nSchools; s++ {
		nStudents := 15 + int(uniform(0, 21))
		for j := 0; j < nStudents; j++ {
			hours := uniform(0, 10)
			school = append(school, float64(s))
			studyHours = append(studyHours, hours)
			score = append(score, trueIntercept+schoolIntercept[s]+(trueSlope+schoolSlope[s])*hours+gauss(4.0))
		}
	}
	df := datamunge.Must(datamunge.NewDataFrame())
	defer df.Close()
	check(df.AddNumericColumn("school", school))
	check(df.AddNumericColumn("study_hours", studyHours))
	check(df.AddNumericColumn("score", score))

	model := datamunge.Must(datamunge.NewLMM(df, "score ~ study_hours + (1 + study_hours | school)"))
	defer model.Close()
	check(model.PrintSummary())

	fmt.Printf("\nTrue generating values: intercept=%.1f, slope=%.1f, random-intercept SD=6.0, random-slope SD=1.2, residual SD=4.0\n",
		trueIntercept, trueSlope)

	fmt.Println("\n--- Best Linear Unbiased Predictors (BLUPs) for a few schools ---")
	labels := datamunge.Must(model.GroupLabels())
	for idx := 0; idx < 3; idx++ {
		shifts := datamunge.Must(model.RandomEffectsForGroup(idx))
		fmt.Printf("school %s: intercept shift=%v, slope shift=%v\n", labels[idx], shifts[0], shifts[1])
	}

	fmt.Println("\n--- Prediction: population-level vs. school-adjusted ---")
	population := datamunge.Must(datamunge.NewDataFrame())
	defer population.Close()
	school0 := datamunge.Must(datamunge.NewDataFrame())
	defer school0.Close()
	check(population.AddNumericColumn("study_hours", []float64{5.0}))
	check(school0.AddNumericColumn("study_hours", []float64{5.0}))
	check(school0.AddNumericColumn("school", []float64{0.0}))
	fmt.Printf("5 study hours, unseen school:      %v (fixed effects only)\n", datamunge.Must(model.Predict(population))[0])
	fmt.Printf("5 study hours, school 0 (known):    %v (fixed effects + school 0's BLUP)\n", datamunge.Must(model.Predict(school0))[0])
}
