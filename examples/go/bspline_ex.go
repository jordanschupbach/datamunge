// CMD: just TARGET=bspline_ex run-go

//go:build ignore

// Tensor-product B-spline regression: fit z ~ bs(x, y) to a smooth surface and predict.
package main

import (
	"datamunge"
	"fmt"
	"math"
)

func check(err error) {
	if err != nil {
		panic(err)
	}
}

func main() {
	// Sample a smooth surface z = sin(x) + y^2 on an 8x8 grid over [0, 1]^2.
	var x, y, z []float64
	for iy := 0; iy < 8; iy++ {
		for ix := 0; ix < 8; ix++ {
			xv, yv := float64(ix)/7.0, float64(iy)/7.0
			x = append(x, xv)
			y = append(y, yv)
			z = append(z, math.Sin(xv)+yv*yv)
		}
	}
	data := datamunge.Must(datamunge.NewDataFrame())
	defer data.Close()
	check(data.AddNumericColumn("x", x))
	check(data.AddNumericColumn("y", y))
	check(data.AddNumericColumn("z", z))

	surface := datamunge.Must(datamunge.NewLM(data, "z ~ bs(x, y)"))
	defer surface.Close()
	fmt.Println("Fitted z ~ bs(x, y)")
	check(surface.PrintSummary())

	newPoints := datamunge.Must(datamunge.NewDataFrame())
	defer newPoints.Close()
	check(newPoints.AddNumericColumn("x", []float64{0.25, 0.75}))
	check(newPoints.AddNumericColumn("y", []float64{0.50, 0.25}))
	pred := datamunge.Must(surface.Predict(newPoints))
	fmt.Printf("Predictions: [%.6f, %.6f]\n", pred[0], pred[1])
}
