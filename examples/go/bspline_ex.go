package main

import (
	"datamunge"
	"fmt"
	"math"
)

func dvector(values []float64) datamunge.DVector {
	out := datamunge.NewDVector(int64(len(values)))
	for i, v := range values {
		out.Set(i, v)
	}
	return out
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

	data := datamunge.NewDataFrame()
	data.Add_numeric_column("x", dvector(x))
	data.Add_numeric_column("y", dvector(y))
	data.Add_numeric_column("z", dvector(z))

	surface := datamunge.NewLM(data, "z ~ bs(x, y)")
	fmt.Println("Fitted z ~ bs(x, y)")
	surface.Print_summary()

	newPoints := datamunge.NewDataFrame()
	newPoints.Add_numeric_column("x", dvector([]float64{0.25, 0.75}))
	newPoints.Add_numeric_column("y", dvector([]float64{0.50, 0.25}))
	pred := surface.Predict(newPoints)
	fmt.Printf("Predictions: [%.6f, %.6f]\n", pred.Get(0), pred.Get(1))
}
