// CMD: just TARGET=tensor_ex run-go

//go:build ignore

// Tensors: construction, shape ops, broadcasting, reductions, linear algebra, masks, iris.
package main

import (
	"datamunge"
	"datamunge/data"
	"fmt"
	"strings"
)

func shapeString(tensor *datamunge.Tensor) string {
	parts := []string{}
	for _, d := range datamunge.Must(tensor.Shape()) {
		parts = append(parts, fmt.Sprint(d))
	}
	return "[" + strings.Join(parts, ", ") + "]"
}

func main() {
	fmt.Println("=================== Construction ===================")
	fmt.Println("zeros([2,3]):", datamunge.Must(datamunge.TensorZeros([]int{2, 3})))
	fmt.Println("eye(3):", datamunge.Must(datamunge.TensorEye(3)))

	r := datamunge.Must(datamunge.Must(datamunge.TensorArangeWithStep(0.0, 12.0, 1.0)).Reshape([]int{3, 4}))
	fmt.Println("arange(0,12).reshape([3,4]):", r)

	fmt.Println("\n=================== Shape ops ===================")
	fmt.Println("transpose -> shape " + shapeString(datamunge.Must(r.Transpose())))
	fmt.Println("slice(axis=1, start=1, stop=3):", datamunge.Must(r.Slice(1, 1, 3)))

	fmt.Println("\n=================== Broadcasting arithmetic ===================")
	col := datamunge.Must(datamunge.TensorFromValues([]int{3, 1}, []float64{1.0, 2.0, 3.0}))
	row := datamunge.Must(datamunge.TensorFromValues([]int{1, 4}, []float64{10.0, 20.0, 30.0, 40.0}))
	fmt.Println("(3,1) + (1,4) ->", datamunge.Must(col.Add(row)))

	fmt.Println("\n=================== Reductions ===================")
	fmt.Printf("r.sum() = %v, r.mean() = %v\n", datamunge.Must(r.Sum()), datamunge.Must(r.Mean()))
	fmt.Println("column means (axis=0):", datamunge.Must(r.MeanAxis(0)))

	fmt.Println("\n=================== Linear algebra ===================")
	a := datamunge.Must(datamunge.TensorFromValues([]int{2, 3}, []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}))
	b := datamunge.Must(datamunge.TensorFromValues([]int{3, 2}, []float64{7.0, 8.0, 9.0, 10.0, 11.0, 12.0}))
	fmt.Println("matmul(2x3, 3x2) ->", datamunge.Must(a.Matmul(b)))
	v1 := datamunge.Must(datamunge.TensorFromValues([]int{3}, []float64{1.0, 2.0, 3.0}))
	v2 := datamunge.Must(datamunge.TensorFromValues([]int{3}, []float64{4.0, 5.0, 6.0}))
	fmt.Println("dot([1,2,3], [4,5,6]) =", datamunge.Must(v1.Dot(v2)))
	fmt.Println("outer(v1, v2) ->", datamunge.Must(v1.Outer(v2)))

	fmt.Println("\n=================== Comparisons & masks ===================")
	mask := datamunge.Must(r.GreaterEqual(datamunge.Must(datamunge.TensorFull([]int{3, 4}, 6.0))))
	fmt.Println("r >= 6 ->", mask)
	fmt.Println("count(r >= 6) =", datamunge.Must(mask.Sum()))

	fmt.Println("\n=================== A real dataset as a Tensor ===================")
	iris := datamunge.Must(data.Iris())
	defer iris.Close()
	n := datamunge.Must(iris.Nrows())
	columns := []string{"Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"}
	flat := make([]float64, 0, n*len(columns))
	for i := 0; i < n; i++ {
		for _, column := range columns {
			flat = append(flat, datamunge.Must(iris.NumericAt(column, i)))
		}
	}
	x := datamunge.Must(datamunge.TensorFromValues([]int{n, 4}, flat))
	fmt.Println("iris feature tensor shape: " + shapeString(x))

	featureMeans := datamunge.Must(x.MeanAxis(0))
	centered := datamunge.Must(x.Subtract(datamunge.Must(featureMeans.Reshape([]int{1, 4}))))
	scatter := datamunge.Must(datamunge.Must(centered.Transpose()).Matmul(centered))
	fmt.Println("feature means (Sepal.Length, Sepal.Width, Petal.Length, Petal.Width):", featureMeans)
	fmt.Println("(X-mean)^T (X-mean) [4x4 scatter matrix]:", datamunge.Must(scatter.ToStringWithMaxElements(16)))

	species := make([]string, n)
	for i := range species {
		species[i] = datamunge.Must(iris.StringAt("Species", i))
	}
	speciesTensor := datamunge.Must(datamunge.TensorFromStringValues([]int{n}, species))
	setosaMask := datamunge.Must(speciesTensor.Equal(datamunge.Must(datamunge.TensorFromStringValues([]int{1}, []string{"setosa"}))))
	fmt.Printf("setosa count = %v (of %d rows)\n", datamunge.Must(setosaMask.Sum()), n)
}
