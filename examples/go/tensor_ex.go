package main

import (
	"datamunge"
	"fmt"
	"strings"
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

func szvector(values []int) datamunge.SizeVector {
	out := datamunge.NewSizeVector(int64(len(values)))
	for i, v := range values {
		out.Set(i, int64(v))
	}
	return out
}

func shapeStr(shp datamunge.SizeVector) string {
	parts := make([]string, shp.Size())
	for i := 0; i < int(shp.Size()); i++ {
		parts[i] = fmt.Sprintf("%d", shp.Get(i))
	}
	return strings.Join(parts, ", ")
}

func main() {
	fmt.Println("=================== Construction ===================")
	z := datamunge.TensorZeros(szvector([]int{2, 3}))
	fmt.Printf("zeros([2,3]): %s\n", z.To_string())

	eyet := datamunge.TensorEye(int64(3))
	fmt.Printf("eye(3): %s\n", eyet.To_string())

	r := datamunge.TensorArange(0.0, 12.0).Reshape(szvector([]int{3, 4}))
	fmt.Printf("arange(0,12).reshape([3,4]): %s\n", r.To_string())

	fmt.Println("\n=================== Shape ops ===================")
	rt := r.Transpose()
	fmt.Printf("transpose -> shape [%s]\n", shapeStr(rt.Shape()))
	sliced := r.Slice(int64(1), int64(1), int64(3))
	fmt.Printf("slice(axis=1, start=1, stop=3): %s\n", sliced.To_string())

	fmt.Println("\n=================== Broadcasting arithmetic ===================")
	col := datamunge.TensorFrom_values(szvector([]int{3, 1}), dvector([]float64{1, 2, 3}))
	row := datamunge.TensorFrom_values(szvector([]int{1, 4}), dvector([]float64{10, 20, 30, 40}))
	broadcastSum := col.Add(row)
	fmt.Printf("(3,1) + (1,4) -> %s\n", broadcastSum.To_string())

	fmt.Println("\n=================== Reductions ===================")
	fmt.Printf("r.sum() = %g, r.mean() = %g\n", r.Sum(), r.Mean())
	colMeans := r.Mean_axis(int64(0))
	fmt.Printf("column means (axis=0): %s\n", colMeans.To_string())

	fmt.Println("\n=================== Linear algebra ===================")
	a := datamunge.TensorFrom_values(szvector([]int{2, 3}), dvector([]float64{1, 2, 3, 4, 5, 6}))
	b := datamunge.TensorFrom_values(szvector([]int{3, 2}), dvector([]float64{7, 8, 9, 10, 11, 12}))
	fmt.Printf("matmul(2x3, 3x2) -> %s\n", a.Matmul(b).To_string())

	v1 := datamunge.TensorFrom_values(szvector([]int{3}), dvector([]float64{1, 2, 3}))
	v2 := datamunge.TensorFrom_values(szvector([]int{3}), dvector([]float64{4, 5, 6}))
	fmt.Printf("dot([1,2,3], [4,5,6]) = %g\n", v1.Dot(v2))
	fmt.Printf("outer(v1, v2) -> %s\n", v1.Outer(v2).To_string())

	fmt.Println("\n=================== Comparisons & masks ===================")
	mask := r.Greater_equal(datamunge.TensorFull(szvector([]int{3, 4}), 6.0))
	fmt.Printf("r >= 6 -> %s\n", mask.To_string())
	fmt.Printf("count(r >= 6) = %g\n", mask.Sum())

	fmt.Println("\n=================== A real dataset as a Tensor ===================")
	iris := datamunge.DataFrameIris()
	n := iris.Nrows()
	flat := make([]float64, 0, n*4)
	for i := int64(0); i < n; i++ {
		flat = append(flat, iris.Numeric_at("Sepal.Length", i))
		flat = append(flat, iris.Numeric_at("Sepal.Width", i))
		flat = append(flat, iris.Numeric_at("Petal.Length", i))
		flat = append(flat, iris.Numeric_at("Petal.Width", i))
	}
	x := datamunge.TensorFrom_values(szvector([]int{int(n), 4}), dvector(flat))
	fmt.Printf("iris feature tensor shape: [%s]\n", shapeStr(x.Shape()))

	featureMeans := x.Mean_axis(int64(0))
	centered := x.Subtract(featureMeans.Reshape(szvector([]int{1, 4})))
	scatter := centered.Transpose().Matmul(centered)
	fmt.Printf("feature means (Sepal.Length, Sepal.Width, Petal.Length, Petal.Width): %s\n", featureMeans.To_string())
	fmt.Printf("(X-mean)^T (X-mean) [4x4 scatter matrix]: %s\n", scatter.To_string(int64(16)))

	species := make([]string, 0, n)
	for i := int64(0); i < n; i++ {
		species = append(species, iris.String_at("Species", i))
	}
	speciesTensor := datamunge.TensorFrom_string_values(szvector([]int{int(n)}), svector(species))
	setosaMask := speciesTensor.Equal(datamunge.TensorFrom_string_values(szvector([]int{1}), svector([]string{"setosa"})))
	fmt.Printf("setosa count = %g (of %d rows)\n", setosaMask.Sum(), n)
}
