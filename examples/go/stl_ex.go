// CMD: just TARGET=stl_ex run-go

//go:build ignore

// C++ std::vector and std::pair values in Go: vectors are slices, pairs are datamunge.Pair structs.
package main

import (
	"datamunge"
	"fmt"
	"math"
)

func main() {
	n := 100
	v := make([]float64, n)
	for i := range v {
		v[i] = float64(i) * 1.5
	}
	for _, x := range v {
		fmt.Println(x)
	}
	v2 := make([]int, n)
	for i := range v2 {
		v2[i] = int(math.Floor(float64(i) * 1.5))
	}
	for _, x := range v2 {
		fmt.Println(x)
	}

	p := datamunge.Pair[int, int]{First: 3, Second: 4}
	fmt.Printf("p: (%v, %v)\n", p.First, p.Second)

	p2 := datamunge.Pair[int, int]{First: 10, Second: 20}
	fmt.Printf("p2: (%v, %v)\n", p2.First, p2.Second)

	// A C++ std::pair<double,double> comes back as a datamunge.Pair[float64, float64].
	p3 := datamunge.Must(datamunge.MakeDpair(1.25, 2.75))
	fmt.Printf("make_dpair(1.25, 2.75): (%v, %v), sum_dpair = %v\n",
		p3.First, p3.Second, datamunge.Must(datamunge.SumDpair(p3)))
}
