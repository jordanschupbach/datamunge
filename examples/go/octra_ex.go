package main

import (
	"fmt"
	"octra"
)

func main() {
	octra.Hello()

	dp := octra.NewDPair(1.0, 2.0)
	fmt.Printf("DPair: First=%v, Second=%v\n", dp.GetFirst(), dp.GetSecond())

	s := octra.NewSVector(int64(3))
	s.Set(0, "1")
	s.Set(1, "2")
	s.Set(2, "3")
	fmt.Printf("SVector: [0]=%v, [1]=%v, [2]=%v\n", s.Get(0), s.Get(1), s.Get(2))

	// Clean up
	octra.DeleteDPair(dp)
	octra.DeleteSVector(s)
}
