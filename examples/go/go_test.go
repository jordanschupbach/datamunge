package main

import (
	"fmt"
	"octra"
)

func main() {
	fmt.Println("Testing Go bindings for OCTRA...")

	// Test the hello function
	octra.Hello()

	// Test DPair (double pair)
	dp := octra.NewDPair(3.14, 2.71)
	fmt.Printf("DPair: First=%v, Second=%v\n", dp.First(), dp.Second())

	// Test SVector (string vector)
	sv := octra.NewSVector(3)
	sv.SetIndex(0, "Hello")
	sv.SetIndex(1, "World")
	sv.SetIndex(2, "OCTRA")
	fmt.Printf("SVector: [0]=%v, [1]=%v, [2]=%v\n", sv.GetIndex(0), sv.GetIndex(1), sv.GetIndex(2))

	// Clean up resources
	dp.Delete()
	sv.Delete()

	fmt.Println("Go bindings test completed successfully!")
}
