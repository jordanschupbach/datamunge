package main

import (
	"datamunge"
	"fmt"
)

func main() {
	n := int64(100)
	v := datamunge.NewDVector(n)
	for i := int64(0); i < n; i++ {
		v.Set(int(i), float64(i)*1.5)
	}
	for i := 0; i < int(n); i++ {
		fmt.Println(v.Get(i))
	}

	v2 := datamunge.NewIVector(n)
	for i := int64(0); i < n; i++ {
		v2.Set(int(i), int(float64(i)*1.5))
	}
	for i := 0; i < int(n); i++ {
		fmt.Println(v2.Get(i))
	}

	p := datamunge.NewIPair(3, 4)
	fmt.Printf("p: (%d, %d)\n", p.GetFirst(), p.GetSecond())

	p2 := datamunge.NewDPair(10.0, 20.0)
	fmt.Printf("p2: (%v, %v)\n", p2.GetFirst(), p2.GetSecond())
}
