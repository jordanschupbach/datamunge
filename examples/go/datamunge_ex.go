// CMD: just TARGET=datamunge_ex run-go

//go:build ignore

// Smoke test of the basic bindings: vectors, pairs and a native Callback object.
package main

import (
	"datamunge"
	"fmt"
)

func main() {
	datamunge.Must(0, datamunge.Hello())

	v := datamunge.Must(datamunge.MakeDvector(1.0, 2.0, 3.0))
	fmt.Printf("sum_dvector:\t%v\n", datamunge.Must(datamunge.SumDvector(v)))

	p := datamunge.Must(datamunge.MakeDpair(1.25, 2.75))
	fmt.Printf("sum_dpair:\t%v\n", datamunge.Must(datamunge.SumDpair(p)))

	cb := datamunge.Must(datamunge.NewCallback())
	defer cb.Close()
	fmt.Printf("call_with_callback(3.0):\t%v\n", datamunge.Must(datamunge.CallWithCallback(3.0, cb)))
	v2 := datamunge.Must(datamunge.MapDvectorWithCallback(datamunge.Must(datamunge.MakeDvector(1.0, 2.0, 3.0)), cb))
	fmt.Printf("sum_dvector(map_dvector_with_callback(1,2,3)):\t%v\n", datamunge.Must(datamunge.SumDvector(v2)))
}
