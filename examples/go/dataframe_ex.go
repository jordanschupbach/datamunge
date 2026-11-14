// CMD: just TARGET=dataframe_ex run-go

//go:build ignore

// DataFrame basics: nulls, drop_duplicates, fill_null, select/sort, group-by, join and summaries.
package main

import (
	"datamunge"
	"fmt"
)

func check(err error) {
	if err != nil {
		panic(err)
	}
}

func main() {
	sales := datamunge.Must(datamunge.NewDataFrame())
	defer sales.Close()
	check(sales.AddStringColumn("region", []string{"west", "west", "east", "south", "south", "south"}))
	check(sales.AddStringColumn("product", []string{"widget", "widget", "widget", "gizmo", "gizmo", "gizmo"}))
	check(sales.AddNumericColumnWithValidMask("sales", []float64{10.0, 10.0, 14.0, 8.0, 0.0, 11.0}, []int{1, 1, 1, 1, 0, 1}))
	check(sales.AddStringColumnWithValidMask("quarter", []string{"Q1", "Q1", "Q1", "Q2", "Q2", ""}, []int{1, 1, 1, 1, 1, 0}))

	fmt.Printf("raw data\n%v\n\n", sales)

	cleaned := datamunge.Must(sales.DropDuplicatesWithSubset([]string{"region", "product", "sales", "quarter"}))
	defer cleaned.Close()
	check(cleaned.FillNullString("quarter", "unknown"))
	check(cleaned.FillNullNumeric("sales", 0.0))
	fmt.Printf("after drop_duplicates + fill_null\n%v\n\n", cleaned)

	selected := datamunge.Must(datamunge.Must(cleaned.Select([]string{"region", "sales", "quarter"})).SortByWithAscending("sales", false))
	fmt.Printf("selected + sorted\n%v\n\n", selected)

	grouped := datamunge.Must(datamunge.Must(cleaned.GroupBySum([]string{"region"}, []string{"sales"})).SortByWithAscending("sales", false))
	fmt.Printf("group_by_sum(region)\n%v\n\n", grouped)

	targets := datamunge.Must(datamunge.NewDataFrame())
	defer targets.Close()
	check(targets.AddStringColumn("region", []string{"west", "east", "south"}))
	check(targets.AddNumericColumn("target", []float64{18.0, 12.0, 25.0}))
	fmt.Printf("joined with targets\n%v\n\n", datamunge.Must(grouped.JoinWithJoinType(targets, "region", "region", "left")))

	shape := datamunge.Must(cleaned.Shape())
	fmt.Printf("shape = (%d, %d)\n", shape[0], shape[1])
	fmt.Printf("sales count = %v\n", datamunge.Must(cleaned.NumericCount("sales")))
	fmt.Printf("sales nulls = %v\n", datamunge.Must(cleaned.NumericNullCount("sales")))
	fmt.Printf("sales sum = %v\n", datamunge.Must(cleaned.NumericSum("sales")))
	fmt.Printf("sales mean = %v\n", datamunge.Must(cleaned.NumericMean("sales")))
}
