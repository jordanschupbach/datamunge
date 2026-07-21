package main

import (
	"datamunge"
	"fmt"
)

func encodeStrings(values []string) string {
	encoded := fmt.Sprintf("%d\x1e", len(values))
	for index, value := range values {
		if index > 0 {
			encoded += "\x1f"
		}
		encoded += value
	}
	return encoded
}

func dvector(values []float64) datamunge.DVector {
	out := datamunge.NewDVector(int64(len(values)))
	for index, value := range values {
		out.Set(index, value)
	}
	return out
}

func ivector(values []int) datamunge.IVector {
	out := datamunge.NewIVector(int64(len(values)))
	for index, value := range values {
		out.Set(index, value)
	}
	return out
}

func main() {
	sales := datamunge.NewDataFrame()
	sales.Add_string_column_encoded("region", encodeStrings([]string{"west", "west", "east", "south", "south", "south"}))
	sales.Add_string_column_encoded("product", encodeStrings([]string{"widget", "widget", "widget", "gizmo", "gizmo", "gizmo"}))
	sales.Add_numeric_column("sales", dvector([]float64{10, 10, 14, 8, 0, 11}), ivector([]int{1, 1, 1, 1, 0, 1}))
	sales.Add_string_column_encoded("quarter", encodeStrings([]string{"Q1", "Q1", "Q1", "Q2", "Q2", ""}), ivector([]int{1, 1, 1, 1, 1, 0}))

	fmt.Println("raw data")
	fmt.Println(sales.To_string())
	fmt.Println()

	cleaned := sales.Drop_duplicates_encoded(encodeStrings([]string{"region", "product", "sales", "quarter"}))
	cleaned.Fill_null_string("quarter", "unknown")
	cleaned.Fill_null_numeric("sales", 0.0)
	fmt.Println("after drop_duplicates + fill_null")
	fmt.Println(cleaned.To_string())
	fmt.Println()

	selected := cleaned.Select_encoded(encodeStrings([]string{"region", "sales", "quarter"})).Sort_by("sales", false)
	fmt.Println("selected + sorted")
	fmt.Println(selected.To_string())
	fmt.Println()

	grouped := cleaned.Group_by_sum_encoded(encodeStrings([]string{"region"}), encodeStrings([]string{"sales"})).Sort_by("sales", false)
	fmt.Println("group_by_sum(region)")
	fmt.Println(grouped.To_string())
	fmt.Println()

	targets := datamunge.NewDataFrame()
	targets.Add_string_column_encoded("region", encodeStrings([]string{"west", "east", "south"}))
	targets.Add_numeric_column("target", dvector([]float64{18, 12, 25}))
	joined := grouped.Join(targets, "region", "region", "left")
	fmt.Println("joined with targets")
	fmt.Println(joined.To_string())
	fmt.Println()

	shape := cleaned.Shape()
	fmt.Printf("shape = (%d, %d)\n", shape.Get(0), shape.Get(1))
	fmt.Printf("sales count = %d\n", cleaned.Numeric_count("sales"))
	fmt.Printf("sales nulls = %d\n", cleaned.Numeric_null_count("sales"))
	fmt.Printf("sales sum = %.0f\n", cleaned.Numeric_sum("sales"))
	fmt.Printf("sales mean = %.1f\n", cleaned.Numeric_mean("sales"))
}
