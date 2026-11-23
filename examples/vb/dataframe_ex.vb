' CMD: just TARGET=dataframe_ex run-vb
' DataFrame basics: nulls, drop_duplicates, fill_null, select/sort, group-by, join and summaries.
Option Strict On
Imports System
Imports Datamunge

Module Example
    Sub Main()
        Dim sales = New DataFrame()
        sales.AddStringColumn("region", {"west", "west", "east", "south", "south", "south"})
        sales.AddStringColumn("product", {"widget", "widget", "widget", "gizmo", "gizmo", "gizmo"})
        sales.AddNumericColumn("sales", {10.0, 10.0, 14.0, 8.0, 0.0, 11.0}, {1, 1, 1, 1, 0, 1})
        sales.AddStringColumn("quarter", {"Q1", "Q1", "Q1", "Q2", "Q2", ""}, {1, 1, 1, 1, 1, 0})

        Console.WriteLine($"raw data{vbLf}{sales}{vbLf}")

        Dim cleaned = sales.DropDuplicates({"region", "product", "sales", "quarter"})
        cleaned.FillNullString("quarter", "unknown")
        cleaned.FillNullNumeric("sales", 0.0)
        Console.WriteLine($"after drop_duplicates + fill_null{vbLf}{cleaned}{vbLf}")

        Dim selected = cleaned.Select({"region", "sales", "quarter"}).SortBy("sales", False)
        Console.WriteLine($"selected + sorted{vbLf}{selected}{vbLf}")

        Dim grouped = cleaned.GroupBySum({"region"}, {"sales"}).SortBy("sales", False)
        Console.WriteLine($"group_by_sum(region){vbLf}{grouped}{vbLf}")

        Dim targets = New DataFrame()
        targets.AddStringColumn("region", {"west", "east", "south"})
        targets.AddNumericColumn("target", {18.0, 12.0, 25.0})
        Console.WriteLine($"joined with targets{vbLf}{grouped.Join(targets, "region", "region", "left")}{vbLf}")

        Dim shape = cleaned.Shape()
        Console.WriteLine($"shape = ({shape(0)}, {shape(1)})")
        Console.WriteLine($"sales count = {cleaned.NumericCount("sales")}")
        Console.WriteLine($"sales nulls = {cleaned.NumericNullCount("sales")}")
        Console.WriteLine($"sales sum = {cleaned.NumericSum("sales")}")
        Console.WriteLine($"sales mean = {cleaned.NumericMean("sales")}")
    End Sub
End Module
