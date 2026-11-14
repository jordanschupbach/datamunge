// CMD: just TARGET=dataframe_ex run-csharp
#:project ../../src/datamungecsharp/Datamunge.csproj
// DataFrame basics: nulls, drop_duplicates, fill_null, select/sort, group-by, join and summaries.
using Datamunge;

var sales = new DataFrame();
sales.AddStringColumn("region", ["west", "west", "east", "south", "south", "south"]);
sales.AddStringColumn("product", ["widget", "widget", "widget", "gizmo", "gizmo", "gizmo"]);
sales.AddNumericColumn("sales", [10.0, 10.0, 14.0, 8.0, 0.0, 11.0], [1, 1, 1, 1, 0, 1]);
sales.AddStringColumn("quarter", ["Q1", "Q1", "Q1", "Q2", "Q2", ""], [1, 1, 1, 1, 1, 0]);

Console.WriteLine($"raw data\n{sales}\n");

var cleaned = sales.DropDuplicates(["region", "product", "sales", "quarter"])!;
cleaned.FillNullString("quarter", "unknown");
cleaned.FillNullNumeric("sales", 0.0);
Console.WriteLine($"after drop_duplicates + fill_null\n{cleaned}\n");

var selected = cleaned.Select(["region", "sales", "quarter"])!.SortBy("sales", false);
Console.WriteLine($"selected + sorted\n{selected}\n");

var grouped = cleaned.GroupBySum(["region"], ["sales"])!.SortBy("sales", false)!;
Console.WriteLine($"group_by_sum(region)\n{grouped}\n");

var targets = new DataFrame();
targets.AddStringColumn("region", ["west", "east", "south"]);
targets.AddNumericColumn("target", [18.0, 12.0, 25.0]);
Console.WriteLine($"joined with targets\n{grouped.Join(targets, "region", "region", "left")}\n");

var shape = cleaned.Shape();
Console.WriteLine($"shape = ({shape[0]}, {shape[1]})");
Console.WriteLine($"sales count = {cleaned.NumericCount("sales")}");
Console.WriteLine($"sales nulls = {cleaned.NumericNullCount("sales")}");
Console.WriteLine($"sales sum = {cleaned.NumericSum("sales")}");
Console.WriteLine($"sales mean = {cleaned.NumericMean("sales")}");
