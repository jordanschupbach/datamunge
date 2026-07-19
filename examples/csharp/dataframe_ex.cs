using System;

class Program {
  static void Main() {
    var sales = new DataFrame();
    sales.add_string_column("region", new SVector(new string[] { "west", "west", "east", "south", "south", "south" }));
    sales.add_string_column("product", new SVector(new string[] { "widget", "widget", "widget", "gizmo", "gizmo", "gizmo" }));
    sales.add_numeric_column("sales", new DVector(new double[] { 10.0, 10.0, 14.0, 8.0, 0.0, 11.0 }), new IVector(new int[] { 1, 1, 1, 1, 0, 1 }));
    sales.add_string_column("quarter", new SVector(new string[] { "Q1", "Q1", "Q1", "Q2", "Q2", "" }), new IVector(new int[] { 1, 1, 1, 1, 1, 0 }));

    Console.WriteLine("raw data");
    Console.WriteLine(sales.to_string());
    Console.WriteLine();

    var cleaned = sales.drop_duplicates(new SVector(new string[] { "region", "product", "sales", "quarter" }));
    cleaned.fill_null_string("quarter", "unknown");
    cleaned.fill_null_numeric("sales", 0.0);
    Console.WriteLine("after drop_duplicates + fill_null");
    Console.WriteLine(cleaned.to_string());
    Console.WriteLine();

    var selected = cleaned.select(new SVector(new string[] { "region", "sales", "quarter" })).sort_by("sales", false);
    Console.WriteLine("selected + sorted");
    Console.WriteLine(selected.to_string());
    Console.WriteLine();

    var grouped = cleaned.group_by_sum(new SVector(new string[] { "region" }), new SVector(new string[] { "sales" })).sort_by("sales", false);
    Console.WriteLine("group_by_sum(region)");
    Console.WriteLine(grouped.to_string());
    Console.WriteLine();

    var targets = new DataFrame();
    targets.add_string_column("region", new SVector(new string[] { "west", "east", "south" }));
    targets.add_numeric_column("target", new DVector(new double[] { 18.0, 12.0, 25.0 }));
    var joined = grouped.join(targets, "region", "region", true);
    Console.WriteLine("joined with targets");
    Console.WriteLine(joined.to_string());
    Console.WriteLine();

    var shape = cleaned.shape();
    Console.WriteLine($"shape = ({shape[0]}, {shape[1]})");
    Console.WriteLine("sales count = " + cleaned.numeric_count("sales"));
    Console.WriteLine("sales nulls = " + cleaned.numeric_null_count("sales"));
    Console.WriteLine("sales sum = " + cleaned.numeric_sum("sales"));
    Console.WriteLine("sales mean = " + cleaned.numeric_mean("sales"));
  }
}
