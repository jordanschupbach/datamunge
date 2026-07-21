module app;

import std.stdio : writeln;
import datamunge;

DVector dv(double[] t) {
  auto v = new DVector();
  foreach (x; t) v.push_back(x);
  return v;
}

SVector sv(string[] t) {
  auto v = new SVector();
  foreach (x; t) v.push_back(x);
  return v;
}

IVector iv(int[] t) {
  auto v = new IVector();
  foreach (x; t) v.push_back(x);
  return v;
}

void main() {
  auto sales = new DataFrame();
  sales.add_string_column("region", sv(["west", "west", "east", "south", "south", "south"]));
  sales.add_string_column("product", sv(["widget", "widget", "widget", "gizmo", "gizmo", "gizmo"]));
  sales.add_numeric_column("sales", dv([10, 10, 14, 8, 0, 11]), iv([1, 1, 1, 1, 0, 1]));
  sales.add_string_column("quarter", sv(["Q1", "Q1", "Q1", "Q2", "Q2", ""]), iv([1, 1, 1, 1, 1, 0]));

  writeln("raw data");
  writeln(sales.to_string());
  writeln();

  auto cleaned = sales.drop_duplicates(sv(["region", "product", "sales", "quarter"]));
  cleaned.fill_null_string("quarter", "unknown");
  cleaned.fill_null_numeric("sales", 0.0);
  writeln("after drop_duplicates + fill_null");
  writeln(cleaned.to_string());
  writeln();

  auto selected = cleaned.select(sv(["region", "sales", "quarter"])).sort_by("sales", false);
  writeln("selected + sorted");
  writeln(selected.to_string());
  writeln();

  auto grouped = cleaned.group_by_sum(sv(["region"]), sv(["sales"])).sort_by("sales", false);
  writeln("group_by_sum(region)");
  writeln(grouped.to_string());
  writeln();

  auto targets = new DataFrame();
  targets.add_string_column("region", sv(["west", "east", "south"]));
  targets.add_numeric_column("target", dv([18, 12, 25]));
  auto joined = grouped.join(targets, "region", "region", "left");
  writeln("joined with targets");
  writeln(joined.to_string());
  writeln();

  auto shape = cleaned.shape();
  writeln("shape = (", shape[0], ", ", shape[1], ")");
  writeln("sales count = ", cleaned.numeric_count("sales"));
  writeln("sales nulls = ", cleaned.numeric_null_count("sales"));
  writeln("sales sum = ", cleaned.numeric_sum("sales"));
  writeln("sales mean = ", cleaned.numeric_mean("sales"));
}
