// CMD: just TARGET=dataframe_ex run-javascript

const datamunge = require("../../index.js");

function printSummary(df) {
  const shape = df.shape();
  console.log(`shape = (${shape[0]}, ${shape[1]})`);
  console.log(`sales count = ${df.numeric_count("sales")}`);
  console.log(`sales nulls = ${df.numeric_null_count("sales")}`);
  console.log(`sales sum = ${df.numeric_sum("sales")}`);
  console.log(`sales mean = ${df.numeric_mean("sales")}`);
}

const sales = new datamunge.DataFrame();
sales.add_string_column("region", ["west", "west", "east", "south", "south", "south"]);
sales.add_string_column("product", ["widget", "widget", "widget", "gizmo", "gizmo", "gizmo"]);
sales.add_numeric_column("sales", [10, 10, 14, 8, 0, 11], [1, 1, 1, 1, 0, 1]);
sales.add_string_column("quarter", ["Q1", "Q1", "Q1", "Q2", "Q2", ""], [1, 1, 1, 1, 1, 0]);

console.log("raw data");
console.log(sales.to_string());
console.log();

const cleaned = sales.drop_duplicates(["region", "product", "sales", "quarter"]);
cleaned.fill_null_string("quarter", "unknown");
cleaned.fill_null_numeric("sales", 0.0);
console.log("after drop_duplicates + fill_null");
console.log(cleaned.to_string());
console.log();

const selected = cleaned.select(["region", "sales", "quarter"]).sort_by("sales", false);
console.log("selected + sorted");
console.log(selected.to_string());
console.log();

const grouped = cleaned.group_by_sum(["region"], ["sales"]).sort_by("sales", false);
console.log("group_by_sum(region)");
console.log(grouped.to_string());
console.log();

const targets = new datamunge.DataFrame();
targets.add_string_column("region", ["west", "east", "south"]);
targets.add_numeric_column("target", [18, 12, 25]);
const joined = grouped.join(targets, "region", "region", "left");
console.log("joined with targets");
console.log(joined.to_string());
console.log();

printSummary(cleaned);
