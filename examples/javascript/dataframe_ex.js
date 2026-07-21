const datamunge = require("../../index.js");

function encodeStrings(values) {
  return `${values.length}\x1e${values.join("\x1f")}`;
}

function dvector(values) {
  const out = new datamunge.DVector(values.length);
  for (let index = 0; index < values.length; index += 1) {
    out.set(index, values[index]);
  }
  return out;
}

function ivector(values) {
  const out = new datamunge.IVector(values.length);
  for (let index = 0; index < values.length; index += 1) {
    out.set(index, values[index]);
  }
  return out;
}

function printSummary(df) {
  const shape = df.shape();
  console.log(`shape = (${shape.get(0)}, ${shape.get(1)})`);
  console.log(`sales count = ${df.numeric_count("sales")}`);
  console.log(`sales nulls = ${df.numeric_null_count("sales")}`);
  console.log(`sales sum = ${df.numeric_sum("sales")}`);
  console.log(`sales mean = ${df.numeric_mean("sales")}`);
}

const sales = new datamunge.DataFrame();
sales.add_string_column_encoded("region", encodeStrings(["west", "west", "east", "south", "south", "south"]));
sales.add_string_column_encoded("product", encodeStrings(["widget", "widget", "widget", "gizmo", "gizmo", "gizmo"]));
sales.add_numeric_column("sales", dvector([10, 10, 14, 8, 0, 11]), ivector([1, 1, 1, 1, 0, 1]));
sales.add_string_column_encoded("quarter", encodeStrings(["Q1", "Q1", "Q1", "Q2", "Q2", ""]), ivector([1, 1, 1, 1, 1, 0]));

console.log("raw data");
console.log(sales.to_string());
console.log();

const cleaned = sales.drop_duplicates_encoded(encodeStrings(["region", "product", "sales", "quarter"]));
cleaned.fill_null_string("quarter", "unknown");
cleaned.fill_null_numeric("sales", 0.0);
console.log("after drop_duplicates + fill_null");
console.log(cleaned.to_string());
console.log();

const selected = cleaned.select_encoded(encodeStrings(["region", "sales", "quarter"])).sort_by("sales", false);
console.log("selected + sorted");
console.log(selected.to_string());
console.log();

const grouped = cleaned.group_by_sum_encoded(encodeStrings(["region"]), encodeStrings(["sales"])).sort_by("sales", false);
console.log("group_by_sum(region)");
console.log(grouped.to_string());
console.log();

const targets = new datamunge.DataFrame();
targets.add_string_column_encoded("region", encodeStrings(["west", "east", "south"]));
targets.add_numeric_column("target", dvector([18, 12, 25]));
const joined = grouped.join(targets, "region", "region", "left");
console.log("joined with targets");
console.log(joined.to_string());
console.log();

printSummary(cleaned);
