from pydatamunge import datamunge


def encode_strings(values):
    return f"{len(values)}\x1e" + "\x1f".join(values)


def dvector(values):
    out = datamunge.DVector(len(values))
    for index, value in enumerate(values):
      out[index] = float(value)
    return out


def ivector(values):
    out = datamunge.IVector(len(values))
    for index, value in enumerate(values):
      out[index] = int(value)
    return out


def print_summary(df):
    shape = df.shape()
    print(f"shape = ({shape[0]}, {shape[1]})")
    print(f"sales count = {df.numeric_count('sales')}")
    print(f"sales nulls = {df.numeric_null_count('sales')}")
    print(f"sales sum = {df.numeric_sum('sales')}")
    print(f"sales mean = {df.numeric_mean('sales')}")


sales = datamunge.DataFrame()
sales.add_string_column_encoded("region", encode_strings(["west", "west", "east", "south", "south", "south"]))
sales.add_string_column_encoded("product", encode_strings(["widget", "widget", "widget", "gizmo", "gizmo", "gizmo"]))
sales.add_numeric_column("sales", dvector([10, 10, 14, 8, 0, 11]), ivector([1, 1, 1, 1, 0, 1]))
sales.add_string_column_encoded("quarter", encode_strings(["Q1", "Q1", "Q1", "Q2", "Q2", ""]), ivector([1, 1, 1, 1, 1, 0]))

print("raw data")
print(sales.to_string())
print()

cleaned = sales.drop_duplicates_encoded(encode_strings(["region", "product", "sales", "quarter"]))
cleaned.fill_null_string("quarter", "unknown")
cleaned.fill_null_numeric("sales", 0.0)
print("after drop_duplicates + fill_null")
print(cleaned.to_string())
print()

selected = cleaned.select_encoded(encode_strings(["region", "sales", "quarter"])).sort_by("sales", False)
print("selected + sorted")
print(selected.to_string())
print()

grouped = cleaned.group_by_sum_encoded(encode_strings(["region"]), encode_strings(["sales"])).sort_by("sales", False)
print("group_by_sum(region)")
print(grouped.to_string())
print()

targets = datamunge.DataFrame()
targets.add_string_column_encoded("region", encode_strings(["west", "east", "south"]))
targets.add_numeric_column("target", dvector([18, 12, 25]))
joined = grouped.join(targets, "region", "region", "left")
print("joined with targets")
print(joined.to_string())
print()

print_summary(cleaned)

# Tibble/dplyr-style piping: every transform below returns a new DataFrame, so method
# chaining composes naturally, e.g. `df.mutate_numeric(...).arrange_encoded(...)`.
piped = (
    cleaned.mutate_numeric("tax", dvector([v * 0.1 for v in cleaned.pull_numeric("sales")]))
    .rename("quarter", "period")
    .arrange_encoded(encode_strings(["region", "sales"]), ivector([1, 0]))
    .relocate_encoded(encode_strings(["region", "sales"]), "")
    .select_encoded(encode_strings(["region", "sales", "tax", "period"]))
)
print("piped (mutate + arrange + rename + relocate + select)")
print(piped.to_string())
print()

summarised = cleaned.summarise_encoded(
    encode_strings(["region"]),
    encode_strings(["sales", "sales", "product"]),
    encode_strings(["sum", "mean", "n_distinct"]),
    encode_strings(["total_sales", "avg_sales", "n_products"]),
)
print("group_by(region).summarise(sum, mean, n_distinct)")
print(summarised.to_string())
print()

right_join = grouped.join(targets, "region", "region", "right")
print("right join with targets")
print(right_join.to_string())
print()

longer = cleaned.pivot_longer_encoded(encode_strings(["sales"]), "metric", "value")
print("pivot_longer(sales)")
print(longer.to_string())
print()

new_region = datamunge.DataFrame()
new_region.add_string_column_encoded("region", encode_strings(["north"]))
new_region.add_string_column_encoded("product", encode_strings(["widget"]))
new_region.add_numeric_column("sales", dvector([6]))
bound = cleaned.select_encoded(encode_strings(["region", "product", "sales"])).bind_rows(new_region)
print("bind_rows with a new region")
print(bound.to_string())
