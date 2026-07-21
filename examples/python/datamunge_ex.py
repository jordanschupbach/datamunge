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
