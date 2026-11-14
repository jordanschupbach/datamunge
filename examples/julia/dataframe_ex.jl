# CMD: just TARGET=dataframe_ex run-julia

# DataFrame basics: nulls, drop_duplicates, fill_null, select/sort, group-by, join and summaries.
using Datamunge

const sales = let frame = DataFrame()
    add_string_column(frame, "region", ["west", "west", "east", "south", "south", "south"])
    add_string_column(frame, "product", ["widget", "widget", "widget", "gizmo", "gizmo", "gizmo"])
    add_numeric_column(frame, "sales", [10.0, 10.0, 14.0, 8.0, 0.0, 11.0], [1, 1, 1, 1, 0, 1])
    add_string_column(frame, "quarter", ["Q1", "Q1", "Q1", "Q2", "Q2", ""], [1, 1, 1, 1, 1, 0])
    frame
end

println("raw data\n$(to_string(sales))\n")

const cleaned = drop_duplicates(sales, ["region", "product", "sales", "quarter"])
fill_null_string(cleaned, "quarter", "unknown")
fill_null_numeric(cleaned, "sales", 0.0)
println("after drop_duplicates + fill_null\n$(to_string(cleaned))\n")

let selected = sort_by(select(cleaned, ["region", "sales", "quarter"]), "sales", false)
    println("selected + sorted\n$(to_string(selected))\n")
end

const grouped = sort_by(group_by_sum(cleaned, ["region"], ["sales"]), "sales", false)
println("group_by_sum(region)\n$(to_string(grouped))\n")

let targets = DataFrame()
    add_string_column(targets, "region", ["west", "east", "south"])
    add_numeric_column(targets, "target", [18.0, 12.0, 25.0])
    println("joined with targets\n$(to_string(join(grouped, targets, "region", "region", "left")))\n")
end

let (rows, cols) = shape(cleaned)
    println("shape = ($rows, $cols)")
end
println("sales count = $(numeric_count(cleaned, "sales"))")
println("sales nulls = $(numeric_null_count(cleaned, "sales"))")
println("sales sum = $(numeric_sum(cleaned, "sales"))")
println("sales mean = $(numeric_mean(cleaned, "sales"))")
