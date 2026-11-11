# CMD: nix develop .#r --command Rscript examples/r/datamunge_ex.r
library(datamunger)


sales <- DataFrame$empty()
sales$add_string_column("region", c("west", "west", "east", "south", "south", "south"))
sales$add_string_column("product", c("widget", "widget", "widget", "gizmo", "gizmo", "gizmo"))
sales$add_numeric_column("sales", c(10.0, 10.0, 14.0, 8.0, 0.0, 11.0), c(1L, 1L, 1L, 1L, 0L, 1L))
sales$add_string_column("quarter", c("Q1", "Q1", "Q1", "Q2", "Q2", ""), c(1L, 1L, 1L, 1L, 1L, 0L))

cat("raw data\n")
cat(sales$to_string(), "\n\n")

# drop_duplicates() with an explicit subset is an *overloaded* method (default subset={}),
# and this package's R bindings can't pass a vector<string> to any overloaded method taking
# one (see examples/r/lm_ex.r notes) -- but no-subset drop_duplicates() checks all columns
# anyway, which is exactly what we want here since our subset would have been all 4 columns.
cleaned <- sales$drop_duplicates()
cleaned$fill_null_string("quarter", "unknown")
cleaned$fill_null_numeric("sales", 0.0)
cat("after drop_duplicates + fill_null\n")
cat(cleaned$to_string(), "\n\n")

selected <- cleaned$select(c("region", "sales", "quarter"))$sort_by("sales", FALSE)
cat("selected + sorted\n")
cat(selected$to_string(), "\n\n")

grouped <- cleaned$group_by_sum(c("region"), c("sales"))$sort_by("sales", FALSE)
cat("group_by_sum(region)\n")
cat(grouped$to_string(), "\n\n")

targets <- DataFrame$empty()
targets$add_string_column("region", c("west", "east", "south"))
targets$add_numeric_column("target", c(18.0, 12.0, 25.0))
joined <- grouped$join(targets, "region", "region", "left")
cat("joined with targets\n")
cat(joined$to_string(), "\n\n")

cat("nrows =", cleaned$nrows(), " ncols =", cleaned$ncols(), "\n")
cat("sales count =", cleaned$numeric_count("sales"), "\n")
cat("sales nulls =", cleaned$numeric_null_count("sales"), "\n")
cat("sales sum =", cleaned$numeric_sum("sales"), "\n")
cat("sales mean =", cleaned$numeric_mean("sales"), "\n")
