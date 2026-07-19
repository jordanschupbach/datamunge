# See examples/r/lm_ex.r for notes on the flat ClassName_method(obj, ...) call convention.
# Note: plain R character vectors work fine for vector<string> *parameters* everywhere
# (select, sort_by, group_by_sum, feature-column lists, ...) except specifically
# add_string_column(), whose generated wrapper is unconditionally broken -- that one needs
# the add_string_column_encoded()+encode_strings() workaround below.
library(datamunger)

encode_strings <- function(values) paste0(length(values), "\x1e", paste(values, collapse = "\x1f"))

sales <- DataFrame_empty()
DataFrame_add_string_column_encoded(sales, "region", encode_strings(c("west", "west", "east", "south", "south", "south")))
DataFrame_add_string_column_encoded(sales, "product", encode_strings(c("widget", "widget", "widget", "gizmo", "gizmo", "gizmo")))
DataFrame_add_numeric_column(sales, "sales", c(10.0, 10.0, 14.0, 8.0, 0.0, 11.0), c(1L, 1L, 1L, 1L, 0L, 1L))
DataFrame_add_string_column_encoded(sales, "quarter", encode_strings(c("Q1", "Q1", "Q1", "Q2", "Q2", "")), c(1L, 1L, 1L, 1L, 1L, 0L))

cat("raw data\n")
cat(DataFrame_to_string(sales), "\n\n")

# drop_duplicates() with an explicit subset is an *overloaded* method (default subset={}),
# and this package's R bindings can't pass a vector<string> to any overloaded method taking
# one (see examples/r/lm_ex.r notes) -- but no-subset drop_duplicates() checks all columns
# anyway, which is exactly what we want here since our subset would have been all 4 columns.
cleaned <- DataFrame_drop_duplicates(sales)
DataFrame_fill_null_string(cleaned, "quarter", "unknown")
DataFrame_fill_null_numeric(cleaned, "sales", 0.0)
cat("after drop_duplicates + fill_null\n")
cat(DataFrame_to_string(cleaned), "\n\n")

selected <- DataFrame_sort_by(DataFrame_select(cleaned, c("region", "sales", "quarter")), "sales", FALSE)
cat("selected + sorted\n")
cat(DataFrame_to_string(selected), "\n\n")

grouped <- DataFrame_sort_by(DataFrame_group_by_sum(cleaned, c("region"), c("sales")), "sales", FALSE)
cat("group_by_sum(region)\n")
cat(DataFrame_to_string(grouped), "\n\n")

targets <- DataFrame_empty()
DataFrame_add_string_column_encoded(targets, "region", encode_strings(c("west", "east", "south")))
DataFrame_add_numeric_column(targets, "target", c(18.0, 12.0, 25.0))
joined <- DataFrame_join(grouped, targets, "region", "region", TRUE)
cat("joined with targets\n")
cat(DataFrame_to_string(joined), "\n\n")

cat("nrows =", DataFrame_nrows(cleaned), " ncols =", DataFrame_ncols(cleaned), "\n")
cat("sales count =", DataFrame_numeric_count(cleaned, "sales"), "\n")
cat("sales nulls =", DataFrame_numeric_null_count(cleaned, "sales"), "\n")
cat("sales sum =", DataFrame_numeric_sum(cleaned, "sales"), "\n")
cat("sales mean =", DataFrame_numeric_mean(cleaned, "sales"), "\n")
