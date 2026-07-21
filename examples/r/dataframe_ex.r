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
joined <- DataFrame_join(grouped, targets, "region", "region", "left")
cat("joined with targets\n")
cat(DataFrame_to_string(joined), "\n\n")

cat("nrows =", DataFrame_nrows(cleaned), " ncols =", DataFrame_ncols(cleaned), "\n")
cat("sales count =", DataFrame_numeric_count(cleaned, "sales"), "\n")
cat("sales nulls =", DataFrame_numeric_null_count(cleaned, "sales"), "\n")
cat("sales sum =", DataFrame_numeric_sum(cleaned, "sales"), "\n")
cat("sales mean =", DataFrame_numeric_mean(cleaned, "sales"), "\n")

# Tibble/dplyr-style piping: the $ accessor (see R/zzz_dataframe_api.R) makes every verb
# chainable since each one returns a new DataFrame, e.g. `df$mutate(...)$arrange(...)`.
# `cleaned` above was built with DataFrame_empty()/flat calls; from here on we use the more
# idiomatic `DataFrame()` + `$` form.
sales2 <- DataFrame()
sales2$add_column("region", c("west", "west", "east", "south", "south", "south"))
sales2$add_column("product", c("widget", "widget", "widget", "gizmo", "gizmo", "gizmo"))
sales2$add_column("sales", c(10.0, 10.0, 14.0, 8.0, NA, 11.0))
sales2$add_column("quarter", c("Q1", "Q1", "Q1", "Q2", "Q2", NA))

cleaned2 <- sales2$distinct(c("region", "product", "sales", "quarter"))
cleaned2$fill_null("quarter", "unknown")
cleaned2$fill_null("sales", 0.0)

piped <- cleaned2$mutate("tax", cleaned2$pull("sales") * 0.1)$
  rename("quarter", "period")$
  arrange(c("region", "sales"), c(TRUE, FALSE))$
  relocate(c("region", "sales"))$
  select(c("region", "sales", "tax", "period"))
cat("piped (mutate + arrange + rename + relocate + select)\n")
cat(piped$to_string(), "\n\n")

summarised <- cleaned2$group_by(c("region"))$summarise(
  total_sales = c("sales", "sum"),
  avg_sales = c("sales", "mean"),
  n_products = c("product", "n_distinct")
)
cat("group_by(region)$summarise(sum, mean, n_distinct)\n")
cat(summarised$to_string(), "\n\n")

right_join <- cleaned2$group_by_sum("region", "sales")$join(targets, "region", "region", JoinType$Right)
cat("right join with targets\n")
cat(right_join$to_string(), "\n\n")

longer <- cleaned2$pivot_longer(c("sales"), "metric", "value")
cat("pivot_longer(sales)\n")
cat(longer$to_string(), "\n\n")

new_region <- DataFrame()
new_region$add_column("region", c("north"))
new_region$add_column("product", c("widget"))
new_region$add_column("sales", c(6.0))
bound <- cleaned2$select(c("region", "product", "sales"))$bind_rows(new_region)
cat("bind_rows with a new region\n")
cat(bound$to_string(), "\n")
