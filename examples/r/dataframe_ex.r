# CMD: nix develop .#r --command env R_LIBS_USER=build/r/library Rscript examples/r/dataframe_ex.r
# REPL: nix develop .#r --command env R_LIBS_USER=build/r/library R --no-save --quiet
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

# Tibble/dplyr-style piping: the $ accessor (see R/zzz_dataframe_api.R) makes every verb
# chainable since each one returns a new DataFrame, e.g. `df$mutate(...)$arrange(...)`.
# `cleaned` above was built with DataFrame_empty()/flat calls; from here on we use the more
# idiomatic `DataFrame()` + `$` form.
sales2 <- DataFrame$new()
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

new_region <- DataFrame$new()
new_region$add_column("region", c("north"))
new_region$add_column("product", c("widget"))
new_region$add_column("sales", c(6.0))
bound <- cleaned2$select(c("region", "product", "sales"))$bind_rows(new_region)
cat("bind_rows with a new region\n")
cat(bound$to_string(), "\n")
