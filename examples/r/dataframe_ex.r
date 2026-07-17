library(datamunger)

sales <- DataFrame()
sales$add_column("region", c("west", "west", "east", "south", "south", "south"))
sales$add_column("product", c("widget", "widget", "widget", "gizmo", "gizmo", "gizmo"))
sales$add_column("sales", c(10, 10, 14, 8, NA, 11))
sales$add_column("quarter", c("Q1", "Q1", "Q1", "Q2", "Q2", NA))

cat("raw data\n")
cat(sales$to_string(), "\n\n", sep = "")

cleaned <- sales$drop_duplicates(c("region", "product", "sales", "quarter"))
invisible(cleaned$fill_null("quarter", "unknown"))
invisible(cleaned$fill_null("sales", 0))
cat("after drop_duplicates + fill_null\n")
cat(cleaned$to_string(), "\n\n", sep = "")

selected <- cleaned$select(c("region", "sales", "quarter"))$sort_by("sales", FALSE)
cat("selected + sorted\n")
cat(selected$to_string(), "\n\n", sep = "")

grouped <- cleaned$group_by_sum("region", "sales")$sort_by("sales", FALSE)
cat("group_by_sum(region)\n")
cat(grouped$to_string(), "\n\n", sep = "")

targets <- DataFrame()
targets$add_column("region", c("west", "east", "south"))
targets$add_column("target", c(18, 12, 25))

joined <- grouped$join(targets, "region", "region", JoinType$Left)
cat("joined with targets\n")
cat(joined$to_string(), "\n\n", sep = "")

shape <- cleaned$shape()
summary <- cleaned$describe_numeric("sales")
cat("shape = (", shape[[1]], ", ", shape[[2]], ")\n", sep = "")
cat("sales count = ", summary$count, "\n", sep = "")
cat("sales nulls = ", summary$null_count, "\n", sep = "")
cat("sales sum = ", summary$sum, "\n", sep = "")
cat("sales mean = ", summary$mean, "\n", sep = "")
