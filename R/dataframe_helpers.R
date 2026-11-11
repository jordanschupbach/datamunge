JoinType <- list(
  Inner = "inner",
  Left = "left",
  Right = "right",
  Full = "full",
  Semi = "semi",
  Anti = "anti"
)

# R conveniences layered on the native DataFrame class (attached in zzz_dataframe_api.R).
# They convert R's NA-carrying vectors to the native (values, valid mask) pairs and give
# the dplyr verbs R-shaped arguments; everything else is the native method itself.

.df_valid_mask <- function(values) ifelse(is.na(values), 0L, 1L)

.df_add_column <- function(df, column_name, values) {
  if (is.character(values)) {
    df$add_string_column(column_name, ifelse(is.na(values), "", values), .df_valid_mask(values))
  } else if (is.numeric(values)) {
    df$add_numeric_column(column_name, ifelse(is.na(values), 0, as.numeric(values)), .df_valid_mask(values))
  } else {
    stop("DataFrame$add_column only supports numeric/integer or character vectors")
  }
  invisible(df)
}

.df_mutate <- function(df, column_name, values) {
  if (is.character(values))
    return(df$mutate_string(column_name, ifelse(is.na(values), "", values), .df_valid_mask(values)))
  if (is.numeric(values))
    return(df$mutate_numeric(column_name, ifelse(is.na(values), 0, as.numeric(values)), .df_valid_mask(values)))
  stop("DataFrame$mutate only supports numeric/integer or character vectors")
}

.df_fill_null <- function(df, column_name, value) {
  if (is.character(value)) df$fill_null_string(column_name, value[[1]])
  else if (is.numeric(value)) df$fill_null_numeric(column_name, as.numeric(value[[1]]))
  else stop("DataFrame$fill_null only supports numeric/integer or character scalars")
  invisible(df)
}

.df_pull <- function(df, column_name) {
  if (df$is_numeric_column(column_name)) {
    values <- df$pull_numeric(column_name)
    values[df$pull_numeric_valid(column_name) == 0L] <- NA_real_
  } else {
    values <- df$pull_string(column_name)
    values[df$pull_string_valid(column_name) == 0L] <- NA_character_
  }
  values
}

# specs: named `...`, each a c(column, func) pair, e.g. total = c("sales", "sum"),
# n = c("", "count"). func is one of "sum", "mean", "min", "max", "median", "stddev",
# "count", "n_distinct".
.df_summarise <- function(df, key_columns, ...) {
  specs <- list(...)
  result_names <- names(specs)
  if (length(specs) == 0 || is.null(result_names) || any(result_names == "")) {
    stop("DataFrame$summarise: every aggregation must be named, e.g. total = c('sales', 'sum')")
  }
  agg_columns <- vapply(specs, function(spec) spec[[1]], character(1), USE.NAMES = FALSE)
  agg_funcs <- vapply(specs, function(spec) spec[[2]], character(1), USE.NAMES = FALSE)
  df$.native_summarise(as.character(key_columns), agg_columns, agg_funcs, result_names)
}

# Mirrors dplyr's group_by(df, ...) %>% summarise(...)/count(): a small handle whose
# $summarise(...)/$count() have `key_columns` already bound.
.df_group_by <- function(df, key_columns) {
  list(
    summarise = function(...) .df_summarise(df, key_columns, ...),
    count = function(count_column_name = "n") df$count(as.character(key_columns), count_column_name)
  )
}

.df_describe_numeric <- function(df, column_name) {
  list(
    count = as.integer(df$numeric_count(column_name)),
    null_count = as.integer(df$numeric_null_count(column_name)),
    sum = df$numeric_sum(column_name),
    mean = df$numeric_mean(column_name),
    min = df$numeric_min(column_name),
    max = df$numeric_max(column_name)
  )
}
