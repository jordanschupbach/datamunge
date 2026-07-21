.encode_strings <- function(values) {
  values <- as.character(values)
  paste0(length(values), "\x1e", paste(values, collapse = "\x1f"))
}

.df_add_column <- function(df, column_name, values) {
  if (is.character(values)) {
    valid_mask <- ifelse(is.na(values), 0L, 1L)
    sanitized <- ifelse(is.na(values), "", as.character(values))
    DataFrame_add_string_column_encoded(df, column_name, .encode_strings(sanitized), valid_mask)
    return(invisible(df))
  }

  if (is.numeric(values) || is.integer(values)) {
    valid_mask <- ifelse(is.na(values), 0L, 1L)
    sanitized <- ifelse(is.na(values), 0.0, as.numeric(values))
    DataFrame_add_numeric_column__SWIG_0(df, column_name, sanitized, valid_mask)
    return(invisible(df))
  }

  stop("DataFrame$add_column only supports numeric/integer or character vectors")
}

.df_fill_null <- function(df, column_name, value) {
  if (is.character(value)) {
    invisible(DataFrame_fill_null_string(df, column_name, as.character(value[[1]])))
    return(invisible(df))
  }

  if (is.numeric(value) || is.integer(value)) {
    invisible(DataFrame_fill_null_numeric(df, column_name, as.numeric(value[[1]])))
    return(invisible(df))
  }

  stop("DataFrame$fill_null only supports numeric/integer or character scalars")
}

.df_select <- function(df, columns) {
  DataFrame_select_encoded(df, .encode_strings(columns))
}

.df_relocate <- function(df, columns, after = "") {
  DataFrame_relocate_encoded__SWIG_0(df, .encode_strings(columns), after)
}

.df_sort_by <- function(df, column_name, ascending = TRUE) {
  DataFrame_sort_by__SWIG_0(df, column_name, ascending)
}

.df_arrange <- function(df, columns, ascending = TRUE) {
  ascending <- as.integer(rep_len(as.logical(ascending), length(columns)))
  DataFrame_arrange_encoded__SWIG_0(df, .encode_strings(columns), ascending)
}

.df_drop_duplicates <- function(df, subset = character()) {
  if (length(subset) == 0) {
    return(DataFrame_drop_duplicates__SWIG_1(df))
  }
  DataFrame_drop_duplicates_encoded(df, .encode_strings(subset))
}

.df_distinct <- function(df, subset = character()) {
  DataFrame_distinct_encoded(df, .encode_strings(subset))
}

.df_mutate <- function(df, column_name, values) {
  if (is.character(values)) {
    valid_mask <- ifelse(is.na(values), 0L, 1L)
    sanitized <- ifelse(is.na(values), "", as.character(values))
    return(DataFrame_mutate_string_encoded__SWIG_0(df, column_name, .encode_strings(sanitized), valid_mask))
  }

  if (is.numeric(values) || is.integer(values)) {
    valid_mask <- ifelse(is.na(values), 0L, 1L)
    sanitized <- ifelse(is.na(values), 0.0, as.numeric(values))
    return(DataFrame_mutate_numeric__SWIG_0(df, column_name, sanitized, valid_mask))
  }

  stop("DataFrame$mutate only supports numeric/integer or character vectors")
}

.df_rename <- function(df, old_name, new_name) {
  DataFrame_rename(df, old_name, new_name)
}

.df_pull <- function(df, column_name) {
  if (DataFrame_is_numeric_column(df, column_name)) {
    values <- DataFrame_pull_numeric(df, column_name)
    valid <- DataFrame_pull_numeric_valid(df, column_name)
    values[valid == 0L] <- NA_real_
    return(values)
  }

  values <- DataFrame_pull_string(df, column_name)
  valid <- DataFrame_pull_string_valid(df, column_name)
  values[valid == 0L] <- NA_character_
  values
}

.df_n_distinct <- function(df, column_name) {
  as.integer(DataFrame_n_distinct(df, column_name))
}

.df_group_by_sum <- function(df, key_columns, value_columns) {
  DataFrame_group_by_sum_encoded(df, .encode_strings(key_columns), .encode_strings(value_columns))
}

.df_count <- function(df, key_columns, count_column_name = "n") {
  DataFrame_count_encoded__SWIG_0(df, .encode_strings(key_columns), count_column_name)
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

  agg_columns <- vapply(specs, function(spec) spec[[1]], character(1))
  agg_funcs <- vapply(specs, function(spec) spec[[2]], character(1))
  DataFrame_summarise_encoded(df, .encode_strings(key_columns), .encode_strings(agg_columns),
                              .encode_strings(agg_funcs), .encode_strings(result_names))
}

# Mirrors dplyr's group_by(df, ...) %>% summarise(...)/count(): returns a small handle
# whose $summarise(...)/$count() defer to .df_summarise()/.df_count() with `key_columns`
# already bound, so `df$group_by(c("region"))$summarise(total = c("sales", "sum"))` reads
# like the dplyr pipeline it mirrors.
.df_group_by <- function(df, key_columns) {
  list(
    summarise = function(...) .df_summarise(df, key_columns, ...),
    count = function(count_column_name = "n") .df_count(df, key_columns, count_column_name)
  )
}

.df_pivot_longer <- function(df, value_columns, names_to = "name", values_to = "value") {
  DataFrame_pivot_longer_encoded__SWIG_0(df, .encode_strings(value_columns), names_to, values_to)
}

.df_pivot_wider <- function(df, names_from, values_from, id_columns = character()) {
  DataFrame_pivot_wider_encoded(df, names_from, values_from, .encode_strings(id_columns))
}

.df_bind_rows <- function(df, other) {
  DataFrame_bind_rows(df, other)
}

.df_bind_cols <- function(df, other) {
  DataFrame_bind_cols(df, other)
}

.df_shape <- function(df) {
  c(as.integer(DataFrame_nrows(df)), as.integer(DataFrame_ncols(df)))
}

.df_to_string <- function(df, max_rows = 10L) {
  DataFrame_to_string__SWIG_0(df, as.integer(max_rows))
}

.df_describe_numeric <- function(df, column_name) {
  list(
    count = as.integer(DataFrame_numeric_count(df, column_name)),
    null_count = as.integer(DataFrame_numeric_null_count(df, column_name)),
    sum = DataFrame_numeric_sum(df, column_name),
    mean = DataFrame_numeric_mean(df, column_name),
    min = DataFrame_numeric_min(df, column_name),
    max = DataFrame_numeric_max(df, column_name)
  )
}

.df_join <- function(df, right, left_key, right_key, join_type = JoinType$Inner, left_suffix = "_x", right_suffix = "_y") {
  DataFrame_join__SWIG_0(df, right, left_key, right_key, join_type, left_suffix, right_suffix)
}
