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

.df_sort_by <- function(df, column_name, ascending = TRUE) {
  DataFrame_sort_by__SWIG_0(df, column_name, ascending)
}

.df_drop_duplicates <- function(df, subset = character()) {
  if (length(subset) == 0) {
    return(DataFrame_drop_duplicates__SWIG_1(df))
  }
  DataFrame_drop_duplicates_encoded(df, .encode_strings(subset))
}

.df_group_by_sum <- function(df, key_columns, value_columns) {
  DataFrame_group_by_sum_encoded(df, .encode_strings(key_columns), .encode_strings(value_columns))
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
  if (!identical(left_suffix, "_x") || !identical(right_suffix, "_y")) {
    warning("DataFrame$join currently ignores custom suffix arguments in the R wrapper", call. = FALSE)
  }

  left_join <- identical(join_type, JoinType$Left) || identical(join_type, "left")
  DataFrame_join__SWIG_0(df, right, left_key, right_key, left_join)
}
