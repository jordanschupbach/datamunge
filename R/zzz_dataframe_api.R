JoinType <- list(
  Inner = "inner",
  Left = "left",
  Right = "right",
  Full = "full",
  Semi = "semi",
  Anti = "anti"
)

# Delegates to DataFrame_empty() rather than calling the raw SWIG no-arg constructor
# directly: see DataFrame::empty()'s doc comment in datamunge.hpp for why the plain
# constructor's ownership-flagged pointer breaks method dispatch on the R backend.
DataFrame <- function() {
  get("DataFrame_empty", envir = asNamespace("datamunger"))()
}

# Tibble/dplyr-style verbs, all chainable since every transform returns a new DataFrame:
#   df$mutate("tax", sales * 0.1)$filter(...)$arrange("sales", FALSE)
# (there is no $filter() -- filter() needs an arbitrary row predicate, which can't cross
# SWIG; use $pull()/vectorized R logic plus $select()/take_rows-style indexing instead.)
setMethod("$", "_p_datamunge__DataFrame", function(x, name) {
  public_accessors <- list(
    add_column = function(column_name, values) .df_add_column(x, column_name, values),
    fill_null = function(column_name, value) .df_fill_null(x, column_name, value),
    mutate = function(column_name, values) .df_mutate(x, column_name, values),
    rename = function(old_name, new_name) .df_rename(x, old_name, new_name),
    select = function(columns) .df_select(x, columns),
    relocate = function(columns, after = "") .df_relocate(x, columns, after),
    pull = function(column_name) .df_pull(x, column_name),
    sort_by = function(column_name, ascending = TRUE) .df_sort_by(x, column_name, ascending),
    arrange = function(columns, ascending = TRUE) .df_arrange(x, columns, ascending),
    drop_duplicates = function(subset = character()) .df_drop_duplicates(x, subset),
    distinct = function(subset = character()) .df_distinct(x, subset),
    n_distinct = function(column_name) .df_n_distinct(x, column_name),
    group_by_sum = function(key_columns, value_columns) .df_group_by_sum(x, key_columns, value_columns),
    count = function(key_columns, count_column_name = "n") .df_count(x, key_columns, count_column_name),
    summarise = function(key_columns, ...) .df_summarise(x, key_columns, ...),
    summarize = function(key_columns, ...) .df_summarise(x, key_columns, ...),
    group_by = function(key_columns) .df_group_by(x, key_columns),
    pivot_longer = function(value_columns, names_to = "name", values_to = "value") {
      .df_pivot_longer(x, value_columns, names_to, values_to)
    },
    pivot_wider = function(names_from, values_from, id_columns = character()) {
      .df_pivot_wider(x, names_from, values_from, id_columns)
    },
    bind_rows = function(other) .df_bind_rows(x, other),
    bind_cols = function(other) .df_bind_cols(x, other),
    join = function(right, left_key, right_key, join_type = JoinType$Inner, left_suffix = "_x", right_suffix = "_y") {
      .df_join(x, right, left_key, right_key, join_type, left_suffix, right_suffix)
    },
    shape = function() .df_shape(x),
    to_string = function(max_rows = 10L) .df_to_string(x, max_rows),
    describe_numeric = function(column_name) .df_describe_numeric(x, column_name)
  )

  raw_accessors <- list(
    nrows = function() DataFrame_nrows(x),
    ncols = function() DataFrame_ncols(x),
    columns = function() DataFrame_columns(x),
    numeric_count = function(column_name) DataFrame_numeric_count(x, column_name),
    numeric_null_count = function(column_name) DataFrame_numeric_null_count(x, column_name),
    numeric_sum = function(column_name) DataFrame_numeric_sum(x, column_name),
    numeric_mean = function(column_name) DataFrame_numeric_mean(x, column_name)
  )

  if (!is.null(public_accessors[[name]])) {
    return(public_accessors[[name]])
  }
  if (!is.null(raw_accessors[[name]])) {
    return(raw_accessors[[name]])
  }
  NULL
})
