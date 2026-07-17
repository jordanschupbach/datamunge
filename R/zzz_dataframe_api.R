JoinType <- list(
  Inner = "inner",
  Left = "left"
)

DataFrame <- function() {
  get("DataFrame", envir = asNamespace("datamunger"))()
}

setMethod("$", "_p_datamunge__DataFrame", function(x, name) {
  public_accessors <- list(
    add_column = function(column_name, values) .df_add_column(x, column_name, values),
    fill_null = function(column_name, value) .df_fill_null(x, column_name, value),
    select = function(columns) .df_select(x, columns),
    sort_by = function(column_name, ascending = TRUE) .df_sort_by(x, column_name, ascending),
    drop_duplicates = function(subset = character()) .df_drop_duplicates(x, subset),
    group_by_sum = function(key_columns, value_columns) .df_group_by_sum(x, key_columns, value_columns),
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
