# Tibble/dplyr-style verbs on the native DataFrame, all chainable since every transform
# returns a new DataFrame:
#   df$mutate("tax", sales * 0.1)$arrange("sales", FALSE)
# Native methods keep their names; where an R-shaped verb replaces one (arrange, summarise,
# shape), the native method stays reachable as .native_<name>.
local({
  extend <- function(name, fn) {
    native <- DataFrame$public_methods[[name]]
    if (!is.null(native)) DataFrame$set("public", paste0(".native_", name), native)
    DataFrame$set("public", name, fn, overwrite = TRUE)
  }
  extend("add_column", function(column_name, values) .df_add_column(self, column_name, values))
  extend("fill_null", function(column_name, value) .df_fill_null(self, column_name, value))
  extend("mutate", function(column_name, values) .df_mutate(self, column_name, values))
  extend("pull", function(column_name) .df_pull(self, column_name))
  extend("arrange", function(columns, ascending = TRUE) {
    columns <- as.character(columns)
    self$.native_arrange(columns, as.integer(rep_len(as.logical(ascending), length(columns))))
  })
  extend("summarise", function(key_columns, ...) .df_summarise(self, key_columns, ...))
  extend("summarize", function(key_columns, ...) .df_summarise(self, key_columns, ...))
  extend("group_by", function(key_columns) .df_group_by(self, key_columns))
  extend("shape", function() as.integer(unlist(self$.native_shape())))
  extend("describe_numeric", function(column_name) .df_describe_numeric(self, column_name))
})
