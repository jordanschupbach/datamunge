classdef DataFrame < datamunge.NativeObject
  % DataFrame  SWIG-friendly facade for the C++ dataframe API exposed to bindings.
  %
  %   obj = datamunge.DataFrame ()
  %   obj = datamunge.DataFrame (values, names)
  %
  %   Methods:
  %     nrows  C++ datamunge::DataFrame::nrows.
  %     ncols  C++ datamunge::DataFrame::ncols.
  %     shape  C++ datamunge::DataFrame::shape.
  %     columns  C++ datamunge::DataFrame::columns.
  %     add_numeric_column  C++ datamunge::DataFrame::add_numeric_column.
  %     add_string_column  C++ datamunge::DataFrame::add_string_column.
  %     add_string_column_encoded  C++ datamunge::DataFrame::add_string_column_encoded.
  %     fill_null_numeric  C++ datamunge::DataFrame::fill_null_numeric.
  %     fill_null_string  C++ datamunge::DataFrame::fill_null_string.
  %     mutate_numeric  dplyr::mutate()-style upsert: adds column_name if absent, replaces it (same type) if present. Always returns a new DataFrame, so it composes into a pipe.
  %     mutate_string  C++ datamunge::DataFrame::mutate_string.
  %     mutate_string_encoded  C++ datamunge::DataFrame::mutate_string_encoded.
  %     rename  Non-mutating, chainable single-pair rename.
  %     select  C++ datamunge::DataFrame::select.
  %     select_encoded  C++ datamunge::DataFrame::select_encoded.
  %     relocate  dplyr::relocate()-style column reorder: moves `columns` to the front (default) or immediately after the column named `after`.
  %     relocate_encoded  C++ datamunge::DataFrame::relocate_encoded.
  %     sort_by  C++ datamunge::DataFrame::sort_by.
  %     arrange  dplyr::arrange()-style multi-key sort. `ascending` defaults to all-true; when provided it must have the same length as `columns`.
  %     arrange_encoded  C++ datamunge::DataFrame::arrange_encoded.
  %     drop_duplicates  C++ datamunge::DataFrame::drop_duplicates.
  %     drop_duplicates_encoded  C++ datamunge::DataFrame::drop_duplicates_encoded.
  %     distinct  dplyr::distinct() alias for drop_duplicates().
  %     distinct_encoded  C++ datamunge::DataFrame::distinct_encoded.
  %     pull_numeric  dplyr::pull()-style column extraction. Nulls come back as NaN (numeric) / "" (string) in the value vector; check pull_numeric_valid()/pull_string_valid() (1 = present, 0 = null, same convention as add_numeric_column's valid_mask) if nulls matter.
  %     pull_numeric_valid  C++ datamunge::DataFrame::pull_numeric_valid.
  %     pull_string  C++ datamunge::DataFrame::pull_string.
  %     pull_string_valid  C++ datamunge::DataFrame::pull_string_valid.
  %     n_distinct  Number of distinct values in column_name; a null counts as one additional distinct value if present.
  %     group_by_sum  C++ datamunge::DataFrame::group_by_sum.
  %     group_by_sum_encoded  C++ datamunge::DataFrame::group_by_sum_encoded.
  %     count  dplyr::count()-style grouped row counts, default result column name "n".
  %     count_encoded  C++ datamunge::DataFrame::count_encoded.
  %     summarise  General dplyr::summarise()-style aggregation: one output row per distinct combination of `key_columns`, with one output column per (agg_columns[i], agg_funcs[i], result_names[i]) triple -- all three arrays must have the same length. agg_funcs entries are one of "sum", "mean", "min", "max", "median", "stddev", "count", "n_distinct" ("count" ignores the corresponding agg_columns entry, which may be ""); a "" result_names entry defaults to the agg_columns entry (or "n" for count).
  %     summarise_encoded  C++ datamunge::DataFrame::summarise_encoded.
  %     pivot_longer  dplyr::pivot_longer()-style reshape: stacks `value_columns` into two new columns (`names_to` holding the source column name, `values_to` holding its value).
  %     pivot_longer_encoded  C++ datamunge::DataFrame::pivot_longer_encoded.
  %     pivot_wider  dplyr::pivot_wider()-style reshape: `names_from` (a string column) supplies new column names, `values_from` supplies their values; `id_columns` defaults to every other column.
  %     pivot_wider_encoded  C++ datamunge::DataFrame::pivot_wider_encoded.
  %     bind_rows  dplyr::bind_rows()-style row union: aligns columns by name (unlike concat_rows, which isn't exposed here), null-filling any column present in only one frame.
  %     bind_cols  dplyr::bind_cols()-style column union: both frames must have the same row count and disjoint column names.
  %     join  C++ datamunge::DataFrame::join.
  %     numeric_count  C++ datamunge::DataFrame::numeric_count.
  %     numeric_null_count  C++ datamunge::DataFrame::numeric_null_count.
  %     numeric_sum  C++ datamunge::DataFrame::numeric_sum.
  %     numeric_mean  C++ datamunge::DataFrame::numeric_mean.
  %     numeric_min  C++ datamunge::DataFrame::numeric_min.
  %     numeric_max  C++ datamunge::DataFrame::numeric_max.
  %     to_string  C++ datamunge::DataFrame::to_string.
  %     to_md  The frame as a Markdown table (numeric columns right-aligned, at most `digits` decimals each), like knitr::kable(); see dstruct::DataFrame::to_md().
  %     is_numeric_column  true if column_name holds numeric values, false if it holds strings.
  %     is_null  C++ datamunge::DataFrame::is_null.
  %     numeric_at  The value at (column_name, row_index); throws if the cell is null or the column isn't numeric.
  %     string_at  The value at (column_name, row_index); throws if the cell is null or the column isn't string-typed.
  %
  %   Static methods:
  %     empty  Equivalent to the default constructor, exposed as a static factory (like read_csv()) as a workaround for a SWIG R-backend bug where the plain no-argument constructor's ownership-flagged pointer breaks method dispatch on the result.
  %     read_csv  Reads a whole CSV file, typing columns automatically; see datasets::read_csv().
  %     parse_csv  Parses CSV text, typing columns automatically; see datasets::parse_csv().
  %     from_image_batch  An ImageLoader batch as a table (`pixel_0`.., `label`, `index`); see datasets::to_dataframe().
  %
  %   C++: class datamunge::DataFrame
  methods
    function obj = DataFrame (varargin)
      % DataFrame  Construct a C++ datamunge::DataFrame.
      %
      %   obj = datamunge.DataFrame ()
      %   obj = datamunge.DataFrame (values, names)
      %
      %   Example:
      %     const dm::DataFrame data(dm::seql(0, 1, 3), {"x"});
      %     std::print("{}\n", data.nrows()); // 3
      %
      %   Copy a numeric vector into a one-column DataFrame.
      %
      %   Errors:
      %     datamunge:invalid_argument  For invalid column names. (C++ std::invalid_argument)
      %
      %   C++: datamunge::DataFrame::DataFrame()
      %   C++: datamunge::DataFrame::DataFrame(const std::vector<double> & values, const std::vector<std::string> & names)
      obj@datamunge.NativeObject(struct('rosetta_skip__', true));
      if nargin == 1 && isstruct (varargin{1}) && isfield (varargin{1}, 'rosetta_skip__')
        return;
      end
      __datamunge__(-2, obj, 2098, false, varargin{:});
    end
    function r = nrows (obj)
      % nrows  C++ datamunge::DataFrame::nrows.
      %
      %   r = obj.nrows ()
      %
      %   Output:
      %     r  integer
      %
      %   C++: std::size_t datamunge::DataFrame::nrows() const
      r = __datamunge__(2041, obj);
    end
    function r = ncols (obj)
      % ncols  C++ datamunge::DataFrame::ncols.
      %
      %   r = obj.ncols ()
      %
      %   Output:
      %     r  integer
      %
      %   C++: std::size_t datamunge::DataFrame::ncols() const
      r = __datamunge__(2042, obj);
    end
    function r = shape (obj)
      % shape  C++ datamunge::DataFrame::shape.
      %
      %   r = obj.shape ()
      %
      %   Output:
      %     r  integer vector
      %
      %   C++: std::vector<std::size_t> datamunge::DataFrame::shape() const
      r = __datamunge__(2043, obj);
    end
    function r = columns (obj)
      % columns  C++ datamunge::DataFrame::columns.
      %
      %   r = obj.columns ()
      %
      %   Output:
      %     r  cell array of char
      %
      %   C++: std::vector<std::string> datamunge::DataFrame::columns() const
      r = __datamunge__(2044, obj);
    end
    function add_numeric_column (obj, varargin)
      % add_numeric_column  C++ datamunge::DataFrame::add_numeric_column.
      %
      %   obj.add_numeric_column (column_name, values)
      %   obj.add_numeric_column (column_name, values, valid_mask)
      %
      %   Inputs:
      %     column_name  char row
      %     values  double vector
      %     valid_mask  integer vector, default the C++ default
      %
      %   C++: void datamunge::DataFrame::add_numeric_column(const std::string & column_name, const std::vector<double> & values, const std::vector<int> & valid_mask = {})
      __datamunge__(2045, obj, varargin{:});
    end
    function add_string_column (obj, varargin)
      % add_string_column  C++ datamunge::DataFrame::add_string_column.
      %
      %   obj.add_string_column (column_name, values)
      %   obj.add_string_column (column_name, values, valid_mask)
      %
      %   Inputs:
      %     column_name  char row
      %     values  cell array of char
      %     valid_mask  integer vector, default the C++ default
      %
      %   C++: void datamunge::DataFrame::add_string_column(const std::string & column_name, const std::vector<std::string> & values, const std::vector<int> & valid_mask = {})
      __datamunge__(2046, obj, varargin{:});
    end
    function add_string_column_encoded (obj, varargin)
      % add_string_column_encoded  C++ datamunge::DataFrame::add_string_column_encoded.
      %
      %   obj.add_string_column_encoded (column_name, encoded_values)
      %   obj.add_string_column_encoded (column_name, encoded_values, valid_mask)
      %
      %   Inputs:
      %     column_name  char row
      %     encoded_values  char row
      %     valid_mask  integer vector, default the C++ default
      %
      %   C++: void datamunge::DataFrame::add_string_column_encoded(const std::string & column_name, const std::string & encoded_values, const std::vector<int> & valid_mask = {})
      __datamunge__(2047, obj, varargin{:});
    end
    function fill_null_numeric (obj, column_name, value_)
      % fill_null_numeric  C++ datamunge::DataFrame::fill_null_numeric.
      %
      %   obj.fill_null_numeric (column_name, value_)
      %
      %   Inputs:
      %     column_name  char row
      %     value_  double
      %
      %   C++: void datamunge::DataFrame::fill_null_numeric(const std::string & column_name, double value)
      __datamunge__(2048, obj, column_name, value_);
    end
    function fill_null_string (obj, column_name, value_)
      % fill_null_string  C++ datamunge::DataFrame::fill_null_string.
      %
      %   obj.fill_null_string (column_name, value_)
      %
      %   Inputs:
      %     column_name  char row
      %     value_  char row
      %
      %   C++: void datamunge::DataFrame::fill_null_string(const std::string & column_name, const std::string & value)
      __datamunge__(2049, obj, column_name, value_);
    end
    function r = mutate_numeric (obj, varargin)
      % mutate_numeric  dplyr::mutate()-style upsert: adds column_name if absent, replaces it (same type) if present. Always returns a new DataFrame, so it composes into a pipe.
      %
      %   r = obj.mutate_numeric (column_name, values)
      %   r = obj.mutate_numeric (column_name, values, valid_mask)
      %
      %   Inputs:
      %     column_name  char row
      %     values  double vector
      %     valid_mask  integer vector, default the C++ default
      %
      %   Output:
      %     r  datamunge.DataFrame or []
      %
      %   C++: DataFrame * datamunge::DataFrame::mutate_numeric(const std::string & column_name, const std::vector<double> & values, const std::vector<int> & valid_mask = {}) const
      r = __datamunge__(2050, obj, varargin{:});
    end
    function r = mutate_string (obj, varargin)
      % mutate_string  C++ datamunge::DataFrame::mutate_string.
      %
      %   r = obj.mutate_string (column_name, values)
      %   r = obj.mutate_string (column_name, values, valid_mask)
      %
      %   Inputs:
      %     column_name  char row
      %     values  cell array of char
      %     valid_mask  integer vector, default the C++ default
      %
      %   Output:
      %     r  datamunge.DataFrame or []
      %
      %   C++: DataFrame * datamunge::DataFrame::mutate_string(const std::string & column_name, const std::vector<std::string> & values, const std::vector<int> & valid_mask = {}) const
      r = __datamunge__(2051, obj, varargin{:});
    end
    function r = mutate_string_encoded (obj, varargin)
      % mutate_string_encoded  C++ datamunge::DataFrame::mutate_string_encoded.
      %
      %   r = obj.mutate_string_encoded (column_name, encoded_values)
      %   r = obj.mutate_string_encoded (column_name, encoded_values, valid_mask)
      %
      %   Inputs:
      %     column_name  char row
      %     encoded_values  char row
      %     valid_mask  integer vector, default the C++ default
      %
      %   Output:
      %     r  datamunge.DataFrame or []
      %
      %   C++: DataFrame * datamunge::DataFrame::mutate_string_encoded(const std::string & column_name, const std::string & encoded_values, const std::vector<int> & valid_mask = {}) const
      r = __datamunge__(2052, obj, varargin{:});
    end
    function r = rename (obj, old_name, new_name)
      % rename  Non-mutating, chainable single-pair rename.
      %
      %   r = obj.rename (old_name, new_name)
      %
      %   Inputs:
      %     old_name  char row
      %     new_name  char row
      %
      %   Output:
      %     r  datamunge.DataFrame or []
      %
      %   C++: DataFrame * datamunge::DataFrame::rename(const std::string & old_name, const std::string & new_name) const
      r = __datamunge__(2053, obj, old_name, new_name);
    end
    function r = select (obj, selected_columns)
      % select  C++ datamunge::DataFrame::select.
      %
      %   r = obj.select (selected_columns)
      %
      %   Inputs:
      %     selected_columns  cell array of char
      %
      %   Output:
      %     r  datamunge.DataFrame or []
      %
      %   C++: DataFrame * datamunge::DataFrame::select(const std::vector<std::string> & selected_columns) const
      r = __datamunge__(2054, obj, selected_columns);
    end
    function r = select_encoded (obj, encoded_columns)
      % select_encoded  C++ datamunge::DataFrame::select_encoded.
      %
      %   r = obj.select_encoded (encoded_columns)
      %
      %   Inputs:
      %     encoded_columns  char row
      %
      %   Output:
      %     r  datamunge.DataFrame or []
      %
      %   C++: DataFrame * datamunge::DataFrame::select_encoded(const std::string & encoded_columns) const
      r = __datamunge__(2055, obj, encoded_columns);
    end
    function r = relocate (obj, varargin)
      % relocate  dplyr::relocate()-style column reorder: moves `columns` to the front (default) or immediately after the column named `after`.
      %
      %   r = obj.relocate (columns)
      %   r = obj.relocate (columns, after)
      %
      %   Inputs:
      %     columns  cell array of char
      %     after  char row, default ''
      %
      %   Output:
      %     r  datamunge.DataFrame or []
      %
      %   C++: DataFrame * datamunge::DataFrame::relocate(const std::vector<std::string> & columns, const std::string & after = "") const
      r = __datamunge__(2056, obj, varargin{:});
    end
    function r = relocate_encoded (obj, varargin)
      % relocate_encoded  C++ datamunge::DataFrame::relocate_encoded.
      %
      %   r = obj.relocate_encoded (encoded_columns)
      %   r = obj.relocate_encoded (encoded_columns, after)
      %
      %   Inputs:
      %     encoded_columns  char row
      %     after  char row, default ''
      %
      %   Output:
      %     r  datamunge.DataFrame or []
      %
      %   C++: DataFrame * datamunge::DataFrame::relocate_encoded(const std::string & encoded_columns, const std::string & after = "") const
      r = __datamunge__(2057, obj, varargin{:});
    end
    function r = sort_by (obj, varargin)
      % sort_by  C++ datamunge::DataFrame::sort_by.
      %
      %   r = obj.sort_by (column_name)
      %   r = obj.sort_by (column_name, ascending)
      %
      %   Inputs:
      %     column_name  char row
      %     ascending  logical, default true
      %
      %   Output:
      %     r  datamunge.DataFrame or []
      %
      %   C++: DataFrame * datamunge::DataFrame::sort_by(const std::string & column_name, bool ascending = true) const
      r = __datamunge__(2058, obj, varargin{:});
    end
    function r = arrange (obj, varargin)
      % arrange  dplyr::arrange()-style multi-key sort. `ascending` defaults to all-true; when provided it must have the same length as `columns`.
      %
      %   r = obj.arrange (columns)
      %   r = obj.arrange (columns, ascending)
      %
      %   Inputs:
      %     columns  cell array of char
      %     ascending  integer vector, default the C++ default
      %
      %   Output:
      %     r  datamunge.DataFrame or []
      %
      %   C++: DataFrame * datamunge::DataFrame::arrange(const std::vector<std::string> & columns, const std::vector<int> & ascending = {}) const
      r = __datamunge__(2059, obj, varargin{:});
    end
    function r = arrange_encoded (obj, varargin)
      % arrange_encoded  C++ datamunge::DataFrame::arrange_encoded.
      %
      %   r = obj.arrange_encoded (encoded_columns)
      %   r = obj.arrange_encoded (encoded_columns, ascending)
      %
      %   Inputs:
      %     encoded_columns  char row
      %     ascending  integer vector, default the C++ default
      %
      %   Output:
      %     r  datamunge.DataFrame or []
      %
      %   C++: DataFrame * datamunge::DataFrame::arrange_encoded(const std::string & encoded_columns, const std::vector<int> & ascending = {}) const
      r = __datamunge__(2060, obj, varargin{:});
    end
    function r = drop_duplicates (obj, varargin)
      % drop_duplicates  C++ datamunge::DataFrame::drop_duplicates.
      %
      %   r = obj.drop_duplicates ()
      %   r = obj.drop_duplicates (subset)
      %
      %   Inputs:
      %     subset  cell array of char, default the C++ default
      %
      %   Output:
      %     r  datamunge.DataFrame or []
      %
      %   C++: DataFrame * datamunge::DataFrame::drop_duplicates(const std::vector<std::string> & subset = {}) const
      r = __datamunge__(2061, obj, varargin{:});
    end
    function r = drop_duplicates_encoded (obj, encoded_subset)
      % drop_duplicates_encoded  C++ datamunge::DataFrame::drop_duplicates_encoded.
      %
      %   r = obj.drop_duplicates_encoded (encoded_subset)
      %
      %   Inputs:
      %     encoded_subset  char row
      %
      %   Output:
      %     r  datamunge.DataFrame or []
      %
      %   C++: DataFrame * datamunge::DataFrame::drop_duplicates_encoded(const std::string & encoded_subset) const
      r = __datamunge__(2062, obj, encoded_subset);
    end
    function r = distinct (obj, varargin)
      % distinct  dplyr::distinct() alias for drop_duplicates().
      %
      %   r = obj.distinct ()
      %   r = obj.distinct (subset)
      %
      %   Inputs:
      %     subset  cell array of char, default the C++ default
      %
      %   Output:
      %     r  datamunge.DataFrame or []
      %
      %   C++: DataFrame * datamunge::DataFrame::distinct(const std::vector<std::string> & subset = {}) const
      r = __datamunge__(2063, obj, varargin{:});
    end
    function r = distinct_encoded (obj, encoded_subset)
      % distinct_encoded  C++ datamunge::DataFrame::distinct_encoded.
      %
      %   r = obj.distinct_encoded (encoded_subset)
      %
      %   Inputs:
      %     encoded_subset  char row
      %
      %   Output:
      %     r  datamunge.DataFrame or []
      %
      %   C++: DataFrame * datamunge::DataFrame::distinct_encoded(const std::string & encoded_subset) const
      r = __datamunge__(2064, obj, encoded_subset);
    end
    function r = pull_numeric (obj, column_name)
      % pull_numeric  dplyr::pull()-style column extraction. Nulls come back as NaN (numeric) / "" (string) in the value vector; check pull_numeric_valid()/pull_string_valid() (1 = present, 0 = null, same convention as add_numeric_column's valid_mask) if nulls matter.
      %
      %   r = obj.pull_numeric (column_name)
      %
      %   Inputs:
      %     column_name  char row
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::DataFrame::pull_numeric(const std::string & column_name) const
      r = __datamunge__(2065, obj, column_name);
    end
    function r = pull_numeric_valid (obj, column_name)
      % pull_numeric_valid  C++ datamunge::DataFrame::pull_numeric_valid.
      %
      %   r = obj.pull_numeric_valid (column_name)
      %
      %   Inputs:
      %     column_name  char row
      %
      %   Output:
      %     r  integer vector
      %
      %   C++: std::vector<int> datamunge::DataFrame::pull_numeric_valid(const std::string & column_name) const
      r = __datamunge__(2066, obj, column_name);
    end
    function r = pull_string (obj, column_name)
      % pull_string  C++ datamunge::DataFrame::pull_string.
      %
      %   r = obj.pull_string (column_name)
      %
      %   Inputs:
      %     column_name  char row
      %
      %   Output:
      %     r  cell array of char
      %
      %   C++: std::vector<std::string> datamunge::DataFrame::pull_string(const std::string & column_name) const
      r = __datamunge__(2067, obj, column_name);
    end
    function r = pull_string_valid (obj, column_name)
      % pull_string_valid  C++ datamunge::DataFrame::pull_string_valid.
      %
      %   r = obj.pull_string_valid (column_name)
      %
      %   Inputs:
      %     column_name  char row
      %
      %   Output:
      %     r  integer vector
      %
      %   C++: std::vector<int> datamunge::DataFrame::pull_string_valid(const std::string & column_name) const
      r = __datamunge__(2068, obj, column_name);
    end
    function r = n_distinct (obj, column_name)
      % n_distinct  Number of distinct values in column_name; a null counts as one additional distinct value if present.
      %
      %   r = obj.n_distinct (column_name)
      %
      %   Inputs:
      %     column_name  char row
      %
      %   Output:
      %     r  integer
      %
      %   C++: std::size_t datamunge::DataFrame::n_distinct(const std::string & column_name) const
      r = __datamunge__(2069, obj, column_name);
    end
    function r = group_by_sum (obj, key_columns, value_columns)
      % group_by_sum  C++ datamunge::DataFrame::group_by_sum.
      %
      %   r = obj.group_by_sum (key_columns, value_columns)
      %
      %   Inputs:
      %     key_columns  cell array of char
      %     value_columns  cell array of char
      %
      %   Output:
      %     r  datamunge.DataFrame or []
      %
      %   C++: DataFrame * datamunge::DataFrame::group_by_sum(const std::vector<std::string> & key_columns, const std::vector<std::string> & value_columns) const
      r = __datamunge__(2070, obj, key_columns, value_columns);
    end
    function r = group_by_sum_encoded (obj, encoded_key_columns, encoded_value_columns)
      % group_by_sum_encoded  C++ datamunge::DataFrame::group_by_sum_encoded.
      %
      %   r = obj.group_by_sum_encoded (encoded_key_columns, encoded_value_columns)
      %
      %   Inputs:
      %     encoded_key_columns  char row
      %     encoded_value_columns  char row
      %
      %   Output:
      %     r  datamunge.DataFrame or []
      %
      %   C++: DataFrame * datamunge::DataFrame::group_by_sum_encoded(const std::string & encoded_key_columns, const std::string & encoded_value_columns) const
      r = __datamunge__(2071, obj, encoded_key_columns, encoded_value_columns);
    end
    function r = count (obj, varargin)
      % count  dplyr::count()-style grouped row counts, default result column name "n".
      %
      %   r = obj.count (key_columns)
      %   r = obj.count (key_columns, count_column_name)
      %
      %   Inputs:
      %     key_columns  cell array of char
      %     count_column_name  char row, default 'n'
      %
      %   Output:
      %     r  datamunge.DataFrame or []
      %
      %   C++: DataFrame * datamunge::DataFrame::count(const std::vector<std::string> & key_columns, const std::string & count_column_name = "n") const
      r = __datamunge__(2072, obj, varargin{:});
    end
    function r = count_encoded (obj, varargin)
      % count_encoded  C++ datamunge::DataFrame::count_encoded.
      %
      %   r = obj.count_encoded (encoded_key_columns)
      %   r = obj.count_encoded (encoded_key_columns, count_column_name)
      %
      %   Inputs:
      %     encoded_key_columns  char row
      %     count_column_name  char row, default 'n'
      %
      %   Output:
      %     r  datamunge.DataFrame or []
      %
      %   C++: DataFrame * datamunge::DataFrame::count_encoded(const std::string & encoded_key_columns, const std::string & count_column_name = "n") const
      r = __datamunge__(2073, obj, varargin{:});
    end
    function r = summarise (obj, key_columns, agg_columns, agg_funcs, result_names)
      % summarise  General dplyr::summarise()-style aggregation: one output row per distinct combination of `key_columns`, with one output column per (agg_columns[i], agg_funcs[i], result_names[i]) triple -- all three arrays must have the same length. agg_funcs entries are one of "sum", "mean", "min", "max", "median", "stddev", "count", "n_distinct" ("count" ignores the corresponding agg_columns entry, which may be ""); a "" result_names entry defaults to the agg_columns entry (or "n" for count).
      %
      %   r = obj.summarise (key_columns, agg_columns, agg_funcs, result_names)
      %
      %   Inputs:
      %     key_columns  cell array of char
      %     agg_columns  cell array of char
      %     agg_funcs  cell array of char
      %     result_names  cell array of char
      %
      %   Output:
      %     r  datamunge.DataFrame or []
      %
      %   C++: DataFrame * datamunge::DataFrame::summarise(const std::vector<std::string> & key_columns, const std::vector<std::string> & agg_columns, const std::vector<std::string> & agg_funcs, const std::vector<std::string> & result_names) const
      r = __datamunge__(2074, obj, key_columns, agg_columns, agg_funcs, result_names);
    end
    function r = summarise_encoded (obj, encoded_key_columns, encoded_agg_columns, encoded_agg_funcs, encoded_result_names)
      % summarise_encoded  C++ datamunge::DataFrame::summarise_encoded.
      %
      %   r = obj.summarise_encoded (encoded_key_columns, encoded_agg_columns, encoded_agg_funcs, encoded_result_names)
      %
      %   Inputs:
      %     encoded_key_columns  char row
      %     encoded_agg_columns  char row
      %     encoded_agg_funcs  char row
      %     encoded_result_names  char row
      %
      %   Output:
      %     r  datamunge.DataFrame or []
      %
      %   C++: DataFrame * datamunge::DataFrame::summarise_encoded(const std::string & encoded_key_columns, const std::string & encoded_agg_columns, const std::string & encoded_agg_funcs, const std::string & encoded_result_names) const
      r = __datamunge__(2075, obj, encoded_key_columns, encoded_agg_columns, encoded_agg_funcs, encoded_result_names);
    end
    function r = pivot_longer (obj, varargin)
      % pivot_longer  dplyr::pivot_longer()-style reshape: stacks `value_columns` into two new columns (`names_to` holding the source column name, `values_to` holding its value).
      %
      %   r = obj.pivot_longer (value_columns)
      %   r = obj.pivot_longer (value_columns, names_to)
      %   r = obj.pivot_longer (value_columns, names_to, values_to)
      %
      %   Inputs:
      %     value_columns  cell array of char
      %     names_to  char row, default 'name'
      %     values_to  char row, default 'value'
      %
      %   Output:
      %     r  datamunge.DataFrame or []
      %
      %   C++: DataFrame * datamunge::DataFrame::pivot_longer(const std::vector<std::string> & value_columns, const std::string & names_to = "name", const std::string & values_to = "value") const
      r = __datamunge__(2076, obj, varargin{:});
    end
    function r = pivot_longer_encoded (obj, varargin)
      % pivot_longer_encoded  C++ datamunge::DataFrame::pivot_longer_encoded.
      %
      %   r = obj.pivot_longer_encoded (encoded_value_columns)
      %   r = obj.pivot_longer_encoded (encoded_value_columns, names_to)
      %   r = obj.pivot_longer_encoded (encoded_value_columns, names_to, values_to)
      %
      %   Inputs:
      %     encoded_value_columns  char row
      %     names_to  char row, default 'name'
      %     values_to  char row, default 'value'
      %
      %   Output:
      %     r  datamunge.DataFrame or []
      %
      %   C++: DataFrame * datamunge::DataFrame::pivot_longer_encoded(const std::string & encoded_value_columns, const std::string & names_to = "name", const std::string & values_to = "value") const
      r = __datamunge__(2077, obj, varargin{:});
    end
    function r = pivot_wider (obj, varargin)
      % pivot_wider  dplyr::pivot_wider()-style reshape: `names_from` (a string column) supplies new column names, `values_from` supplies their values; `id_columns` defaults to every other column.
      %
      %   r = obj.pivot_wider (names_from, values_from)
      %   r = obj.pivot_wider (names_from, values_from, id_columns)
      %
      %   Inputs:
      %     names_from  char row
      %     values_from  char row
      %     id_columns  cell array of char, default the C++ default
      %
      %   Output:
      %     r  datamunge.DataFrame or []
      %
      %   C++: DataFrame * datamunge::DataFrame::pivot_wider(const std::string & names_from, const std::string & values_from, const std::vector<std::string> & id_columns = {}) const
      r = __datamunge__(2078, obj, varargin{:});
    end
    function r = pivot_wider_encoded (obj, names_from, values_from, encoded_id_columns)
      % pivot_wider_encoded  C++ datamunge::DataFrame::pivot_wider_encoded.
      %
      %   r = obj.pivot_wider_encoded (names_from, values_from, encoded_id_columns)
      %
      %   Inputs:
      %     names_from  char row
      %     values_from  char row
      %     encoded_id_columns  char row
      %
      %   Output:
      %     r  datamunge.DataFrame or []
      %
      %   C++: DataFrame * datamunge::DataFrame::pivot_wider_encoded(const std::string & names_from, const std::string & values_from, const std::string & encoded_id_columns) const
      r = __datamunge__(2079, obj, names_from, values_from, encoded_id_columns);
    end
    function r = bind_rows (obj, other)
      % bind_rows  dplyr::bind_rows()-style row union: aligns columns by name (unlike concat_rows, which isn't exposed here), null-filling any column present in only one frame.
      %
      %   r = obj.bind_rows (other)
      %
      %   Inputs:
      %     other  datamunge.DataFrame
      %
      %   Output:
      %     r  datamunge.DataFrame or []
      %
      %   C++: DataFrame * datamunge::DataFrame::bind_rows(const DataFrame & other) const
      r = __datamunge__(2080, obj, other);
    end
    function r = bind_cols (obj, other)
      % bind_cols  dplyr::bind_cols()-style column union: both frames must have the same row count and disjoint column names.
      %
      %   r = obj.bind_cols (other)
      %
      %   Inputs:
      %     other  datamunge.DataFrame
      %
      %   Output:
      %     r  datamunge.DataFrame or []
      %
      %   C++: DataFrame * datamunge::DataFrame::bind_cols(const DataFrame & other) const
      r = __datamunge__(2081, obj, other);
    end
    function r = join (obj, varargin)
      % join  C++ datamunge::DataFrame::join.
      %
      %   r = obj.join (right, left_key, right_key)
      %   r = obj.join (right, left_key, right_key, join_type)
      %   r = obj.join (right, left_key, right_key, join_type, left_suffix)
      %   r = obj.join (right, left_key, right_key, join_type, left_suffix, right_suffix)
      %
      %   Inputs:
      %     right  datamunge.DataFrame
      %     left_key  char row
      %     right_key  char row
      %     join_type  char row, default 'inner': One of "inner" (default), "left", "right", "full", "semi", "anti". Any column name present in both frames (other than the key column when left_key == right_key) is suffixed on both sides so the result never has duplicate names.
      %     left_suffix  char row, default '_x'
      %     right_suffix  char row, default '_y'
      %
      %   Output:
      %     r  datamunge.DataFrame or []
      %
      %   C++: DataFrame * datamunge::DataFrame::join(const DataFrame & right, const std::string & left_key, const std::string & right_key, const std::string & join_type = "inner", const std::string & left_suffix = "_x", const std::string & right_suffix = "_y") const
      r = __datamunge__(2082, obj, varargin{:});
    end
    function r = numeric_count (obj, column_name)
      % numeric_count  C++ datamunge::DataFrame::numeric_count.
      %
      %   r = obj.numeric_count (column_name)
      %
      %   Inputs:
      %     column_name  char row
      %
      %   Output:
      %     r  integer
      %
      %   C++: std::size_t datamunge::DataFrame::numeric_count(const std::string & column_name) const
      r = __datamunge__(2083, obj, column_name);
    end
    function r = numeric_null_count (obj, column_name)
      % numeric_null_count  C++ datamunge::DataFrame::numeric_null_count.
      %
      %   r = obj.numeric_null_count (column_name)
      %
      %   Inputs:
      %     column_name  char row
      %
      %   Output:
      %     r  integer
      %
      %   C++: std::size_t datamunge::DataFrame::numeric_null_count(const std::string & column_name) const
      r = __datamunge__(2084, obj, column_name);
    end
    function r = numeric_sum (obj, column_name)
      % numeric_sum  C++ datamunge::DataFrame::numeric_sum.
      %
      %   r = obj.numeric_sum (column_name)
      %
      %   Inputs:
      %     column_name  char row
      %
      %   Output:
      %     r  double
      %
      %   C++: double datamunge::DataFrame::numeric_sum(const std::string & column_name) const
      r = __datamunge__(2085, obj, column_name);
    end
    function r = numeric_mean (obj, column_name)
      % numeric_mean  C++ datamunge::DataFrame::numeric_mean.
      %
      %   r = obj.numeric_mean (column_name)
      %
      %   Inputs:
      %     column_name  char row
      %
      %   Output:
      %     r  double
      %
      %   C++: double datamunge::DataFrame::numeric_mean(const std::string & column_name) const
      r = __datamunge__(2086, obj, column_name);
    end
    function r = numeric_min (obj, column_name)
      % numeric_min  C++ datamunge::DataFrame::numeric_min.
      %
      %   r = obj.numeric_min (column_name)
      %
      %   Inputs:
      %     column_name  char row
      %
      %   Output:
      %     r  double
      %
      %   C++: double datamunge::DataFrame::numeric_min(const std::string & column_name) const
      r = __datamunge__(2087, obj, column_name);
    end
    function r = numeric_max (obj, column_name)
      % numeric_max  C++ datamunge::DataFrame::numeric_max.
      %
      %   r = obj.numeric_max (column_name)
      %
      %   Inputs:
      %     column_name  char row
      %
      %   Output:
      %     r  double
      %
      %   C++: double datamunge::DataFrame::numeric_max(const std::string & column_name) const
      r = __datamunge__(2088, obj, column_name);
    end
    function r = to_string (obj, varargin)
      % to_string  C++ datamunge::DataFrame::to_string.
      %
      %   r = obj.to_string ()
      %   r = obj.to_string (max_rows)
      %
      %   Inputs:
      %     max_rows  integer, default 10
      %
      %   Output:
      %     r  char row
      %
      %   C++: std::string datamunge::DataFrame::to_string(std::size_t max_rows = 10) const
      r = __datamunge__(2089, obj, varargin{:});
    end
    function r = to_md (obj, varargin)
      % to_md  The frame as a Markdown table (numeric columns right-aligned, at most `digits` decimals each), like knitr::kable(); see dstruct::DataFrame::to_md().
      %
      %   r = obj.to_md ()
      %   r = obj.to_md (digits)
      %
      %   Inputs:
      %     digits  integer, default 3
      %
      %   Output:
      %     r  char row
      %
      %   C++: std::string datamunge::DataFrame::to_md(int digits = 3) const
      r = __datamunge__(2090, obj, varargin{:});
    end
    function r = is_numeric_column (obj, column_name)
      % is_numeric_column  true if column_name holds numeric values, false if it holds strings.
      %
      %   r = obj.is_numeric_column (column_name)
      %
      %   Inputs:
      %     column_name  char row
      %
      %   Output:
      %     r  logical
      %
      %   C++: bool datamunge::DataFrame::is_numeric_column(const std::string & column_name) const
      r = __datamunge__(2091, obj, column_name);
    end
    function r = is_null (obj, column_name, row_index)
      % is_null  C++ datamunge::DataFrame::is_null.
      %
      %   r = obj.is_null (column_name, row_index)
      %
      %   Inputs:
      %     column_name  char row
      %     row_index  integer
      %
      %   Output:
      %     r  logical
      %
      %   C++: bool datamunge::DataFrame::is_null(const std::string & column_name, std::size_t row_index) const
      r = __datamunge__(2092, obj, column_name, row_index);
    end
    function r = numeric_at (obj, column_name, row_index)
      % numeric_at  The value at (column_name, row_index); throws if the cell is null or the column isn't numeric.
      %
      %   r = obj.numeric_at (column_name, row_index)
      %
      %   Inputs:
      %     column_name  char row
      %     row_index  integer
      %
      %   Output:
      %     r  double
      %
      %   C++: double datamunge::DataFrame::numeric_at(const std::string & column_name, std::size_t row_index) const
      r = __datamunge__(2093, obj, column_name, row_index);
    end
    function r = string_at (obj, column_name, row_index)
      % string_at  The value at (column_name, row_index); throws if the cell is null or the column isn't string-typed.
      %
      %   r = obj.string_at (column_name, row_index)
      %
      %   Inputs:
      %     column_name  char row
      %     row_index  integer
      %
      %   Output:
      %     r  char row
      %
      %   C++: std::string datamunge::DataFrame::string_at(const std::string & column_name, std::size_t row_index) const
      r = __datamunge__(2094, obj, column_name, row_index);
    end
  end
  methods (Static)
    function r = empty ()
      % empty  Equivalent to the default constructor, exposed as a static factory (like read_csv()) as a workaround for a SWIG R-backend bug where the plain no-argument constructor's ownership-flagged pointer breaks method dispatch on the result.
      %
      %   r = datamunge.DataFrame.empty ()
      %
      %   Output:
      %     r  datamunge.DataFrame or []
      %
      %   C++: DataFrame * datamunge::DataFrame::empty()
      r = __datamunge__(2040);
    end
    function r = read_csv (varargin)
      % read_csv  Reads a whole CSV file, typing columns automatically; see datasets::read_csv().
      %
      %   r = datamunge.DataFrame.read_csv (path)
      %   r = datamunge.DataFrame.read_csv (path, delimiter)
      %
      %   Inputs:
      %     path  char row: The file to read.
      %     delimiter  char, default ',': The field separator.
      %
      %   Output:
      %     r  datamunge.DataFrame or []
      %
      %   Errors:
      %     datamunge:native  if the file cannot be read or is malformed. (C++ std::runtime_error)
      %
      %   C++: DataFrame * datamunge::DataFrame::read_csv(const std::string & path, char delimiter = ',')
      r = __datamunge__(2095, varargin{:});
    end
    function r = parse_csv (varargin)
      % parse_csv  Parses CSV text, typing columns automatically; see datasets::parse_csv().
      %
      %   r = datamunge.DataFrame.parse_csv (text)
      %   r = datamunge.DataFrame.parse_csv (text, delimiter)
      %
      %   Inputs:
      %     text  char row
      %     delimiter  char, default ','
      %
      %   Output:
      %     r  datamunge.DataFrame or []
      %
      %   Errors:
      %     datamunge:native  if the text is malformed. (C++ std::runtime_error)
      %
      %   C++: DataFrame * datamunge::DataFrame::parse_csv(const std::string & text, char delimiter = ',')
      r = __datamunge__(2096, varargin{:});
    end
    function r = from_image_batch (batch)
      % from_image_batch  An ImageLoader batch as a table (`pixel_0`.., `label`, `index`); see datasets::to_dataframe().
      %
      %   r = datamunge.DataFrame.from_image_batch (batch)
      %
      %   Inputs:
      %     batch  datamunge.ImageBatch
      %
      %   Output:
      %     r  datamunge.DataFrame or []
      %
      %   C++: DataFrame * datamunge::DataFrame::from_image_batch(const datamunge::datasets::ImageBatch & batch)
      r = __datamunge__(2097, batch);
    end
  end
end
% Generated by Rosetta from datamungeoctave/rosetta/bindings.hpp: class datamunge::DataFrame. Do not edit.
