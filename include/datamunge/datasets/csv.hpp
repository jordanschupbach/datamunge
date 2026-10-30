#pragma once

/// @file csv.hpp
/// @brief CSV text to dstruct::DataFrame: whole files at once, or in fixed-size row chunks.
///
/// Columns are typed automatically. A column is numeric when every non-missing cell parses
/// as a number, otherwise it is a string column. Cells matching CsvOptions::na_values (by
/// default empty and "NA") become nulls. Quoting follows RFC 4180: fields may be wrapped in
/// double quotes, may then contain delimiters and newlines, and `""` is a literal quote.
///
/// @code
/// const auto frame = dm::datasets::read_csv("measurements.csv");
///
/// dm::datasets::CsvReader reader("huge.csv", 10000);   // never holds more than 10000 rows
/// while (reader.has_next()) {
///   const auto chunk = reader.next();
///   // ... process chunk ...
/// }
/// @endcode

#include <datamunge/dstruct/dataframe.hpp>

#include <memory>
#include <string>
#include <vector>

namespace datamunge::datasets {

/// @brief How CSV text is split into cells and typed. Every field has a usable default.
struct CsvOptions {
  /// @brief The field separator, `,` by default (use `'\t'` for TSV, `';'` for European CSV).
  char delimiter = ',';

  /// @brief true when the first record holds the column names. When false, columns are
  ///        named `V1`, `V2`, ... like R's read.csv(header = FALSE).
  bool header = true;

  /// @brief Cell texts read as missing values (nulls), compared after trimming spaces.
  std::vector<std::string> na_values = {"", "NA"};

  /// @brief Columns always read as strings even if every cell looks numeric (IDs, zip codes).
  std::vector<std::string> string_columns;

  /// @brief How many data rows CsvReader inspects up front to decide each column's type.
  ///        Ignored by parse_csv()/read_csv(), which inspect every row. Must be positive.
  int infer_rows = 1000;
};

/// @brief Parses CSV text into a new DataFrame, typing each column from all of its cells.
///
/// @param text The whole CSV document; `\n` and `\r\n` line ends are both accepted.
/// @param options Delimiter, header, missing-value and typing rules.
/// @return A DataFrame with one column per field, in file order.
/// @throws std::runtime_error if a record has a different number of fields than the header,
///         a quoted field is not closed, or two columns share a name.
///
/// @par Example
/// @code
/// const auto frame = dm::datasets::parse_csv("x,label\n1.5,a\nNA,b\n");
/// // 2 x 2: x numeric (1.5, null), label string ("a", "b")
/// @endcode
/// @see read_csv(), CsvReader
dstruct::DataFrame parse_csv(const std::string& text, const CsvOptions& options = {});

/// @brief Reads a whole CSV file into a new DataFrame. Same rules as parse_csv().
///
/// The whole file is held in memory; use CsvReader for files that do not fit.
///
/// @param path The file to read.
/// @param options Delimiter, header, missing-value and typing rules.
/// @return A DataFrame with one column per field.
/// @throws std::runtime_error if the file cannot be opened or is malformed (see parse_csv()).
///
/// @par Example
/// @code
/// const auto diamonds = dm::datasets::read_csv(dm::datasets::fetch("diamonds"));
/// @endcode
/// @see parse_csv(), CsvReader
dstruct::DataFrame read_csv(const std::string& path, const CsvOptions& options = {});

/// @brief Writes a DataFrame as CSV text, the inverse of parse_csv().
///
/// Numbers use the shortest text that reads back to the same double (`3.6`, `79`), so
/// `parse_csv(to_csv(frame))` reproduces numeric columns exactly. Missing values are written as
/// `NA`. A string cell is quoted when it contains the delimiter, a quote, a line break or
/// leading or trailing spaces, with inner quotes doubled as in RFC 4180. Lines end in `\n`.
///
/// @param frame The table to write, in its column order.
/// @param options Only `delimiter` and `header` are used.
/// @return The whole CSV document.
///
/// @par Example
/// @code
/// const std::string text = dm::datasets::to_csv(dm::datasets::faithful());
/// // "eruptions,waiting\n3.6,79\n1.8,54\n..."
/// @endcode
/// @see parse_csv()
std::string to_csv(const dstruct::DataFrame& frame, const CsvOptions& options = {});

/// @brief Streams a CSV file as DataFrames of at most batch_size() rows each.
///
/// Only the current chunk is ever in memory, so files larger than RAM can be processed. Column
/// types are decided once, at construction, from the first CsvOptions::infer_rows data rows,
/// so every chunk has the same columns and types. A later cell that does not fit its numeric
/// column throws; list such columns in CsvOptions::string_columns.
///
/// Not thread-safe: one reader holds one open file position.
///
/// @par Example
/// @code
/// dm::datasets::CsvReader reader(dm::datasets::fetch("diamonds"), 5000);
/// double total = 0;
/// while (reader.has_next()) {
///   const auto chunk = reader.next();               // <= 5000 rows
///   for (double price : chunk.get_column("price").values()) total += price;
/// }
/// reader.reset();                                    // start again from the first row
/// @endcode
class CsvReader {
 public:
  /// @brief Opens @p path, reads its header and infers column types.
  /// @param path The CSV file to stream.
  /// @param batch_size Rows per chunk returned by next(); must be positive.
  /// @param options Delimiter, header, missing-value and typing rules.
  /// @throws std::invalid_argument if batch_size or CsvOptions::infer_rows is not positive.
  /// @throws std::runtime_error if the file cannot be opened or its start is malformed.
  explicit CsvReader(const std::string& path, int batch_size = 1000, CsvOptions options = {});

  /// @brief Move-only: a reader owns an open file.
  CsvReader(CsvReader&&) noexcept;
  /// @brief Move-assigns, closing this reader's file first.
  CsvReader& operator=(CsvReader&&) noexcept;
  /// @brief Not copyable: a reader owns one open file position.
  CsvReader(const CsvReader&) = delete;
  /// @brief Not copy-assignable; see the deleted copy constructor.
  CsvReader& operator=(const CsvReader&) = delete;
  /// @brief Closes the file.
  ~CsvReader();

  /// @brief true while at least one more data row remains.
  [[nodiscard]] bool has_next();

  /// @brief Reads the next chunk of up to batch_size() rows (fewer only at the end of file).
  /// @throws std::out_of_range when no rows remain (check has_next() first).
  /// @throws std::runtime_error for a malformed record or a cell that does not fit its column.
  [[nodiscard]] dstruct::DataFrame next();

  /// @brief Rewinds to the first data row; the next next() call returns the first chunk again.
  void reset();

  /// @brief Rows per chunk.
  [[nodiscard]] int batch_size() const noexcept;

  /// @brief Changes the rows per chunk for subsequent next() calls.
  /// @throws std::invalid_argument if @p rows is not positive.
  void set_batch_size(int rows);

  /// @brief The column names, in file order.
  [[nodiscard]] const std::vector<std::string>& column_names() const noexcept;

  /// @brief For each column, true if it is read as numeric, false if as string.
  [[nodiscard]] const std::vector<bool>& numeric_columns() const noexcept;

  /// @brief Data rows returned by next() since construction or the last reset().
  [[nodiscard]] long long rows_read() const noexcept;

 private:
  struct State;
  std::unique_ptr<State> state_;  ///< Open file, parsed header and column types.
};

} // namespace datamunge::datasets
