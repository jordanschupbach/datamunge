/// @file csv.cpp
/// @brief RFC 4180 record splitting, column type inference, and the streaming CsvReader.

#include <datamunge/datasets/csv.hpp>

#include <algorithm>
#include <charconv>
#include <fstream>
#include <limits>
#include <istream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

namespace datamunge::datasets {

/// @brief Record parsing and typing helpers shared by parse_csv(), read_csv() and CsvReader.
namespace {

/// @brief One parsed CSV field: its text and whether it was wrapped in quotes.
struct Field {
  std::string text;     ///< The unquoted cell text.
  bool quoted = false;  ///< true if the field was quoted in the source.
};

/// @brief Splits records out of a stream, tracking the line number for error messages.
class RecordParser {
 public:
  /// @brief Parses @p input with @p delimiter; the stream must outlive the parser.
  RecordParser(std::istream& input, char delimiter) : input_(&input), delimiter_(delimiter) {}

  /// @brief Reads the next record into @p fields. Returns false at end of input. Blank lines
  ///        are skipped. Throws std::runtime_error for an unterminated quoted field.
  bool next(std::vector<Field>& fields) {
    fields.clear();
    std::streambuf* buffer = input_->rdbuf();
    using traits = std::char_traits<char>;
    for (;;) {
      int c = buffer->sgetc();
      if (c == traits::eof()) return false;
      if (c == '\n') { buffer->sbumpc(); ++line_; continue; }
      if (c == '\r') { buffer->sbumpc(); continue; }
      break;
    }
    record_line_ = line_ + 1;
    Field field;
    bool in_quotes = false;
    bool after_quote = false;
    for (;;) {
      const int c = buffer->sbumpc();
      if (c == traits::eof()) {
        if (in_quotes) throw error("unterminated quoted field");
        fields.push_back(std::move(field));
        ++line_;
        return true;
      }
      const char ch = traits::to_char_type(c);
      if (in_quotes) {
        if (ch == '"') {
          if (buffer->sgetc() == '"') {
            buffer->sbumpc();
            field.text += '"';
          } else {
            in_quotes = false;
            after_quote = true;
          }
        } else {
          if (ch == '\n') ++line_;
          field.text += ch;
        }
        continue;
      }
      if (ch == delimiter_) {
        fields.push_back(std::move(field));
        field = Field{};
        after_quote = false;
      } else if (ch == '\n') {
        fields.push_back(std::move(field));
        ++line_;
        return true;
      } else if (ch == '\r') {
        // Dropped: handles \r\n line ends.
      } else if (ch == '"' && field.text.find_first_not_of(' ') == std::string::npos && !after_quote) {
        field.text.clear();
        field.quoted = true;
        in_quotes = true;
      } else if (!(after_quote && ch == ' ')) {
        field.text += ch;
      }
    }
  }

  /// @brief An exception naming the line where the current record started.
  [[nodiscard]] std::runtime_error error(const std::string& what) const {
    return std::runtime_error("CSV line " + std::to_string(record_line_) + ": " + what);
  }

  /// @brief The 1-based line number where the most recent record started.
  [[nodiscard]] long long record_line() const noexcept { return record_line_; }

  /// @brief Rebinds to a (rewound) stream and resets line counting.
  void restart(std::istream& input) {
    input_ = &input;
    line_ = 0;
    record_line_ = 0;
  }

 private:
  std::istream* input_;       ///< The stream being parsed (not owned).
  char delimiter_;            ///< Field separator.
  long long line_ = 0;        ///< Newlines consumed so far.
  long long record_line_ = 0; ///< First line of the most recent record.
};

/// @brief Removes leading and trailing spaces and tabs.
std::string trim(const std::string& text) {
  const auto begin = text.find_first_not_of(" \t");
  if (begin == std::string::npos) return {};
  const auto end = text.find_last_not_of(" \t");
  return text.substr(begin, end - begin + 1);
}

/// @brief Parses a whole cell as a double; std::nullopt if it is not entirely a number.
std::optional<double> to_number(const std::string& text) {
  const std::string value = trim(text);
  if (value.empty()) return std::nullopt;
  const char* begin = value.data();
  const char* end = begin + value.size();
  if (*begin == '+') ++begin;
  double result = 0;
  const auto [ptr, ec] = std::from_chars(begin, end, result);
  if (ec != std::errc() || ptr != end) {
    // from_chars rejects "Inf"/"NaN" spellings that R writes; accept the common ones.
    if (value == "Inf" || value == "inf") return std::numeric_limits<double>::infinity();
    if (value == "-Inf" || value == "-inf") return -std::numeric_limits<double>::infinity();
    if (value == "NaN" || value == "nan") return std::numeric_limits<double>::quiet_NaN();
    return std::nullopt;
  }
  return result;
}

/// @brief Reads the header record (or synthesizes V1.. names from the first record, which is
///        then kept as data in @p pending). Returns false for an empty input.
bool read_header(RecordParser& parser, const CsvOptions& options, std::vector<std::string>& names,
                 std::vector<Field>& pending, bool& has_pending) {
  std::vector<Field> first;
  if (!parser.next(first)) return false;
  names.clear();
  if (options.header) {
    for (const auto& f : first) names.push_back(trim(f.text));
    has_pending = false;
  } else {
    for (std::size_t i = 0; i < first.size(); ++i) names.push_back("V" + std::to_string(i + 1));
    pending = std::move(first);
    has_pending = true;
  }
  std::unordered_set<std::string> seen;
  for (const auto& name : names)
    if (!seen.insert(name).second) throw parser.error("duplicate column name \"" + name + "\"");
  return true;
}

/// @brief true if @p cell is a missing value under @p options.
bool is_na(const Field& cell, const CsvOptions& options) {
  const std::string value = trim(cell.text);
  return std::find(options.na_values.begin(), options.na_values.end(), value) != options.na_values.end();
}

/// @brief Decides, per column, whether every non-missing cell is numeric.
std::vector<bool> infer_types(const std::vector<std::string>& names,
                              const std::vector<std::vector<Field>>& columns, const CsvOptions& options) {
  std::vector<bool> numeric(names.size(), true);
  for (std::size_t c = 0; c < names.size(); ++c) {
    if (std::find(options.string_columns.begin(), options.string_columns.end(), names[c]) !=
        options.string_columns.end()) {
      numeric[c] = false;
      continue;
    }
    bool any_value = false;
    for (const auto& cell : columns[c]) {
      if (is_na(cell, options)) continue;
      any_value = true;
      if (cell.quoted || !to_number(cell.text)) {
        numeric[c] = false;
        break;
      }
    }
    // A column with no values at all reads as string so later text cannot break it.
    if (!any_value) numeric[c] = false;
  }
  return numeric;
}

/// @brief Builds a DataFrame from columnar cells with fixed column types.
/// @param lines Source line of each row, for error messages.
dstruct::DataFrame build_frame(const std::vector<std::string>& names, const std::vector<std::vector<Field>>& columns,
                               const std::vector<bool>& numeric, const CsvOptions& options,
                               const std::vector<long long>& lines) {
  dstruct::DataFrame frame;
  for (std::size_t c = 0; c < names.size(); ++c) {
    const auto& cells = columns[c];
    if (numeric[c]) {
      std::vector<std::optional<double>> values;
      values.reserve(cells.size());
      for (std::size_t r = 0; r < cells.size(); ++r) {
        if (is_na(cells[r], options)) {
          values.emplace_back(std::nullopt);
          continue;
        }
        const auto number = cells[r].quoted ? std::nullopt : to_number(cells[r].text);
        if (!number)
          throw std::runtime_error("CSV line " + std::to_string(lines[r]) + ": column \"" + names[c] +
                                   "\" was typed numeric but holds \"" + cells[r].text +
                                   "\"; list it in CsvOptions::string_columns");
        values.emplace_back(*number);
      }
      frame.add_column(names[c], std::move(values));
    } else {
      std::vector<std::optional<std::string>> values;
      values.reserve(cells.size());
      for (const auto& cell : cells) {
        if (is_na(cell, options))
          values.emplace_back(std::nullopt);
        else
          values.emplace_back(cell.text);
      }
      frame.add_column(names[c], dstruct::DataFrame::string_column_type(std::move(values)));
    }
  }
  return frame;
}

/// @brief Appends one record's fields to the column buffers, checking its width.
void append_record(const RecordParser& parser, std::vector<Field>& record,
                   std::vector<std::vector<Field>>& columns, std::vector<long long>& lines) {
  if (record.size() != columns.size())
    throw parser.error("expected " + std::to_string(columns.size()) + " fields but found " +
                       std::to_string(record.size()));
  for (std::size_t c = 0; c < record.size(); ++c) columns[c].push_back(std::move(record[c]));
  lines.push_back(parser.record_line());
}

/// @brief Parses an entire stream into a DataFrame, typing columns from all rows.
dstruct::DataFrame parse_stream(std::istream& input, const CsvOptions& options) {
  RecordParser parser(input, options.delimiter);
  std::vector<std::string> names;
  std::vector<Field> pending;
  bool has_pending = false;
  if (!read_header(parser, options, names, pending, has_pending)) return {};
  std::vector<std::vector<Field>> columns(names.size());
  std::vector<long long> lines;
  if (has_pending) append_record(parser, pending, columns, lines);
  std::vector<Field> record;
  while (parser.next(record)) append_record(parser, record, columns, lines);
  return build_frame(names, columns, infer_types(names, columns, options), options, lines);
}

} // namespace

dstruct::DataFrame parse_csv(const std::string& text, const CsvOptions& options) {
  std::istringstream input(text);
  return parse_stream(input, options);
}

std::string to_csv(const dstruct::DataFrame& frame, const CsvOptions& options) {
  const char delim = options.delimiter;
  const auto field = [delim](const std::string& text) {
    const bool quote = text.find_first_of(std::string{delim, '"', '\n', '\r'}) != std::string::npos ||
                       (!text.empty() && (text.front() == ' ' || text.back() == ' '));
    if (!quote) return text;
    std::string out = "\"";
    for (const char c : text) {
      if (c == '"') out += '"';
      out += c;
    }
    return out + '"';
  };
  const auto columns = frame.columns();
  std::string out;
  if (options.header) {
    for (std::size_t c = 0; c < columns.size(); ++c) out += (c ? std::string(1, delim) : "") + field(columns[c]);
    out += '\n';
  }
  char buffer[64];
  for (std::size_t r = 0; r < frame.nrows(); ++r) {
    for (std::size_t c = 0; c < columns.size(); ++c) {
      if (c) out += delim;
      const auto cell = frame.value(columns[c], r);
      if (const auto* number = std::get_if<double>(&cell)) {
        const auto result = std::to_chars(buffer, buffer + sizeof buffer, *number);
        out.append(buffer, result.ptr);
      } else if (const auto* text = std::get_if<std::string>(&cell)) {
        out += field(*text);
      } else {
        out += "NA";
      }
    }
    out += '\n';
  }
  return out;
}

dstruct::DataFrame read_csv(const std::string& path, const CsvOptions& options) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("cannot open CSV file " + path);
  return parse_stream(input, options);
}

/// @brief Everything a CsvReader owns: the file, its parser and the fixed column layout.
struct CsvReader::State {
  std::string path;                 ///< The file being streamed.
  CsvOptions options;               ///< Parsing rules.
  int batch_size = 1000;            ///< Rows per chunk.
  std::ifstream input;              ///< The open file.
  RecordParser parser{input, ','};  ///< Record splitter over input.
  std::vector<std::string> names;   ///< Column names.
  std::vector<bool> numeric;        ///< Column types, fixed at construction.
  std::vector<Field> lookahead;     ///< The next record, if already read.
  bool has_lookahead = false;       ///< true when lookahead holds an unreturned record.
  long long rows_read = 0;          ///< Rows returned since the last rewind.

  /// @brief Opens (or reopens) the file and positions it just after the header.
  void open() {
    input.close();
    input.clear();
    input.open(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open CSV file " + path);
    parser = RecordParser(input, options.delimiter);
    std::vector<std::string> header_names;
    has_lookahead = false;
    if (!read_header(parser, options, header_names, lookahead, has_lookahead)) header_names.clear();
    names = std::move(header_names);
    rows_read = 0;
  }

  /// @brief Ensures lookahead holds the next record if one exists; returns whether it does.
  bool peek() {
    if (!has_lookahead) has_lookahead = parser.next(lookahead);
    return has_lookahead;
  }
};

CsvReader::CsvReader(const std::string& path, int batch_size, CsvOptions options)
    : state_(std::make_unique<State>()) {
  if (batch_size <= 0) throw std::invalid_argument("CsvReader batch_size must be positive");
  if (options.infer_rows <= 0) throw std::invalid_argument("CsvOptions::infer_rows must be positive");
  state_->path = path;
  state_->options = std::move(options);
  state_->batch_size = batch_size;
  state_->open();
  // Type the columns from a sample of leading rows, then rewind.
  std::vector<std::vector<Field>> sample(state_->names.size());
  std::vector<long long> lines;
  for (int r = 0; r < state_->options.infer_rows && state_->peek(); ++r) {
    append_record(state_->parser, state_->lookahead, sample, lines);
    state_->has_lookahead = false;
  }
  state_->numeric = infer_types(state_->names, sample, state_->options);
  state_->open();
}

CsvReader::CsvReader(CsvReader&&) noexcept = default;
CsvReader& CsvReader::operator=(CsvReader&&) noexcept = default;
CsvReader::~CsvReader() = default;

bool CsvReader::has_next() { return state_->peek(); }

dstruct::DataFrame CsvReader::next() {
  if (!state_->peek()) throw std::out_of_range("CsvReader::next: no rows remain");
  std::vector<std::vector<Field>> columns(state_->names.size());
  std::vector<long long> lines;
  for (int r = 0; r < state_->batch_size && state_->peek(); ++r) {
    append_record(state_->parser, state_->lookahead, columns, lines);
    state_->has_lookahead = false;
  }
  state_->rows_read += static_cast<long long>(lines.size());
  return build_frame(state_->names, columns, state_->numeric, state_->options, lines);
}

void CsvReader::reset() { state_->open(); }

int CsvReader::batch_size() const noexcept { return state_->batch_size; }

void CsvReader::set_batch_size(int rows) {
  if (rows <= 0) throw std::invalid_argument("CsvReader batch_size must be positive");
  state_->batch_size = rows;
}

const std::vector<std::string>& CsvReader::column_names() const noexcept { return state_->names; }

const std::vector<bool>& CsvReader::numeric_columns() const noexcept { return state_->numeric; }

long long CsvReader::rows_read() const noexcept { return state_->rows_read; }

} // namespace datamunge::datasets
