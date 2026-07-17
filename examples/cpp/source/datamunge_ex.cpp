#include <datamunge/dstruct/dstruct.hpp>

#include <iostream>
#include <optional>
#include <string>
#include <vector>

using datamunge::dstruct::DataFrame;

int main() {
  DataFrame sales{{{"region", DataFrame::string_column_type(std::vector<std::optional<std::string>>{
                             "west", "west", "east", "south", "south", "south"})},
                   {"product", DataFrame::string_column_type(std::vector<std::optional<std::string>>{
                              "widget", "widget", "widget", "gizmo", "gizmo", "gizmo"})},
                   {"sales", DataFrame::numeric_column_type(
                                 std::vector<std::optional<double>>{10.0, 10.0, 14.0, 8.0, std::nullopt, 11.0})},
                   {"quarter", DataFrame::string_column_type(std::vector<std::optional<std::string>>{
                              "Q1", "Q1", "Q1", "Q2", "Q2", std::nullopt})}}};

  std::cout << "raw data\n" << sales.to_string() << "\n\n";

  auto cleaned = sales.drop_duplicates({"region", "product", "sales", "quarter"});
  cleaned.fill_null("quarter", "unknown");
  cleaned.fill_null("sales", 0.0);
  std::cout << "after drop_duplicates + fill_null\n" << cleaned.to_string() << "\n\n";

  const auto selected = cleaned.select({"region", "sales", "quarter"}).sort_by("sales", false);
  std::cout << "selected + sorted\n" << selected.to_string() << "\n\n";

  const auto grouped = cleaned.group_by_sum({"region"}, {"sales"}).sort_by("sales", false);
  std::cout << "group_by_sum(region)\n" << grouped.to_string() << "\n\n";

  DataFrame targets{{{"region", DataFrame::string_column_type(std::vector<std::string>{"west", "east", "south"})},
                     {"target", DataFrame::numeric_column_type(std::vector<double>{18.0, 12.0, 25.0})}}};
  const auto joined = grouped.join(targets, "region", "region", DataFrame::JoinType::Left);
  std::cout << "joined with targets\n" << joined.to_string() << "\n\n";

  const auto summary = cleaned.describe_numeric("sales");
  const auto [rows, cols] = cleaned.shape();
  std::cout << "shape = (" << rows << ", " << cols << ")\n"
            << "sales count = " << summary.count << "\n"
            << "sales nulls = " << summary.null_count << "\n"
            << "sales sum = " << summary.sum << "\n"
            << "sales mean = " << summary.mean << "\n";

#ifdef DATAMUNGE_HAVE_ARROW
  const auto table = cleaned.to_arrow();
  std::cout << "arrow rows = " << table->num_rows() << ", cols = " << table->num_columns() << "\n";
#endif

  return 0;
}
