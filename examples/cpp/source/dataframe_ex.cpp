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

  // Tibble/dplyr-style piping: every transform below returns a new DataFrame by value, so
  // mutate/arrange/rename/select/... chain directly, same as `df |> mutate(...) |> arrange(...)`.
  const auto piped = cleaned.mutate_with("tax", [](const DataFrame::Row& row) { return row.get_double("sales") * 0.1; })
                          .rename("quarter", "period")
                          .arrange({"region", "sales"}, {true, false})
                          .relocate({"region", "sales"})
                          .select({"region", "sales", "tax", "period"});
  std::cout << "piped (mutate + arrange + rename + relocate + select)\n" << piped.to_string() << "\n\n";

  const DataFrame::AggSpec total_sales{"sales", DataFrame::AggFunc::Sum, "total_sales"};
  const DataFrame::AggSpec avg_sales{"sales", DataFrame::AggFunc::Mean, "avg_sales"};
  const DataFrame::AggSpec n_products{"product", DataFrame::AggFunc::NDistinct, "n_products"};
  const auto summarised = cleaned.group_by({"region"}).summarise({total_sales, avg_sales, n_products});
  std::cout << "group_by(region).summarise(sum, mean, n_distinct)\n" << summarised.to_string() << "\n\n";

  const auto right_join = grouped.join(targets, "region", "region", DataFrame::JoinType::Right);
  std::cout << "right join with targets\n" << right_join.to_string() << "\n\n";

  const auto unmatched = grouped.join(targets, "region", "region", DataFrame::JoinType::Anti);
  std::cout << "anti join (regions with no target)\n" << unmatched.to_string() << "\n\n";

  const auto longer = cleaned.pivot_longer({"sales"}, "metric", "value");
  std::cout << "pivot_longer(sales)\n" << longer.to_string() << "\n\n";

  DataFrame new_region{{{"region", DataFrame::string_column_type(std::vector<std::string>{"north"})},
                        {"product", DataFrame::string_column_type(std::vector<std::string>{"widget"})},
                        {"sales", DataFrame::numeric_column_type(std::vector<double>{6.0})}}};
  const auto bound = cleaned.select({"region", "product", "sales"}).bind_rows(new_region);
  std::cout << "bind_rows with a new region\n" << bound.to_string() << "\n\n";

  return 0;
}
