// Tour of the example datasets: built-in tables, cached downloads, and streaming loaders.
// The built-in part runs offline; the download/stream parts fetch data once into
// dm::datasets::data_home() (set DATAMUNGE_DATA_OFFLINE=1 to skip them).
#include <dm/dm.hpp>

#include <iostream>
#include <stdexcept>
#include <string>

int main() {
  namespace ds = dm::datasets;

  // 1. What is available, and how each dataset is stored.
  std::cout << ds::catalog().select({"name", "rows", "columns", "storage"}).to_string() << "\n\n";

  // 2. Built-in tables are ordinary DataFrames.
  const auto wine = ds::wine();
  std::cout << "wine: mean alcohol and proline by cultivar\n"
            << (wine | dm::group_by("cultivar") |
                dm::summarise(dm::agg::n(), dm::agg::mean("alcohol"), dm::agg::mean("proline")))
                   .to_string()
            << "\n\n";

  const auto titanic = ds::titanic();
  std::cout << "titanic: " << titanic.nrows() << " passengers, " << titanic.null_count("Age")
            << " missing ages; survival rate by sex\n"
            << (titanic | dm::group_by("Sex") | dm::summarise(dm::agg::mean("Survived", "survival_rate"))).to_string()
            << "\n\n";

  const auto schools = ds::eight_schools();
  std::cout << "eight schools\n" << schools.to_string() << "\n\n";

  // 3. Any CSV of your own: read it whole, or stream it in fixed-size chunks.
  const auto small = ds::parse_csv("city,temp\nOslo,4.5\nLima,NA\n");
  std::cout << "parsed CSV with " << small.null_count("temp") << " missing value\n\n";

  // 4. Downloaded tables: fetched once, cached, then loaded like the built-in ones.
  try {
    const auto diamonds = ds::diamonds();
    std::cout << "diamonds: mean price by cut\n"
              << (diamonds | dm::group_by("cut") | dm::summarise(dm::agg::mean("price"))).to_string() << "\n\n";

    // The same file streamed 10000 rows at a time, never fully in memory.
    ds::CsvReader reader(ds::fetch("diamonds"), 10000);
    double carats = 0;
    while (reader.has_next())
      for (double c : reader.next().get_column("carat").values()) carats += c;
    std::cout << "streamed " << reader.rows_read() << " diamonds, " << carats << " carats in total\n\n";
  } catch (const std::runtime_error& e) {
    std::cout << "skipping diamonds: " << e.what() << "\n\n";
  }

  // 5. Image datasets stream batch by batch; configure how the batches come back.
  try {
    ds::ImageLoader train("mnist", "train");
    train.set_batch_size(256).set_shuffle(true).set_seed(1).set_layout("flat").set_limit(2048);
    std::cout << "mnist: " << train.split_size() << " training images, using " << train.size() << " in "
              << train.num_batches() << " batches of " << train.batch_size() << "\n";
    for (int epoch = 0; epoch < 2; ++epoch) {
      int counts[10] = {};
      while (train.has_next())
        for (int label : train.next().labels) ++counts[label];
      std::cout << "  epoch " << train.epoch() << " label counts:";
      for (int c : counts) std::cout << ' ' << c;
      std::cout << "\n";
      train.reset();  // next epoch, reshuffled
    }
    ds::ImageLoader test("mnist", "test");
    const auto first = test.take({0});
    std::cout << "  first test image is a " << test.class_names()[first.labels[0]] << "\n";
  } catch (const std::runtime_error& e) {
    std::cout << "skipping mnist: " << e.what() << "\n";
  }
  return 0;
}
