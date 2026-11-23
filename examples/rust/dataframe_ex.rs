// CMD: just TARGET=dataframe_ex run-rust
// DataFrame basics: nulls, drop_duplicates, fill_null, select/sort, group-by, join and summaries.
use datamunge::*;
use datamunge_rust_examples::N;

fn main() -> datamunge::Result<()> {
    let sales = DataFrame::new()?;
    sales.add_string_column("region", &["west", "west", "east", "south", "south", "south"])?;
    sales.add_string_column("product", &["widget", "widget", "widget", "gizmo", "gizmo", "gizmo"])?;
    sales.add_numeric_column_with_valid_mask("sales", &[10.0, 10.0, 14.0, 8.0, 0.0, 11.0], &[1, 1, 1, 1, 0, 1])?;
    sales.add_string_column_with_valid_mask("quarter", &["Q1", "Q1", "Q1", "Q2", "Q2", ""], &[1, 1, 1, 1, 1, 0])?;

    println!("raw data\n{sales}\n");

    let cleaned = sales.drop_duplicates_with_subset(&["region", "product", "sales", "quarter"])?.expect("deduplicated");
    cleaned.fill_null_string("quarter", "unknown")?;
    cleaned.fill_null_numeric("sales", 0.0)?;
    println!("after drop_duplicates + fill_null\n{cleaned}\n");

    let selected = cleaned
        .select(&["region", "sales", "quarter"])?
        .expect("selected")
        .sort_by_with_ascending("sales", false)?
        .expect("sorted");
    println!("selected + sorted\n{selected}\n");

    let grouped = cleaned
        .group_by_sum(&["region"], &["sales"])?
        .expect("grouped")
        .sort_by_with_ascending("sales", false)?
        .expect("sorted");
    println!("group_by_sum(region)\n{grouped}\n");

    let targets = DataFrame::new()?;
    targets.add_string_column("region", &["west", "east", "south"])?;
    targets.add_numeric_column("target", &[18.0, 12.0, 25.0])?;
    let joined = grouped.join_with_join_type(&targets, "region", "region", "left")?.expect("joined");
    println!("joined with targets\n{joined}\n");

    let shape = cleaned.shape()?;
    println!("shape = ({}, {})", shape[0], shape[1]);
    println!("sales count = {}", cleaned.numeric_count("sales")?);
    println!("sales nulls = {}", cleaned.numeric_null_count("sales")?);
    println!("sales sum = {}", N(cleaned.numeric_sum("sales")?));
    println!("sales mean = {}", N(cleaned.numeric_mean("sales")?));
    Ok(())
}
