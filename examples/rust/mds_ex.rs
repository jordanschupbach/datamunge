// CMD: just TARGET=mds_ex run-rust
// Classical multidimensional scaling of iris: euclidean embedding, plot, and a manhattan comparison.
use datamunge::*;

const FEATURE_COLUMNS: [&str; 4] = ["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"];

fn main() -> datamunge::Result<()> {
    let iris = data::iris()?.expect("iris");
    let species = (0..iris.nrows()?).map(|i| iris.string_at("Species", i)).collect::<Result<Vec<_>>>()?;

    println!("=================== Classical MDS on iris (euclidean) ===================");
    let mds = MDS::new_with_metric(&iris, &FEATURE_COLUMNS, 2, "euclidean")?;
    mds.print_summary()?;

    println!("\nEmbedding as a DataFrame:\n{}", mds.embedding_frame()?.expect("embedding"));

    mds.plot_embedding_grouped(&species)?.save_svg("mds_iris_embedding.svg")?;
    println!("\nEmbedding scatter (colored by species) saved as mds_iris_embedding.svg");

    println!("\n=================== Classical MDS on iris (manhattan) ===================");
    let manhattan_mds = MDS::new_with_metric(&iris, &FEATURE_COLUMNS, 2, "manhattan")?;
    println!(
        "Goodness of fit: euclidean={:.4}, manhattan={:.4}",
        mds.goodness_of_fit()?,
        manhattan_mds.goodness_of_fit()?
    );
    Ok(())
}
