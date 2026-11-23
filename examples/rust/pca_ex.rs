// CMD: just TARGET=pca_ex run-rust
// PCA on iris: summary, PC1 loadings, scores frame, score/scree plots, scaled vs unscaled.
use datamunge::*;

const FEATURE_COLUMNS: [&str; 4] = ["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"];

fn main() -> datamunge::Result<()> {
    let iris = data::iris()?.expect("iris");

    // species labels, for the grouped (colored-by-species) scores plot
    let species = (0..iris.nrows()?).map(|i| iris.string_at("Species", i)).collect::<Result<Vec<_>>>()?;

    println!("=================== PCA on iris (scaled) ===================");
    let pca = PCA::new_with_scale(&iris, &FEATURE_COLUMNS, true, true)?;
    pca.print_summary()?;

    println!("\nPC1 loadings (which original features drive it):");
    for (name, loading) in pca.feature_names()?.iter().zip(pca.component_loadings(0)?) {
        println!("  {name}: {loading:.4}");
    }

    println!("\nTraining scores as a DataFrame:\n{}", pca.scores_frame()?.expect("scores"));

    pca.plot_scores_grouped(&species)?.save_svg("pca_iris_scores.svg")?;
    println!("\nScores scatter (colored by species) saved as pca_iris_scores.svg");

    pca.plot_scree()?.save_svg("pca_iris_scree.svg")?;
    println!("Scree plot saved as pca_iris_scree.svg");

    println!("\n=================== PCA on iris (unscaled) ===================");
    let unscaled = PCA::new_with_scale(&iris, &FEATURE_COLUMNS, true, false)?;
    println!(
        "PC1 explains {:.2}% of variance (vs. {:.2}% scaled) -- Sepal.Length's larger raw variance dominates the unscaled covariance matrix.",
        100.0 * unscaled.explained_variance_ratio()?[0],
        100.0 * pca.explained_variance_ratio()?[0]
    );
    Ok(())
}
