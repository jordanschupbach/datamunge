// CMD: just TARGET=knn_iris_ex run-rust
// k-nearest-neighbours classifier on iris: leave-one-out accuracy and decision regions for several k.
use datamunge::*;
use datamunge_rust_examples::N;

fn main() -> datamunge::Result<()> {
    let iris = data::iris()?.expect("iris");
    println!("iris: {} rows x {} cols\n", iris.nrows()?, iris.ncols()?);

    let model = KNNClassifier::new(&iris, "Species ~ Petal.Length + Petal.Width")?;
    model.print_summary()?;

    println!("\nConfusion matrix (leave-one-out; rows = actual, cols = predicted):");
    println!("{}", model.confusion_matrix()?.expect("confusion matrix"));

    println!("\nLeave-one-out misclassified rows:");
    let fitted = model.predict(&iris)?;
    let n = iris.nrows()?;
    let mut misclassified = 0;
    for (i, predicted) in fitted.iter().enumerate().take(n) {
        let actual = iris.string_at("Species", i)?;
        if *predicted == actual {
            continue;
        }
        misclassified += 1;
        println!(
            "  row {i}: Petal.Length={} Petal.Width={}  actual={actual}  predicted={predicted}",
            N(iris.numeric_at("Petal.Length", i)?),
            N(iris.numeric_at("Petal.Width", i)?)
        );
    }
    println!("{misclassified} of {n} misclassified ({}%)", N(100.0 * misclassified as f64 / n as f64));

    model.plot_decision_regions("Petal.Length", "Petal.Width")?.save("knn_iris_decision_regions_k5.svg")?;
    println!("\nSaved knn_iris_decision_regions_k5.svg");

    // k=1 memorizes every training point exactly (jagged, overfit boundary with an island around
    // every point, including noise); k=25 averages over a much larger neighborhood (very smooth,
    // underfit boundary). KNNClassifier::new_with_<last param>(data, formula, k, metric, weighted, standardize)
    for k in [1, 25] {
        let knn = KNNClassifier::new_with_k(&iris, "Species ~ Petal.Length + Petal.Width", k)?;
        let file = format!("knn_iris_decision_regions_k{k}.svg");
        println!("\nk={k:<2} leave-one-out accuracy: {}%", N(knn.training_accuracy()? * 100.0));
        knn.plot_decision_regions("Petal.Length", "Petal.Width")?.save(&file)?;
        println!("Saved {file}");
    }
    Ok(())
}
