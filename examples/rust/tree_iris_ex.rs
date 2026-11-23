// CMD: just TARGET=tree_iris_ex run-rust
// Decision tree classifier on iris: summary, confusion matrix, misclassified rows and plots.
use datamunge::*;
use datamunge_rust_examples::N;

fn main() -> datamunge::Result<()> {
    let iris = data::iris()?.expect("iris");
    println!("iris: {} rows x {} cols\n", iris.nrows()?, iris.ncols()?);

    let model = DecisionTreeClassifier::new(&iris, "Species ~ Petal.Length + Petal.Width")?;
    model.print_summary()?;

    println!("\nConfusion matrix (rows = actual, cols = predicted):");
    println!("{}", model.confusion_matrix()?.expect("confusion matrix"));

    println!("\nMisclassified rows:");
    let predictions = model.predict(&iris)?;
    let n = iris.nrows()?;
    let mut misclassified = 0;
    for (i, predicted) in predictions.iter().enumerate().take(n) {
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

    model.plot_classification(&iris, "Petal.Length", "Petal.Width")?.save("tree_iris_classification.svg")?;
    model.plot_decision_regions("Petal.Length", "Petal.Width")?.save("tree_iris_decision_regions.svg")?;
    println!("\nSaved tree_iris_classification.svg and tree_iris_decision_regions.svg");

    // A shallower tree, for comparison, showing a coarser (but still fairly accurate) decision boundary.
    let shallow = DecisionTreeClassifier::new_with_max_depth(&iris, "Species ~ Petal.Length + Petal.Width", 2)?;
    println!(
        "\nDepth-2 tree training accuracy: {}% ({} leaves)",
        N(shallow.training_accuracy()? * 100.0),
        shallow.leaf_count()?
    );
    shallow.plot_decision_regions("Petal.Length", "Petal.Width")?.save("tree_iris_decision_regions_depth2.svg")?;
    println!("Saved tree_iris_decision_regions_depth2.svg");
    Ok(())
}
