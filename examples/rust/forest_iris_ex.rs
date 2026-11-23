// CMD: just TARGET=forest_iris_ex run-rust
// Random forest classifier on iris: summary, confusion matrix, misclassifications, OOB accuracy.
use datamunge::*;
use datamunge_rust_examples::N;

fn main() -> datamunge::Result<()> {
    let iris = data::iris()?.expect("iris");
    println!("iris: {} rows x {} cols\n", iris.nrows()?, iris.ncols()?);

    let model = RandomForestClassifier::new(&iris, "Species ~ Petal.Length + Petal.Width")?;
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

    model.plot_classification(&iris, "Petal.Length", "Petal.Width")?.save("forest_iris_classification.svg")?;
    model.plot_decision_regions("Petal.Length", "Petal.Width")?.save("forest_iris_decision_regions.svg")?;
    println!("\nSaved forest_iris_classification.svg and forest_iris_decision_regions.svg");

    // A small forest, for comparison, showing how out-of-bag accuracy stabilizes as more trees
    // are added -- the defining random forest effect a single decision tree cannot demonstrate.
    // RandomForestClassifier::new_with_<last param>(data, formula, n_trees, max_depth, min_samples_split,
    //     min_samples_leaf, max_features, criterion, bootstrap, sample_fraction, seed)
    let small_forest = RandomForestClassifier::new_with_n_trees(&iris, "Species ~ Petal.Length + Petal.Width", 5)?;
    println!(
        "\n5-tree forest:   training accuracy={}%  OOB accuracy={}%",
        N(small_forest.training_accuracy()? * 100.0),
        N(small_forest.oob_accuracy()? * 100.0)
    );
    println!(
        "100-tree forest: training accuracy={}%  OOB accuracy={}%",
        N(model.training_accuracy()? * 100.0),
        N(model.oob_accuracy()? * 100.0)
    );
    small_forest.plot_decision_regions("Petal.Length", "Petal.Width")?.save("forest_iris_decision_regions_5trees.svg")?;
    println!("Saved forest_iris_decision_regions_5trees.svg");
    Ok(())
}
