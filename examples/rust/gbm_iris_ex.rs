// CMD: just TARGET=gbm_iris_ex run-rust
// Gradient-boosted trees (GBM) on iris: summary, confusion matrix, misclassified rows, 5 vs 100 rounds.
use datamunge::*;
use datamunge_rust_examples::N;

fn main() -> datamunge::Result<()> {
    let iris = data::iris()?.expect("iris");
    println!("iris: {} rows x {} cols\n", iris.nrows()?, iris.ncols()?);

    let model = GBMClassifier::new(&iris, "Species ~ Petal.Length + Petal.Width")?;
    model.print_summary()?;

    println!(
        "\nConfusion matrix (rows = actual, cols = predicted):\n{}",
        model.confusion_matrix()?.expect("confusion matrix")
    );

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

    model.plot_training_deviance()?.save("gbm_iris_training_deviance.svg")?;
    model.plot_decision_regions("Petal.Length", "Petal.Width")?.save("gbm_iris_decision_regions.svg")?;
    println!("\nSaved gbm_iris_training_deviance.svg and gbm_iris_decision_regions.svg");

    // A handful of boosting rounds vs a well-boosted ensemble: each additional round chips away
    // at the training loss, gradually sharpening the decision boundary.
    // GBMClassifier::new_with_<last param>(data, formula, n_trees, learning_rate, max_depth,
    //     min_samples_split, min_samples_leaf, subsample, seed)
    let few = GBMClassifier::new_with_n_trees(&iris, "Species ~ Petal.Length + Petal.Width", 5)?;
    let last = |deviance: Vec<f64>| N(*deviance.last().expect("deviance"));
    println!(
        "\n5-round ensemble:   training accuracy={}%  deviance={}",
        N(100.0 * few.training_accuracy()?),
        last(few.training_deviance()?)
    );
    println!(
        "100-round ensemble: training accuracy={}%  deviance={}",
        N(100.0 * model.training_accuracy()?),
        last(model.training_deviance()?)
    );
    few.plot_decision_regions("Petal.Length", "Petal.Width")?.save("gbm_iris_decision_regions_5rounds.svg")?;
    println!("Saved gbm_iris_decision_regions_5rounds.svg");
    Ok(())
}
