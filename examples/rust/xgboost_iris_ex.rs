// CMD: just TARGET=xgboost_iris_ex run-rust
// XGBoost classifier on iris: summary, confusion matrix, misclassified rows, L2-regularization comparison.
use datamunge::*;
use datamunge_rust_examples::N;

fn main() -> datamunge::Result<()> {
    let iris = data::iris()?.expect("iris");
    println!("iris: {} rows x {} cols\n", iris.nrows()?, iris.ncols()?);

    let model = XGBoostClassifier::new(&iris, "Species ~ Petal.Length + Petal.Width")?;
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

    model.plot_training_deviance()?.save("xgboost_iris_training_deviance.svg")?;
    model.plot_decision_regions("Petal.Length", "Petal.Width")?.save("xgboost_iris_decision_regions.svg")?;
    println!("\nSaved xgboost_iris_training_deviance.svg and xgboost_iris_decision_regions.svg");

    // XGBoost's defining lever is regularization: a heavily L2-regularized model (large lambda)
    // grows the same deep trees but keeps every leaf weight small, producing a much smoother
    // decision boundary than the lightly-regularized default.
    // XGBoostClassifier::new_with_<last param>(data, formula, n_trees, learning_rate, max_depth, lambda,
    //     alpha, gamma, min_child_weight, min_samples_leaf, subsample, colsample_bytree, seed)
    let heavy = XGBoostClassifier::new_with_lambda(&iris, "Species ~ Petal.Length + Petal.Width", 100, 0.3, 6, 50.0)?;
    let last = |deviance: Vec<f64>| N(*deviance.last().expect("deviance"));
    println!(
        "\nlambda=1 (default):   training accuracy={}%  deviance={}",
        N(100.0 * model.training_accuracy()?),
        last(model.training_deviance()?)
    );
    println!(
        "lambda=50 (heavy L2): training accuracy={}%  deviance={}",
        N(100.0 * heavy.training_accuracy()?),
        last(heavy.training_deviance()?)
    );
    heavy.plot_decision_regions("Petal.Length", "Petal.Width")?.save("xgboost_iris_decision_regions_heavy_lambda.svg")?;
    println!("Saved xgboost_iris_decision_regions_heavy_lambda.svg");
    Ok(())
}
