// CMD: just TARGET=naive_bayes_penguins_ex run-rust
// Naive Bayes on penguins: mixed numeric/categorical predictors, confusion matrix, misclassified rows, plots.
//
// Note: NaiveBayesClassifier drops incomplete rows for its own formula columns internally
// when fitting, matching every other formula-based model in this library.
use datamunge::*;
use datamunge_rust_examples::N;

fn main() -> datamunge::Result<()> {
    let penguins = data::penguins()?.expect("penguins");
    println!("penguins: {} rows x {} cols\n", penguins.nrows()?, penguins.ncols()?);

    // A mix of numeric (Gaussian likelihood) and categorical (frequency-table likelihood)
    // predictors in one formula -- each modeled independently given the class, per the naive
    // Bayes assumption.
    let model = NaiveBayesClassifier::new(&penguins, "species ~ bill_length_mm + bill_depth_mm + island + sex")?;
    model.print_summary()?;

    println!(
        "\nConfusion matrix (rows = actual, cols = predicted):\n{}",
        model.confusion_matrix()?.expect("confusion matrix")
    );

    println!("\nMisclassified rows:");
    let predictions = model.predict(&penguins)?;
    let mut misclassified = 0;
    let columns = ["species", "bill_length_mm", "bill_depth_mm", "island", "sex"];
    for (i, predicted) in predictions.iter().enumerate() {
        if columns.iter().map(|c| penguins.is_null(c, i)).collect::<Result<Vec<_>>>()?.contains(&true) {
            continue;
        }
        let actual = penguins.string_at("species", i)?;
        if *predicted == actual {
            continue;
        }
        misclassified += 1;
        println!(
            "  row {i}: bill_length={} bill_depth={} island={} sex={}  actual={actual}  predicted={predicted}",
            N(penguins.numeric_at("bill_length_mm", i)?),
            N(penguins.numeric_at("bill_depth_mm", i)?),
            penguins.string_at("island", i)?,
            penguins.string_at("sex", i)?
        );
    }
    println!("{misclassified} misclassified (of {} rows, some incomplete)", penguins.nrows()?);

    // plot_decision_regions requires exactly two NUMERIC predictors, so build a separate
    // two-predictor model (bill measurements alone) just for visualization.
    let bill_only = NaiveBayesClassifier::new(&penguins, "species ~ bill_length_mm + bill_depth_mm")?;
    println!("\nbill-measurements-only model training accuracy: {}%", N(100.0 * bill_only.training_accuracy()?));
    bill_only
        .plot_classification(&penguins, "bill_length_mm", "bill_depth_mm")?
        .save("naive_bayes_penguins_classification.svg")?;
    bill_only
        .plot_decision_regions("bill_length_mm", "bill_depth_mm")?
        .save("naive_bayes_penguins_decision_regions.svg")?;
    println!("Saved naive_bayes_penguins_classification.svg and naive_bayes_penguins_decision_regions.svg");
    Ok(())
}
