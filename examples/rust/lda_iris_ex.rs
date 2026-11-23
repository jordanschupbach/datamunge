// CMD: just TARGET=lda_iris_ex run-rust
// Linear discriminant analysis on iris: summary, confusion matrix, predictions and discriminant plot.
use datamunge::*;

fn main() -> datamunge::Result<()> {
    let iris = data::iris()?.expect("iris");
    println!("iris: {} rows x {} cols\n", iris.nrows()?, iris.ncols()?);

    let model = LDA::new(&iris, "Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width")?;
    model.print_summary()?;

    println!(
        "\nConfusion matrix (rows = actual, cols = predicted):\n{}",
        model.confusion_matrix()?.expect("confusion matrix")
    );

    let newdata = DataFrame::new()?;
    newdata.add_numeric_column("Sepal.Length", &[5.1, 6.0, 6.5, 6.2])?;
    newdata.add_numeric_column("Sepal.Width", &[3.5, 2.7, 3.0, 2.8])?;
    newdata.add_numeric_column("Petal.Length", &[1.4, 4.5, 5.5, 4.8])?;
    newdata.add_numeric_column("Petal.Width", &[0.2, 1.5, 2.0, 1.8])?;
    println!("\nPredictions for new flowers:\n{}", model.predict_frame(&newdata)?.expect("predictions"));

    model.save_discriminant_plot("lda_iris_discriminants.svg")?;
    println!("\nSaved discriminant plot as lda_iris_discriminants.svg");
    Ok(())
}
