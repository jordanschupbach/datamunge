// CMD: just TARGET=svm_iris_ex run-rust
// Support vector machine on iris: RBF vs. linear kernels, confusion matrix and predictions.
use datamunge::*;
use datamunge_rust_examples::N;

const FORMULA: &str = "Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width";

fn main() -> datamunge::Result<()> {
    let iris = data::iris()?.expect("iris");
    println!("iris: {} rows x {} cols\n", iris.nrows()?, iris.ncols()?);

    println!("=== RBF kernel (default) ===");
    let rbf_model = SVM::new(&iris, FORMULA)?;
    rbf_model.print_summary()?;
    println!(
        "\nConfusion matrix (rows = actual, cols = predicted):\n{}",
        rbf_model.confusion_matrix()?.expect("confusion matrix")
    );

    println!("\n=== Linear kernel, for comparison ===");
    let linear_model = SVM::new_with_kernel(&iris, FORMULA, "linear")?;
    println!("Training accuracy: {}%", N(linear_model.training_accuracy()? * 100.0));
    println!("Support vectors: {}", linear_model.num_support_vectors()?);

    let newdata = DataFrame::new()?;
    newdata.add_numeric_column("Sepal.Length", &[5.1, 6.0, 6.5, 6.2])?;
    newdata.add_numeric_column("Sepal.Width", &[3.5, 2.7, 3.0, 2.8])?;
    newdata.add_numeric_column("Petal.Length", &[1.4, 4.5, 5.5, 4.8])?;
    newdata.add_numeric_column("Petal.Width", &[0.2, 1.5, 2.0, 1.8])?;
    println!(
        "\nRBF predictions for new flowers (votes out of 3 one-vs-one pairs):\n{}",
        rbf_model.predict_frame(&newdata)?.expect("predictions")
    );
    Ok(())
}
