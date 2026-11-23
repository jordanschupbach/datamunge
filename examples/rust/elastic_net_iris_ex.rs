// CMD: just TARGET=elastic_net_iris_ex run-rust
// Ridge, Lasso and Elastic Net on iris, with coefficient-path and CV-curve figures.
use datamunge::*;

const FORMULA: &str = "Petal.Length ~ Sepal.Length + Sepal.Width + Petal.Width + Sepal.Length:Petal.Width";

fn main() -> datamunge::Result<()> {
    let iris = data::iris()?.expect("iris");
    println!("iris: {} rows x {} cols", iris.nrows()?, iris.ncols()?);
    println!("formula: {FORMULA}\n");

    println!("=================== Ridge ===================");
    let ridge = Ridge::new(&iris, FORMULA)?;
    ridge.print_summary()?;

    println!("\n=================== Lasso ===================");
    let lasso = Lasso::new(&iris, FORMULA)?;
    lasso.print_summary()?;

    println!("\n================= Elastic Net =================");
    // ElasticNet::new_with_<last param>(data, formula, alpha=0.5, lambda, n_lambda, cv_folds, standardize, seed)
    let elastic = ElasticNet::new_with_alpha(&iris, FORMULA, 0.5)?;
    elastic.print_summary()?;

    println!(
        "\nSaved figures showing how each model's coefficients respond to the regularization strength, \
         and the cross-validation curve used to pick it:"
    );

    let figures = [
        (ridge.plot_coefficient_path()?, ridge.plot_cv_curve()?, "ridge", "ridge:      "),
        (lasso.plot_coefficient_path()?, lasso.plot_cv_curve()?, "lasso", "lasso:      "),
        (elastic.plot_coefficient_path()?, elastic.plot_cv_curve()?, "elasticnet", "elastic net:"),
    ];
    for (path, cv, name, label) in figures {
        path.save(&format!("elastic_net_{name}_path.svg"))?;
        cv.save(&format!("elastic_net_{name}_cv.svg"))?;
        println!("  {label} elastic_net_{name}_path.svg, elastic_net_{name}_cv.svg");
    }
    Ok(())
}
