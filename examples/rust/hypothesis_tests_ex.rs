// CMD: just TARGET=hypothesis_tests_ex run-rust
// Classical hypothesis tests (t, Wilcoxon, ANOVA, correlation, F, normality, chi-squared, binomial) on iris.
use datamunge::*;
use datamunge_rust_examples::N;

fn print_result(label: &str, r: &HypothesisTestResult) {
    let alternative = match r.alternative() {
        Alternative::LESS => "less",
        Alternative::GREATER => "greater",
        _ => "two.sided",
    };
    let mut line = format!("{label}: statistic={}", N(r.statistic()));
    if r.parameter1() != 0.0 {
        line += &format!(", df1={}", N(r.parameter1()));
    }
    if r.parameter2() != 0.0 {
        line += &format!(", df2={}", N(r.parameter2()));
    }
    line += &format!(", p={} ({alternative})", N(r.p_value()));
    if r.has_conf_int() {
        line += &format!(", CI=[{}, {}]", N(r.conf_int_lower()), N(r.conf_int_upper()));
    }
    println!("{line} -- {}", r.method());
}

fn column_for_species(df: &DataFrame, column: &str, species: &str) -> Result<Vec<f64>> {
    let mut values = Vec::new();
    for i in 0..df.nrows()? {
        if df.string_at("Species", i)? == species {
            values.push(df.numeric_at(column, i)?);
        }
    }
    Ok(values)
}

fn column(df: &DataFrame, column: &str) -> Result<Vec<f64>> {
    (0..df.nrows()?).map(|i| df.numeric_at(column, i)).collect()
}

fn main() -> datamunge::Result<()> {
    let iris = data::iris()?.expect("iris");

    let setosa_petal = column_for_species(&iris, "Petal.Length", "setosa")?;
    let versicolor_petal = column_for_species(&iris, "Petal.Length", "versicolor")?;
    let virginica_petal = column_for_species(&iris, "Petal.Length", "virginica")?;

    println!("=================== t-tests: petal length, setosa vs. versicolor ===================");
    print_result("Welch two-sample t-test", &t_test_two_sample(&setosa_petal, &versicolor_petal)?);
    print_result("Wilcoxon rank-sum test", &wilcoxon_rank_sum_test(&setosa_petal, &versicolor_petal)?);

    println!("\n=================== One-way ANOVA / Kruskal-Wallis across all 3 species ===================");
    let all_petal = [setosa_petal.as_slice(), &versicolor_petal, &virginica_petal].concat();
    let sizes = [setosa_petal.len(), versicolor_petal.len(), virginica_petal.len()];
    print_result("One-way ANOVA", &one_way_anova(&all_petal, &sizes)?);
    print_result("Kruskal-Wallis", &kruskal_wallis_test(&all_petal, &sizes)?);

    println!("\n=================== Correlation: sepal length vs. petal length ===================");
    let sepal_length = column(&iris, "Sepal.Length")?;
    let petal_length = column(&iris, "Petal.Length")?;
    print_result("Pearson correlation", &pearson_correlation_test(&sepal_length, &petal_length)?);
    print_result("Spearman correlation", &spearman_correlation_test(&sepal_length, &petal_length)?);

    println!("\n=================== F-test: petal length variance, setosa vs. virginica ===================");
    print_result("F test", &f_test_variance(&setosa_petal, &virginica_petal)?);

    println!("\n=================== Normality: is sepal length normally distributed within setosa? ===================");
    let setosa_sepal = column_for_species(&iris, "Sepal.Length", "setosa")?;
    print_result("Shapiro-Francia", &shapiro_francia_test(&setosa_sepal)?);
    print_result("KS vs. fitted normal", &ks_test_one_sample_normal_with_sd(&setosa_sepal, 5.006, 0.3525)?);

    println!("\n=================== Chi-squared / Fisher: is petal length \"long\" independent of species? ===================");
    let mut sorted = all_petal.clone();
    sorted.sort_by(f64::total_cmp);
    let median_all = sorted[sorted.len() / 2];
    let count_long = |v: &[f64]| v.iter().filter(|&&x| x > median_all).count();
    {
        let setosa_long = count_long(&setosa_petal);
        let setosa_short = setosa_petal.len() - setosa_long;
        let versicolor_long = count_long(&versicolor_petal);
        let versicolor_short = versicolor_petal.len() - versicolor_long;
        println!("table: setosa=[{setosa_long},{setosa_short}] versicolor=[{versicolor_long},{versicolor_short}]");
        let table = [setosa_long, setosa_short, versicolor_long, versicolor_short].map(|c| c as f64);
        print_result("Chi-squared independence", &chi_squared_test_independence(&table, 2, 2)?);
        print_result(
            "Fisher's exact test",
            &fisher_exact_test_2x2(setosa_long, setosa_short, versicolor_long, versicolor_short)?,
        );
    }

    println!("\n=================== Proportion / binomial: fraction of \"long\" petals overall ===================");
    {
        let long_count = count_long(&all_petal);
        let n = all_petal.len();
        print_result("One-sample proportion test (vs 0.5)", &proportion_test_one_sample_with_p(long_count, n, 0.5)?);
        print_result("Exact binomial test (vs 0.5)", &binomial_test_with_p(long_count, n, 0.5)?);
    }
    Ok(())
}
