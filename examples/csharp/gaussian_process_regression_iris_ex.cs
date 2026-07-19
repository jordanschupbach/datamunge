using System;

class Program {
  static void Main() {
    const string FORMULA = "Petal.Length ~ Petal.Width";

    var iris = DataFrame.iris();
    Console.WriteLine($"iris: {iris.nrows()} rows x {iris.ncols()} cols");
    Console.WriteLine($"formula: {FORMULA}\n");

    var model = new GaussianProcessRegression(iris, FORMULA);
    model.print_summary();

    model.plot_fit(iris).save("gpr_iris_fit.svg");
    model.plot_length_scale_profile().save("gpr_iris_length_scale_profile.svg");
    Console.WriteLine("\nSaved gpr_iris_fit.svg and gpr_iris_length_scale_profile.svg");

    var query = new DataFrame();
    query.add_numeric_column("Petal.Width", new DVector(new double[] { 0.2, 1.3, 2.5, 10.0 }));
    var detail = model.predict_frame(query, "confidence");
    Console.WriteLine("\nPredictions with 95% confidence intervals:");
    Console.WriteLine(detail.to_string());
    Console.WriteLine("(Petal.Width=10.0 is far outside the training range [0.1, 2.5] -- note how much wider its\n interval is than the in-range predictions.)");
  }
}
