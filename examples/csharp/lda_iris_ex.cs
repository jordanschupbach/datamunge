using System;

class Program {
  static void Main() {
    var iris = DataFrame.iris();
    Console.WriteLine($"iris: {iris.nrows()} rows x {iris.ncols()} cols\n");

    var model = new LDA(iris, "Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width");
    model.print_summary();

    Console.WriteLine("\nConfusion matrix (rows = actual, cols = predicted):");
    Console.WriteLine(model.confusion_matrix().to_string());

    var newdata = new DataFrame();
    newdata.add_numeric_column("Sepal.Length", new DVector(new double[] { 5.1, 6.0, 6.5, 6.2 }));
    newdata.add_numeric_column("Sepal.Width", new DVector(new double[] { 3.5, 2.7, 3.0, 2.8 }));
    newdata.add_numeric_column("Petal.Length", new DVector(new double[] { 1.4, 4.5, 5.5, 4.8 }));
    newdata.add_numeric_column("Petal.Width", new DVector(new double[] { 0.2, 1.5, 2.0, 1.8 }));

    Console.WriteLine("\nPredictions for new flowers:");
    Console.WriteLine(model.predict_frame(newdata).to_string());

    model.save_discriminant_plot("lda_iris_discriminants.svg");
    Console.WriteLine("\nSaved discriminant plot as lda_iris_discriminants.svg");
  }
}
