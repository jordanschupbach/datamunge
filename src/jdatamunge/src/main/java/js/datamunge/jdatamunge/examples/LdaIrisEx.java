package js.datamunge.jdatamunge.examples;

import js.datamunge.jdatamunge.DVector;
import js.datamunge.jdatamunge.DataFrame;
import js.datamunge.jdatamunge.LDA;

public class LdaIrisEx {
  static {
    System.loadLibrary("datamunge_jni");
  }

  public static void run() {
    var iris = DataFrame.iris();
    System.out.println("iris: " + iris.nrows() + " rows x " + iris.ncols() + " cols\n");

    var model = new LDA(iris, "Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width");
    model.print_summary();

    System.out.println("\nConfusion matrix (rows = actual, cols = predicted):");
    System.out.println(model.confusion_matrix().to_string());

    var newdata = new DataFrame();
    newdata.add_numeric_column("Sepal.Length", new DVector(new double[] {5.1, 6.0, 6.5, 6.2}));
    newdata.add_numeric_column("Sepal.Width", new DVector(new double[] {3.5, 2.7, 3.0, 2.8}));
    newdata.add_numeric_column("Petal.Length", new DVector(new double[] {1.4, 4.5, 5.5, 4.8}));
    newdata.add_numeric_column("Petal.Width", new DVector(new double[] {0.2, 1.5, 2.0, 1.8}));

    System.out.println("\nPredictions for new flowers:");
    System.out.println(model.predict_frame(newdata).to_string());

    model.save_discriminant_plot("lda_iris_discriminants.svg");
    System.out.println("\nSaved discriminant plot as lda_iris_discriminants.svg");
  }
}
