package js.datamunge.jdatamunge.examples;

import js.datamunge.jdatamunge.DVector;
import js.datamunge.jdatamunge.DataFrame;
import js.datamunge.jdatamunge.SVM;

public class SvmIrisEx {
  static {
    System.loadLibrary("datamunge_jni");
  }

  public static void run() {
    var iris = DataFrame.iris();
    System.out.println("iris: " + iris.nrows() + " rows x " + iris.ncols() + " cols\n");

    System.out.println("=== RBF kernel (default) ===");
    var rbfModel = new SVM(iris, "Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width");
    rbfModel.print_summary();
    System.out.println("\nConfusion matrix (rows = actual, cols = predicted):");
    System.out.println(rbfModel.confusion_matrix().to_string());

    System.out.println("\n=== Linear kernel, for comparison ===");
    var linearModel = new SVM(iris, "Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width", "linear");
    System.out.println("Training accuracy: " + (linearModel.training_accuracy() * 100.0) + "%");
    System.out.println("Support vectors: " + linearModel.num_support_vectors());

    var newdata = new DataFrame();
    newdata.add_numeric_column("Sepal.Length", new DVector(new double[] {5.1, 6.0, 6.5, 6.2}));
    newdata.add_numeric_column("Sepal.Width", new DVector(new double[] {3.5, 2.7, 3.0, 2.8}));
    newdata.add_numeric_column("Petal.Length", new DVector(new double[] {1.4, 4.5, 5.5, 4.8}));
    newdata.add_numeric_column("Petal.Width", new DVector(new double[] {0.2, 1.5, 2.0, 1.8}));

    System.out.println("\nRBF predictions for new flowers (votes out of 3 one-vs-one pairs):");
    System.out.println(rbfModel.predict_frame(newdata).to_string());
  }
}
