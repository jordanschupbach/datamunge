// CMD: just TARGET=svm_iris_ex run-java
// Support vector machine on iris: RBF vs. linear kernels, confusion matrix and predictions.
import datamunge.DataFrame;
import datamunge.SVM;
import datamunge.data.Data;

public class SvmIrisEx {
  public static void main(String[] args) {
    String formula = "Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width";
    var iris = Data.iris();
    System.out.println("iris: " + iris.nrows() + " rows x " + iris.ncols() + " cols\n");

    System.out.println("=== RBF kernel (default) ===");
    var rbfModel = new SVM(iris, formula);
    System.out.print(rbfModel.summary());
    System.out.println(
        "\nConfusion matrix (rows = actual, cols = predicted):\n" + rbfModel.confusionMatrix());

    System.out.println("\n=== Linear kernel, for comparison ===");
    var linearModel = new SVM(iris, formula, "linear");
    System.out.println("Training accuracy: " + (linearModel.trainingAccuracy() * 100) + "%");
    System.out.println("Support vectors: " + linearModel.numSupportVectors());

    var newdata = new DataFrame();
    newdata.addNumericColumn("Sepal.Length", new double[] {5.1, 6.0, 6.5, 6.2});
    newdata.addNumericColumn("Sepal.Width", new double[] {3.5, 2.7, 3.0, 2.8});
    newdata.addNumericColumn("Petal.Length", new double[] {1.4, 4.5, 5.5, 4.8});
    newdata.addNumericColumn("Petal.Width", new double[] {0.2, 1.5, 2.0, 1.8});
    System.out.println(
        "\nRBF predictions for new flowers (votes out of 3 one-vs-one pairs):\n"
            + rbfModel.predictFrame(newdata));
  }
}
