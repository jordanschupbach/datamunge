// CMD: just TARGET=lda_iris_ex run-java
// Linear discriminant analysis on iris: summary, confusion matrix, predictions and discriminant plot.
import datamunge.DataFrame;
import datamunge.LDA;
import datamunge.data.Data;

public class LdaIrisEx {
  public static void main(String[] args) {
    var iris = Data.iris();
    System.out.println("iris: " + iris.nrows() + " rows x " + iris.ncols() + " cols\n");

    var model =
        new LDA(iris, "Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width");
    model.printSummary();

    System.out.println(
        "\nConfusion matrix (rows = actual, cols = predicted):\n" + model.confusionMatrix());

    var newdata = new DataFrame();
    newdata.addNumericColumn("Sepal.Length", new double[] {5.1, 6.0, 6.5, 6.2});
    newdata.addNumericColumn("Sepal.Width", new double[] {3.5, 2.7, 3.0, 2.8});
    newdata.addNumericColumn("Petal.Length", new double[] {1.4, 4.5, 5.5, 4.8});
    newdata.addNumericColumn("Petal.Width", new double[] {0.2, 1.5, 2.0, 1.8});
    System.out.println("\nPredictions for new flowers:\n" + model.predictFrame(newdata));

    model.saveDiscriminantPlot("lda_iris_discriminants.svg");
    System.out.println("\nSaved discriminant plot as lda_iris_discriminants.svg");
  }
}
