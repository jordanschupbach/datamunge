// CMD: just TARGET=mds_ex run-java
// Classical multidimensional scaling of iris: euclidean embedding, plot, and a manhattan comparison.
import datamunge.MDS;
import datamunge.data.Data;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

public class MdsEx {
  public static void main(String[] args) {
    var iris = Data.iris();
    List<String> featureColumns =
        List.of("Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width");
    List<String> species = new ArrayList<>();
    for (long i = 0; i < iris.nrows(); i++) {
      species.add(iris.stringAt("Species", i));
    }

    System.out.println(
        "=================== Classical MDS on iris (euclidean) ===================");
    var mds = new MDS(iris, featureColumns, 2, "euclidean");
    mds.printSummary();
    System.out.flush();

    System.out.println("\nEmbedding as a DataFrame:\n" + mds.embeddingFrame());

    mds.plotEmbeddingGrouped(species).saveSvg("mds_iris_embedding.svg");
    System.out.println("\nEmbedding scatter (colored by species) saved as mds_iris_embedding.svg");

    System.out.println(
        "\n=================== Classical MDS on iris (manhattan) ===================");
    var manhattanMds = new MDS(iris, featureColumns, 2, "manhattan");
    System.out.println(
        String.format(
            Locale.ROOT,
            "Goodness of fit: euclidean=%.4f, manhattan=%.4f",
            mds.goodnessOfFit(),
            manhattanMds.goodnessOfFit()));
  }
}
