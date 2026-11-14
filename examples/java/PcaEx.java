// CMD: just TARGET=pca_ex run-java
// PCA on iris: summary, PC1 loadings, scores frame, score/scree plots, scaled vs unscaled.
import datamunge.PCA;
import datamunge.data.Data;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

public class PcaEx {
  public static void main(String[] args) {
    var iris = Data.iris();
    List<String> featureColumns =
        List.of("Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width");

    // species labels, for the grouped (colored-by-species) scores plot
    List<String> species = new ArrayList<>();
    for (long i = 0; i < iris.nrows(); i++) {
      species.add(iris.stringAt("Species", i));
    }

    System.out.println("=================== PCA on iris (scaled) ===================");
    var pca = new PCA(iris, featureColumns, true, true);
    pca.printSummary();
    System.out.flush();

    System.out.println("\nPC1 loadings (which original features drive it):");
    List<String> names = pca.featureNames();
    double[] loadings = pca.componentLoadings(0);
    for (int i = 0; i < names.size(); i++) {
      System.out.println(String.format(Locale.ROOT, "  %s: %.4f", names.get(i), loadings[i]));
    }

    System.out.println("\nTraining scores as a DataFrame:\n" + pca.scoresFrame());

    pca.plotScoresGrouped(species).saveSvg("pca_iris_scores.svg");
    System.out.println("\nScores scatter (colored by species) saved as pca_iris_scores.svg");

    pca.plotScree().saveSvg("pca_iris_scree.svg");
    System.out.println("Scree plot saved as pca_iris_scree.svg");

    System.out.println("\n=================== PCA on iris (unscaled) ===================");
    var unscaled = new PCA(iris, featureColumns, true, false);
    System.out.println(
        String.format(
            Locale.ROOT,
            "PC1 explains %.2f%% of variance (vs. %.2f%% scaled) -- Sepal.Length's larger raw"
                + " variance dominates the unscaled covariance matrix.",
            100 * unscaled.explainedVarianceRatio()[0],
            100 * pca.explainedVarianceRatio()[0]));
  }
}
