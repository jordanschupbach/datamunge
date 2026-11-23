// CMD: just TARGET=pca_ex run-dart
// PCA on iris: summary, PC1 loadings, scores frame, score/scree plots, scaled vs unscaled.
import 'package:datamunge/datamunge.dart';

void main() {
  final iris = Data.iris()!;
  const featureColumns = ['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width'];

  // species labels, for the grouped (colored-by-species) scores plot
  final species = List.generate(iris.nrows(), (i) => iris.stringAt('Species', i));

  print('=================== PCA on iris (scaled) ===================');
  final pca = Pca(iris, featureColumns, true, true);
  pca.printSummary();

  print('\nPC1 loadings (which original features drive it):');
  final names = pca.featureNames(), loadings = pca.componentLoadings(0);
  for (var i = 0; i < names.length && i < loadings.length; i++) {
    print('  ${names[i]}: ${loadings[i].toStringAsFixed(4)}');
  }

  print('\nTraining scores as a DataFrame:\n${pca.scoresFrame()!.toString()}');

  pca.plotScoresGrouped(species).saveSvg('pca_iris_scores.svg');
  print('\nScores scatter (colored by species) saved as pca_iris_scores.svg');

  pca.plotScree().saveSvg('pca_iris_scree.svg');
  print('Scree plot saved as pca_iris_scree.svg');

  print('\n=================== PCA on iris (unscaled) ===================');
  final unscaled = Pca(iris, featureColumns, true, false);
  print('PC1 explains ${(100 * unscaled.explainedVarianceRatio()[0]).toStringAsFixed(2)}% of variance '
      '(vs. ${(100 * pca.explainedVarianceRatio()[0]).toStringAsFixed(2)}% scaled) -- '
      "Sepal.Length's larger raw variance dominates the unscaled covariance matrix.");
}
