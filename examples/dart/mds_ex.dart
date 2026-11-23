// CMD: just TARGET=mds_ex run-dart
// Classical multidimensional scaling of iris: euclidean embedding, plot, and a manhattan comparison.
import 'package:datamunge/datamunge.dart';

void main() {
  final iris = Data.iris()!;
  const featureColumns = ['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width'];
  final species = List.generate(iris.nrows(), (i) => iris.stringAt('Species', i));

  print('=================== Classical MDS on iris (euclidean) ===================');
  final mds = Mds(iris, featureColumns, 2, 'euclidean');
  mds.printSummary();

  print('\nEmbedding as a DataFrame:\n${mds.embeddingFrame()!.toString()}');

  mds.plotEmbeddingGrouped(species).saveSvg('mds_iris_embedding.svg');
  print('\nEmbedding scatter (colored by species) saved as mds_iris_embedding.svg');

  print('\n=================== Classical MDS on iris (manhattan) ===================');
  final manhattanMds = Mds(iris, featureColumns, 2, 'manhattan');
  print('Goodness of fit: euclidean=${mds.goodnessOfFit().toStringAsFixed(4)}, '
      'manhattan=${manhattanMds.goodnessOfFit().toStringAsFixed(4)}');
}
