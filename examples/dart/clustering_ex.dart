// CMD: just TARGET=clustering_ex run-dart
// Clustering iris: k-means, agglomerative (several linkages, dendrogram re-cut) and DBSCAN.
import 'package:datamunge/datamunge.dart';

/// A double as C#'s default ToString() prints it: the shortest round-trip digits, in E notation
/// when the decimal exponent is below -4 or at least 15 (6, 0.1, 1E-05, 1E+15).
String fmt(double x) {
  if (x.isNaN) return 'NaN';
  if (x.isInfinite) return x < 0 ? '-∞' : '∞';
  var s = x.toString();
  final negative = s.startsWith('-');
  if (negative) s = s.substring(1);
  var mantissa = s;
  var exponent = 0;
  final e = s.indexOf('e');
  if (e >= 0) {
    mantissa = s.substring(0, e);
    exponent = int.parse(s.substring(e + 1));
  }
  final parts = mantissa.split('.');
  var digits = parts[0] + (parts.length > 1 ? parts[1] : '');
  var point = parts[0].length + exponent;
  while (digits.startsWith('0')) {
    digits = digits.substring(1);
    point--;
  }
  while (digits.endsWith('0')) {
    digits = digits.substring(0, digits.length - 1);
  }
  final sign = negative ? '-' : '';
  if (digits.isEmpty) return '${sign}0';
  final exp10 = point - 1;
  if (exp10 < -4 || exp10 >= 15) {
    final rest = digits.substring(1);
    final magnitude = exp10.abs().toString().padLeft(2, '0');
    return '$sign${digits[0]}${rest.isEmpty ? '' : '.$rest'}E${exp10 < 0 ? '-' : '+'}$magnitude';
  }
  if (point <= 0) return '${sign}0.${'0' * -point}$digits';
  if (point >= digits.length) return '$sign$digits${'0' * (point - digits.length)}';
  return '$sign${digits.substring(0, point)}.${digits.substring(point)}';
}

void main() {
  final iris = Data.iris()!;
  final species = List.generate(iris.nrows(), (i) => iris.stringAt('Species', i));
  const featureColumns = ['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width'];

  // For each cluster, print its size and its majority species (negative labels are noise).
  void reportPurity(List<int> clusterLabels, int nClusters) {
    final votes = List.generate(nClusters, (_) => <String, int>{});
    for (var i = 0; i < clusterLabels.length && i < species.length; i++) {
      final label = clusterLabels[i];
      if (label >= 0) votes[label][species[i]] = (votes[label][species[i]] ?? 0) + 1;
    }
    for (var c = 0; c < nClusters; c++) {
      final total = votes[c].values.fold(0, (a, b) => a + b);
      if (total == 0) continue;
      var (bestSpecies, best) = ('', -1);
      for (final MapEntry(:key, :value) in votes[c].entries) {
        if (value > best) (bestSpecies, best) = (key, value);
      }
      print('  cluster $c: $total points, majority $bestSpecies ($best/$total)');
    }
  }

  print('=================== K-means (k=3) on iris ===================');
  final kmeans = KMeans(iris, featureColumns, 3);
  print('Inertia: ${fmt(kmeans.inertia())}, iterations (best run): ${kmeans.iterationsUsed()}');
  reportPurity(kmeans.labels(), 3);

  print('\n=================== Agglomerative clustering on iris ===================');
  for (final linkage in ['ward', 'complete', 'average']) {
    final model = AgglomerativeClustering(iris, featureColumns, 3, linkage);
    print('--- linkage=$linkage ---');
    reportPurity(model.labels(), 3);
  }

  print('\n--- Re-cutting the ward dendrogram at k=2 without refitting ---');
  final wardModel = AgglomerativeClustering(iris, featureColumns, 3, 'ward');
  reportPurity(wardModel.cut(2), 2);

  print('\n=================== DBSCAN on iris ===================');
  final dbscan = Dbscan(iris, featureColumns, 0.6, 5);
  print('Clusters found: ${dbscan.nClusters()}, noise points: ${dbscan.nNoise()} (of ${dbscan.observations()})');
  reportPurity(dbscan.labels(), dbscan.nClusters());
}
