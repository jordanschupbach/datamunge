// CMD: just TARGET=tensor_ex run-dart
// Tensors: construction, shape ops, broadcasting, reductions, linear algebra, masks, iris.
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

String shapeString(Tensor tensor) => '[${tensor.shape().join(', ')}]';

void main() {
  print('=================== Construction ===================');
  print('zeros([2,3]): ${Tensor.zeros([2, 3])!.toString()}');
  print('eye(3): ${Tensor.eye(3)!.toString()}');

  final r = Tensor.arange(0.0, 12.0, 1.0)!.reshape([3, 4])!;
  print('arange(0,12).reshape([3,4]): ${r.toString()}');

  print('\n=================== Shape ops ===================');
  print('transpose -> shape ${shapeString(r.transpose()!)}');
  print('slice(axis=1, start=1, stop=3): ${r.slice(1, 1, 3)!.toString()}');

  print('\n=================== Broadcasting arithmetic ===================');
  {
    final col = Tensor.fromValues([3, 1], [1.0, 2.0, 3.0])!;
    final row = Tensor.fromValues([1, 4], [10.0, 20.0, 30.0, 40.0])!;
    print('(3,1) + (1,4) -> ${col.add(row)!.toString()}');
  }

  print('\n=================== Reductions ===================');
  print('r.sum() = ${fmt(r.sum())}, r.mean() = ${fmt(r.mean())}');
  print('column means (axis=0): ${r.meanAxis(0)!.toString()}');

  print('\n=================== Linear algebra ===================');
  {
    final a = Tensor.fromValues([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])!;
    final b = Tensor.fromValues([3, 2], [7.0, 8.0, 9.0, 10.0, 11.0, 12.0])!;
    print('matmul(2x3, 3x2) -> ${a.matmul(b)!.toString()}');
  }
  {
    final v1 = Tensor.fromValues([3], [1.0, 2.0, 3.0])!;
    final v2 = Tensor.fromValues([3], [4.0, 5.0, 6.0])!;
    print('dot([1,2,3], [4,5,6]) = ${fmt(v1.dot(v2))}');
    print('outer(v1, v2) -> ${v1.outer(v2)!.toString()}');
  }

  print('\n=================== Comparisons & masks ===================');
  {
    final mask = r.greaterEqual(Tensor.full([3, 4], 6.0)!)!;
    print('r >= 6 -> ${mask.toString()}');
    print('count(r >= 6) = ${fmt(mask.sum())}');
  }

  print('\n=================== A real dataset as a Tensor ===================');
  {
    final iris = Data.iris()!;
    final n = iris.nrows();
    const columns = ['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width'];
    final flat = <double>[];
    for (var i = 0; i < n; i++) {
      for (final column in columns) {
        flat.add(iris.numericAt(column, i));
      }
    }
    final x = Tensor.fromValues([n, 4], flat)!;
    print('iris feature tensor shape: ${shapeString(x)}');

    final featureMeans = x.meanAxis(0)!;
    final centered = x.subtract(featureMeans.reshape([1, 4])!)!;
    final scatter = centered.transpose()!.matmul(centered)!;
    print('feature means (Sepal.Length, Sepal.Width, Petal.Length, Petal.Width): ${featureMeans.toString()}');
    print('(X-mean)^T (X-mean) [4x4 scatter matrix]: ${scatter.toString(16)}');

    final species = <String>[];
    for (var i = 0; i < n; i++) {
      species.add(iris.stringAt('Species', i));
    }
    final speciesTensor = Tensor.fromStringValues([n], species)!;
    final setosaMask = speciesTensor.equal(Tensor.fromStringValues([1], ['setosa'])!)!;
    print('setosa count = ${fmt(setosaMask.sum())} (of $n rows)');
  }
}
