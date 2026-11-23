// CMD: just TARGET=stl_ex run-dart
// C++ std::vector and std::pair values in Dart: vectors are List<T>, pairs are records.
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
  const n = 100;
  final v = List.generate(n, (i) => i * 1.5);
  v.map(fmt).forEach(print);
  final v2 = List.generate(n, (i) => (i * 1.5).floor());
  v2.forEach(print);

  final p = (3, 4);
  print('p: (${p.$1}, ${p.$2})');

  final p2 = (10, 20);
  print('p2: (${p2.$1}, ${p2.$2})');

  // A C++ std::pair<double,double> comes back as a record.
  final p3 = makeDpair(1.25, 2.75);
  print('make_dpair(1.25, 2.75): (${fmt(p3.$1)}, ${fmt(p3.$2)}), sum_dpair = ${fmt(sumDpair(p3))}');
}
