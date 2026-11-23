// CMD: just TARGET=dataframe_ex run-dart
// DataFrame basics: nulls, drop_duplicates, fill_null, select/sort, group-by, join and summaries.
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
  final sales = DataFrame();
  sales.addStringColumn('region', ['west', 'west', 'east', 'south', 'south', 'south']);
  sales.addStringColumn('product', ['widget', 'widget', 'widget', 'gizmo', 'gizmo', 'gizmo']);
  sales.addNumericColumn('sales', [10.0, 10.0, 14.0, 8.0, 0.0, 11.0], [1, 1, 1, 1, 0, 1]);
  sales.addStringColumn('quarter', ['Q1', 'Q1', 'Q1', 'Q2', 'Q2', ''], [1, 1, 1, 1, 1, 0]);

  print('raw data\n${sales.toString()}\n');

  final cleaned = sales.dropDuplicates(['region', 'product', 'sales', 'quarter'])!;
  cleaned.fillNullString('quarter', 'unknown');
  cleaned.fillNullNumeric('sales', 0.0);
  print('after drop_duplicates + fill_null\n${cleaned.toString()}\n');

  final selected = cleaned.select(['region', 'sales', 'quarter'])!.sortBy('sales', false)!;
  print('selected + sorted\n${selected.toString()}\n');

  final grouped = cleaned.groupBySum(['region'], ['sales'])!.sortBy('sales', false)!;
  print('group_by_sum(region)\n${grouped.toString()}\n');

  final targets = DataFrame();
  targets.addStringColumn('region', ['west', 'east', 'south']);
  targets.addNumericColumn('target', [18.0, 12.0, 25.0]);
  print('joined with targets\n${grouped.join(targets, 'region', 'region', 'left')!.toString()}\n');

  final shape = cleaned.shape();
  print('shape = (${shape[0]}, ${shape[1]})');
  print('sales count = ${cleaned.numericCount('sales')}');
  print('sales nulls = ${cleaned.numericNullCount('sales')}');
  print('sales sum = ${fmt(cleaned.numericSum('sales'))}');
  print('sales mean = ${fmt(cleaned.numericMean('sales'))}');
}
