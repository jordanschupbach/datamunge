// CMD: just TARGET=lm_ex run-dart
// Linear model: mpg ~ hp + wt + transmission, its ANOVA table and predictions.
import 'package:datamunge/datamunge.dart';

void main() {
  final cars = DataFrame();
  cars.addNumericColumn('hp', [110.0, 110.0, 93.0, 110.0, 175.0, 105.0, 245.0, 62.0, 95.0, 123.0]);
  cars.addNumericColumn('wt', [2.62, 2.875, 2.32, 3.215, 3.44, 3.46, 3.57, 3.19, 3.15, 3.44]);
  cars.addStringColumn('transmission', ['manual', 'manual', 'manual', 'automatic', 'automatic',
      'automatic', 'automatic', 'automatic', 'automatic', 'automatic']);
  cars.addNumericColumn('mpg', [21.0, 21.0, 22.8, 21.4, 18.7, 18.1, 14.3, 24.4, 22.8, 19.2]);

  print('Fitting: mpg ~ hp + wt + transmission\n');
  final model = Lm(cars, 'mpg ~ hp + wt + transmission');
  model.printSummary();

  print('\nSequential ANOVA:\n${model.anova()!.toString()}');

  final newcars = DataFrame();
  newcars.addNumericColumn('hp', [150.0, 90.0]);
  newcars.addNumericColumn('wt', [3.0, 2.5]);
  newcars.addStringColumn('transmission', ['manual', 'automatic']);
  print('\nPredictions with 95% confidence intervals:\n${model.predictFrame(newcars, 'confidence')!.toString()}');

  model.saveDiagnosticPlots('lm_ex_diagnostics');
  print('\nSaved diagnostic plots as lm_ex_diagnostics_*.svg');
}
