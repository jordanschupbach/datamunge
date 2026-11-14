% CMD: just TARGET=xgboost_iris_ex run-octave
% XGBoost classifier on iris: summary, confusion matrix, misclassified rows, L2-regularization comparison.

1;

% Shortest round-trip decimal text for x (like C#'s default double printing).
function s = num (x)
  for d = 1:17
    s = sprintf ('%.*g', d, x);
    if str2double (s) == x
      break;
    end
  end
end

iris = datamunge.Data.iris ();
printf ('iris: %d rows x %d cols\n\n', iris.nrows (), iris.ncols ());

model = datamunge.XGBoostClassifier (iris, 'Species ~ Petal.Length + Petal.Width');
model.print_summary ();

printf ('\nConfusion matrix (rows = actual, cols = predicted):\n%s\n', model.confusion_matrix ().to_string ());

printf ('\nMisclassified rows:\n');
predictions = model.predict (iris);
n = iris.nrows ();
misclassified = 0;
for i = 0:n - 1
  actual = iris.string_at ('Species', i);
  predicted = predictions{i + 1};
  if strcmp (predicted, actual)
    continue;
  end
  misclassified += 1;
  printf ('  row %d: Petal.Length=%s Petal.Width=%s  actual=%s  predicted=%s\n', i, ...
          num (iris.numeric_at ('Petal.Length', i)), num (iris.numeric_at ('Petal.Width', i)), actual, predicted);
end
printf ('%d of %d misclassified (%s%%)\n', misclassified, n, num (100.0 * misclassified / n));

model.plot_training_deviance ().save ('xgboost_iris_training_deviance.svg');
model.plot_decision_regions ('Petal.Length', 'Petal.Width').save ('xgboost_iris_decision_regions.svg');
printf ('\nSaved xgboost_iris_training_deviance.svg and xgboost_iris_decision_regions.svg\n');

% XGBoost's defining lever is regularization: a heavily L2-regularized model (large lambda)
% grows the same deep trees but keeps every leaf weight small, producing a much smoother
% decision boundary than the lightly-regularized default.
% datamunge.XGBoostClassifier (data, formula, n_trees, learning_rate, max_depth, lambda, alpha, gamma,
%                              min_child_weight, min_samples_leaf, subsample, colsample_bytree, seed)
heavy = datamunge.XGBoostClassifier (iris, 'Species ~ Petal.Length + Petal.Width', 100, 0.3, 6, 50.0);
deviance = model.training_deviance ();
heavy_deviance = heavy.training_deviance ();
printf ('\nlambda=1 (default):   training accuracy=%s%%  deviance=%s\n', ...
        num (100 * model.training_accuracy ()), num (deviance(end)));
printf ('lambda=50 (heavy L2): training accuracy=%s%%  deviance=%s\n', ...
        num (100 * heavy.training_accuracy ()), num (heavy_deviance(end)));
heavy.plot_decision_regions ('Petal.Length', 'Petal.Width').save ('xgboost_iris_decision_regions_heavy_lambda.svg');
printf ('Saved xgboost_iris_decision_regions_heavy_lambda.svg\n');
