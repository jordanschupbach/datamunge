% CMD: just TARGET=gbm_iris_ex run-octave
% Gradient-boosted trees (GBM) on iris: summary, confusion matrix, misclassified rows, 5 vs 100 rounds.
1;

% Shortest round-trip decimal text for x (like C#'s default double printing).
function s = num (x)
  for p = 1:17
    s = sprintf ('%.*g', p, x);
    if (str2double (s) == x)
      return;
    end
  end
end

iris = datamunge.Data.iris ();
printf ('iris: %d rows x %d cols\n\n', iris.nrows (), iris.ncols ());

model = datamunge.GBMClassifier (iris, 'Species ~ Petal.Length + Petal.Width');
model.print_summary ();

printf ('\nConfusion matrix (rows = actual, cols = predicted):\n%s\n', model.confusion_matrix ().to_string ());

printf ('\nMisclassified rows:\n');
predictions = model.predict (iris);
n = iris.nrows ();
misclassified = 0;
for i = 0:n - 1
  actual = iris.string_at ('Species', i);
  predicted = predictions{i + 1};
  if (strcmp (predicted, actual))
    continue;
  end
  misclassified++;
  printf ('  row %d: Petal.Length=%s Petal.Width=%s  actual=%s  predicted=%s\n', i, ...
          num (iris.numeric_at ('Petal.Length', i)), num (iris.numeric_at ('Petal.Width', i)), actual, predicted);
end
printf ('%d of %d misclassified (%s%%)\n', misclassified, n, num (100 * misclassified / n));

model.plot_training_deviance ().save ('gbm_iris_training_deviance.svg');
model.plot_decision_regions ('Petal.Length', 'Petal.Width').save ('gbm_iris_decision_regions.svg');
printf ('\nSaved gbm_iris_training_deviance.svg and gbm_iris_decision_regions.svg\n');

% A handful of boosting rounds vs a well-boosted ensemble: each additional round chips away
% at the training loss, gradually sharpening the decision boundary.
% datamunge.GBMClassifier (data, formula, n_trees, learning_rate, max_depth, min_samples_split,
%                          min_samples_leaf, subsample, seed)
few = datamunge.GBMClassifier (iris, 'Species ~ Petal.Length + Petal.Width', 5);
dev_few = few.training_deviance ();
dev_model = model.training_deviance ();
printf ('\n5-round ensemble:   training accuracy=%s%%  deviance=%s\n', num (100 * few.training_accuracy ()), num (dev_few(end)));
printf ('100-round ensemble: training accuracy=%s%%  deviance=%s\n', num (100 * model.training_accuracy ()), num (dev_model(end)));
few.plot_decision_regions ('Petal.Length', 'Petal.Width').save ('gbm_iris_decision_regions_5rounds.svg');
printf ('Saved gbm_iris_decision_regions_5rounds.svg\n');
