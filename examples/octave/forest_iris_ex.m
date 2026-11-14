% CMD: just TARGET=forest_iris_ex run-octave
% Random forest classifier on iris: summary, confusion matrix, misclassifications, OOB accuracy.

1;

% Shortest round-trip decimal text for x (like C#'s default double printing).
function s = num (x)
  if (x == fix (x) && abs (x) < 1e15)
    s = sprintf ('%d', x);
    return;
  end
  for p = 1:17
    s = sprintf ('%.*g', p, x);
    if (str2double (s) == x)
      return;
    end
  end
end

iris = datamunge.Data.iris ();
printf ('iris: %d rows x %d cols\n\n', iris.nrows (), iris.ncols ());

model = datamunge.RandomForestClassifier (iris, 'Species ~ Petal.Length + Petal.Width');
model.print_summary ();

printf ('\nConfusion matrix (rows = actual, cols = predicted):\n');
printf ('%s\n', model.confusion_matrix ().to_string ());

printf ('\nMisclassified rows:\n');
predictions = model.predict (iris);
n = iris.nrows ();
misclassified = 0;
for i = 0:n-1
  actual = iris.string_at ('Species', i);
  predicted = predictions{i+1};
  if (strcmp (predicted, actual))
    continue;
  end
  misclassified++;
  printf ('  row %d: Petal.Length=%s Petal.Width=%s  actual=%s  predicted=%s\n', i, ...
          num (iris.numeric_at ('Petal.Length', i)), num (iris.numeric_at ('Petal.Width', i)), ...
          actual, predicted);
end
printf ('%d of %d misclassified (%s%%)\n', misclassified, n, num (100.0 * misclassified / n));

model.plot_classification (iris, 'Petal.Length', 'Petal.Width').save ('forest_iris_classification.svg');
model.plot_decision_regions ('Petal.Length', 'Petal.Width').save ('forest_iris_decision_regions.svg');
printf ('\nSaved forest_iris_classification.svg and forest_iris_decision_regions.svg\n');

% A small forest, for comparison, showing how out-of-bag accuracy stabilizes as more trees
% are added -- the defining random forest effect a single decision tree cannot demonstrate.
% datamunge.RandomForestClassifier (data, formula, n_trees, max_depth, min_samples_split,
%   min_samples_leaf, max_features, criterion, bootstrap, sample_fraction, seed)
small_forest = datamunge.RandomForestClassifier (iris, 'Species ~ Petal.Length + Petal.Width', 5);
printf ('\n5-tree forest:   training accuracy=%s%%  OOB accuracy=%s%%\n', ...
        num (small_forest.training_accuracy () * 100), num (small_forest.oob_accuracy () * 100));
printf ('100-tree forest: training accuracy=%s%%  OOB accuracy=%s%%\n', ...
        num (model.training_accuracy () * 100), num (model.oob_accuracy () * 100));
small_forest.plot_decision_regions ('Petal.Length', 'Petal.Width').save ('forest_iris_decision_regions_5trees.svg');
printf ('Saved forest_iris_decision_regions_5trees.svg\n');
