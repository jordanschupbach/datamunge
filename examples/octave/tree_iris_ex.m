% CMD: just TARGET=tree_iris_ex run-octave
% Decision tree classifier on iris: summary, confusion matrix, misclassified rows and plots.

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

model = datamunge.DecisionTreeClassifier (iris, 'Species ~ Petal.Length + Petal.Width');
model.print_summary ();

printf ('\nConfusion matrix (rows = actual, cols = predicted):\n');
printf ('%s\n', model.confusion_matrix ().to_string ());

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

model.plot_classification (iris, 'Petal.Length', 'Petal.Width').save ('tree_iris_classification.svg');
model.plot_decision_regions ('Petal.Length', 'Petal.Width').save ('tree_iris_decision_regions.svg');
printf ('\nSaved tree_iris_classification.svg and tree_iris_decision_regions.svg\n');

% A shallower tree, for comparison, showing a coarser (but still fairly accurate) decision boundary.
shallow = datamunge.DecisionTreeClassifier (iris, 'Species ~ Petal.Length + Petal.Width', 2);
printf ('\nDepth-2 tree training accuracy: %s%% (%d leaves)\n', num (shallow.training_accuracy () * 100), ...
        shallow.leaf_count ());
shallow.plot_decision_regions ('Petal.Length', 'Petal.Width').save ('tree_iris_decision_regions_depth2.svg');
printf ('Saved tree_iris_decision_regions_depth2.svg\n');
