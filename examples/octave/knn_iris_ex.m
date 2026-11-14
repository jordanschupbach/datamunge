% CMD: just TARGET=knn_iris_ex run-octave
% k-nearest-neighbours classifier on iris: leave-one-out accuracy and decision regions for several k.
1;

% Shortest round-trip decimal text for x (like C#/Julia default double printing).
function s = num (x)
  if x == fix (x) && abs (x) < 1e15
    s = sprintf ('%d', x);
    return;
  end
  for d = 1:17
    s = sprintf ('%.*e', d - 1, x);
    if str2double (s) == x
      break;
    end
  end
  epos = strfind (s, 'e');
  e = str2double (s(epos+1:end));
  if e >= -5 && e < 15
    s = sprintf ('%.*f', max (1, d - 1 - e), x);
  else
    s = sprintf ('%se%d', s(1:epos-1), e);
  end
end

iris = datamunge.Data.iris ();
printf ('iris: %d rows x %d cols\n\n', iris.nrows (), iris.ncols ());

model = datamunge.KNNClassifier (iris, 'Species ~ Petal.Length + Petal.Width');
model.print_summary ();

printf ('\nConfusion matrix (leave-one-out; rows = actual, cols = predicted):\n');
printf ('%s\n', model.confusion_matrix ().to_string ());

printf ('\nLeave-one-out misclassified rows:\n');
fitted = model.predict (iris);
n = iris.nrows ();
misclassified = 0;
for i = 0:n-1
  actual = iris.string_at ('Species', i);
  predicted = fitted{i+1};
  if strcmp (predicted, actual)
    continue;
  end
  misclassified++;
  printf ('  row %d: Petal.Length=%s Petal.Width=%s  actual=%s  predicted=%s\n', i, ...
          num (iris.numeric_at ('Petal.Length', i)), num (iris.numeric_at ('Petal.Width', i)), actual, predicted);
end
printf ('%d of %d misclassified (%s%%)\n', misclassified, n, num (100 * misclassified / n));

model.plot_decision_regions ('Petal.Length', 'Petal.Width').save ('knn_iris_decision_regions_k5.svg');
printf ('\nSaved knn_iris_decision_regions_k5.svg\n');

% k=1 memorizes every training point exactly (jagged, overfit boundary with an island around
% every point, including noise); k=25 averages over a much larger neighborhood (very smooth,
% underfit boundary). datamunge.KNNClassifier (data, formula, k, metric, weighted, standardize)
for k = [1 25]
  knn = datamunge.KNNClassifier (iris, 'Species ~ Petal.Length + Petal.Width', k);
  file = sprintf ('knn_iris_decision_regions_k%d.svg', k);
  printf ('\nk=%-2d leave-one-out accuracy: %s%%\n', k, num (knn.training_accuracy () * 100));
  knn.plot_decision_regions ('Petal.Length', 'Petal.Width').save (file);
  printf ('Saved %s\n', file);
end
