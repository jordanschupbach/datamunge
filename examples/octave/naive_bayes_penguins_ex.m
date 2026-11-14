% CMD: just TARGET=naive_bayes_penguins_ex run-octave
% Naive Bayes on penguins: mixed numeric/categorical predictors, confusion matrix, misclassified rows, plots.
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

% Note: NaiveBayesClassifier drops incomplete rows for its own formula columns internally
% when fitting, matching every other formula-based model in this library.
penguins = datamunge.Data.penguins ();
printf ('penguins: %d rows x %d cols\n\n', penguins.nrows (), penguins.ncols ());

% A mix of numeric (Gaussian likelihood) and categorical (frequency-table likelihood)
% predictors in one formula -- each modeled independently given the class, per the naive
% Bayes assumption.
model = datamunge.NaiveBayesClassifier (penguins, 'species ~ bill_length_mm + bill_depth_mm + island + sex');
model.print_summary ();

printf ('\nConfusion matrix (rows = actual, cols = predicted):\n%s\n', model.confusion_matrix ().to_string ());

printf ('\nMisclassified rows:\n');
predictions = model.predict (penguins);
misclassified = 0;
columns = {'species', 'bill_length_mm', 'bill_depth_mm', 'island', 'sex'};
for i = 0:penguins.nrows ()-1
  if any (cellfun (@(c) penguins.is_null (c, i), columns))
    continue;
  end
  actual = penguins.string_at ('species', i);
  predicted = predictions{i+1};
  if strcmp (predicted, actual)
    continue;
  end
  misclassified++;
  printf ('  row %d: bill_length=%s bill_depth=%s island=%s sex=%s  actual=%s  predicted=%s\n', i, ...
          num (penguins.numeric_at ('bill_length_mm', i)), num (penguins.numeric_at ('bill_depth_mm', i)), ...
          penguins.string_at ('island', i), penguins.string_at ('sex', i), actual, predicted);
end
printf ('%d misclassified (of %d rows, some incomplete)\n', misclassified, penguins.nrows ());

% plot_decision_regions requires exactly two NUMERIC predictors, so build a separate
% two-predictor model (bill measurements alone) just for visualization.
bill_only = datamunge.NaiveBayesClassifier (penguins, 'species ~ bill_length_mm + bill_depth_mm');
printf ('\nbill-measurements-only model training accuracy: %s%%\n', num (100 * bill_only.training_accuracy ()));
bill_only.plot_classification (penguins, 'bill_length_mm', 'bill_depth_mm').save ('naive_bayes_penguins_classification.svg');
bill_only.plot_decision_regions ('bill_length_mm', 'bill_depth_mm').save ('naive_bayes_penguins_decision_regions.svg');
printf ('Saved naive_bayes_penguins_classification.svg and naive_bayes_penguins_decision_regions.svg\n');
