% CMD: just TARGET=glm_iris_ex run-octave
% Generalized linear models on iris: logistic regression (binomial) and Poisson regression.
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
n = iris.nrows ();
species = cell (1, n);
for i = 1:n
  species{i} = iris.string_at ('Species', i - 1);
end
column = @(name) arrayfun (@(i) iris.numeric_at (name, i), 0:n - 1);

% Logistic regression: versicolor vs virginica only -- setosa is perfectly separable from
% the other two on these predictors, which sends logistic regression's coefficients toward
% +/-infinity (a genuine degeneracy of the method, not a bug) -- so this is the well-behaved
% binary split for a demo.
kept = ismember (species, {'versicolor', 'virginica'});
is_virginica = double (strcmp (species(kept), 'virginica'));
all_petal_length = column ('Petal.Length');
all_petal_width = column ('Petal.Width');
petal_length = all_petal_length(kept);
petal_width = all_petal_width(kept);

sub = datamunge.DataFrame ();
sub.add_numeric_column ('Petal.Length', petal_length);
sub.add_numeric_column ('Petal.Width', petal_width);
sub.add_numeric_column ('is_virginica', is_virginica);

printf ('=================== Logistic regression (binomial, logit link) ===================\n');
logit = datamunge.GLM (sub, 'is_virginica ~ Petal.Length + Petal.Width', 'binomial');
logit.print_summary ();

fitted = logit.fitted_values ();
correct = sum ((fitted >= 0.5) == (is_virginica >= 0.5));
printf ('\nResubstitution accuracy at 0.5 threshold: %s%%\n', num (100 * correct / numel (is_virginica)));

logit.save_diagnostic_plots ('glm_logistic_iris');
printf ('\nSaved glm_logistic_iris_{residuals_vs_fitted,normal_qq,scale_location,residuals_vs_leverage}.svg\n');

% Predicted-probability curve across Petal.Length, with Petal.Width held at its mean -- the
% classic sigmoid shape of a fitted logistic regression, with a 95% confidence band.
grid_n = 100;
width_mean = mean (petal_width);
pl_min = min (petal_length) - 0.3;
pl_max = max (petal_length) + 0.3;
grid = datamunge.DataFrame ();
grid.add_numeric_column ('Petal.Length', pl_min + (pl_max - pl_min) * (0:grid_n - 1) / (grid_n - 1));
grid.add_numeric_column ('Petal.Width', repmat (width_mean, 1, grid_n));
curve_frame = logit.predict_frame (grid, 'confidence');
printf ('\nPredicted-probability curve (first 5 rows):\n');
printf ('%s\n', curve_frame.to_string (5));

% Poisson regression, for contrast: same IRLS engine, different family/link.
printf ('\n=================== Poisson regression (log link) ===================\n');
count_data = datamunge.DataFrame ();
count_data.add_numeric_column ('Sepal.Width', column ('Sepal.Width'));
count_data.add_numeric_column ('Petal.Length', all_petal_length);
count_data.add_numeric_column ('count', floor (column ('Sepal.Length') + 0.5));
datamunge.GLM (count_data, 'count ~ Sepal.Width + Petal.Length', 'poisson').print_summary ();
