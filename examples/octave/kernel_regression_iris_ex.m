% CMD: just TARGET=kernel_regression_iris_ex run-octave
% Nadaraya-Watson kernel regression on iris: CV-selected bandwidth vs. too-small and too-large ones.
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

formula = 'Petal.Length ~ Petal.Width';

iris = datamunge.Data.iris ();
printf ('iris: %d rows x %d cols\n', iris.nrows (), iris.ncols ());
printf ('formula: %s\n\n', formula);

model = datamunge.KernelRegression (iris, formula);  % bandwidth auto-selected via leave-one-out CV
model.print_summary ();

model.plot_fit (iris).save ('kernel_regression_iris_fit.svg');
model.plot_cv_curve ().save ('kernel_regression_iris_cv.svg');
printf ('\nSaved kernel_regression_iris_fit.svg and kernel_regression_iris_cv.svg\n');

% A too-small and a too-large bandwidth, for comparison -- the classic kernel regression
% bias-variance story. datamunge.KernelRegression (data, formula, kernel, bandwidth, n_bandwidth, standardize)
small = datamunge.KernelRegression (iris, formula, 'gaussian', 0.05);
printf ('\nbandwidth=0.05 (too small): LOO R-squared=%s  LOO RMSE=%s\n', num (small.r_squared ()), num (small.rmse ()));
small.plot_fit (iris).save ('kernel_regression_iris_fit_small_bandwidth.svg');

large = datamunge.KernelRegression (iris, formula, 'gaussian', 5.0);
printf ('bandwidth=5.0 (too large):  LOO R-squared=%s  LOO RMSE=%s\n', num (large.r_squared ()), num (large.rmse ()));
large.plot_fit (iris).save ('kernel_regression_iris_fit_large_bandwidth.svg');

printf ('bandwidth=%s (CV-selected): LOO R-squared=%s  LOO RMSE=%s\n', num (model.bandwidth ()), num (model.r_squared ()), num (model.rmse ()));
printf ('\nSaved kernel_regression_iris_fit_small_bandwidth.svg and kernel_regression_iris_fit_large_bandwidth.svg\n');
