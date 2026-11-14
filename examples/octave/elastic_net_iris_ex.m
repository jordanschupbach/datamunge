% CMD: just TARGET=elastic_net_iris_ex run-octave
% Ridge, Lasso and Elastic Net on iris, with coefficient-path and CV-curve figures.

formula = 'Petal.Length ~ Sepal.Length + Sepal.Width + Petal.Width + Sepal.Length:Petal.Width';

iris = datamunge.Data.iris ();
printf ('iris: %d rows x %d cols\n', iris.nrows (), iris.ncols ());
printf ('formula: %s\n\n', formula);

printf ('=================== Ridge ===================\n');
ridge = datamunge.Ridge (iris, formula);
ridge.print_summary ();

printf ('\n=================== Lasso ===================\n');
lasso = datamunge.Lasso (iris, formula);
lasso.print_summary ();

printf ('\n================= Elastic Net =================\n');
% datamunge.ElasticNet (data, formula, alpha=0.5, lambda, n_lambda, cv_folds, standardize, seed)
elastic = datamunge.ElasticNet (iris, formula, 0.5);
elastic.print_summary ();

printf (['\nSaved figures showing how each model''s coefficients respond to the regularization strength, ' ...
         'and the cross-validation curve used to pick it:\n']);

figures = {ridge.plot_coefficient_path(), ridge.plot_cv_curve(), 'ridge', 'ridge:      ';
           lasso.plot_coefficient_path(), lasso.plot_cv_curve(), 'lasso', 'lasso:      ';
           elastic.plot_coefficient_path(), elastic.plot_cv_curve(), 'elasticnet', 'elastic net:'};
for k = 1:rows (figures)
  [path_plot, cv_plot, name, label] = figures{k, :};
  path_plot.save (sprintf ('elastic_net_%s_path.svg', name));
  cv_plot.save (sprintf ('elastic_net_%s_cv.svg', name));
  printf ('  %s elastic_net_%s_path.svg, elastic_net_%s_cv.svg\n', label, name, name);
end
