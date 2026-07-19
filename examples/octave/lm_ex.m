1;

datamunge;

function v = dv(t)
  datamunge;
  v = DVector();
  for i = 1:numel(t)
    DVector_push_back(v, t(i));
  end
endfunction

function v = sv(t)
  datamunge;
  v = SVector();
  for i = 1:numel(t)
    SVector_push_back(v, t{i});
  end
endfunction

hp = [110.0, 110.0, 93.0, 110.0, 175.0, 105.0, 245.0, 62.0, 95.0, 123.0];
wt = [2.62, 2.875, 2.32, 3.215, 3.44, 3.46, 3.57, 3.19, 3.15, 3.44];
transmission = {"manual", "manual", "manual", "automatic", "automatic", ...
                "automatic", "automatic", "automatic", "automatic", "automatic"};
mpg = [21.0, 21.0, 22.8, 21.4, 18.7, 18.1, 14.3, 24.4, 22.8, 19.2];

cars = DataFrame_empty();
DataFrame_add_numeric_column(cars, "hp", dv(hp));
DataFrame_add_numeric_column(cars, "wt", dv(wt));
DataFrame_add_string_column(cars, "transmission", sv(transmission));
DataFrame_add_numeric_column(cars, "mpg", dv(mpg));

printf("Fitting: mpg ~ hp + wt + transmission\n\n");
model = LM(cars, "mpg ~ hp + wt + transmission");
LM_print_summary(model);

printf("\nSequential ANOVA:\n");
printf("%s\n", DataFrame_to_string(LM_anova(model)));

newcars = DataFrame_empty();
DataFrame_add_numeric_column(newcars, "hp", dv([150.0, 90.0]));
DataFrame_add_numeric_column(newcars, "wt", dv([3.0, 2.5]));
DataFrame_add_string_column(newcars, "transmission", sv({"manual", "automatic"}));

frame = LM_predict_frame(model, newcars, "confidence");
printf("\nPredictions with 95%% confidence intervals:\n");
printf("%s\n", DataFrame_to_string(frame));

LM_save_diagnostic_plots(model, "lm_ex_diagnostics");
printf("\nSaved diagnostic plots as lm_ex_diagnostics_*.svg\n");
