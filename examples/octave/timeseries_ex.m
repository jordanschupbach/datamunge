1;

datamunge;

function v = dv(t)
  datamunge;
  v = DVector();
  for i = 1:numel(t)
    DVector_push_back(v, t(i));
  end
endfunction

function s = cell_join(c)
  parts = {};
  for i = 1:numel(c)
    parts{end + 1} = num2str(c{i});
  end
  s = strjoin(parts, ", ");
endfunction

printf("=================== ARIMA(1,1,1) on a simulated random walk with drift ===================\n");
rand("seed", 42);
randn("seed", 42);
y = [];
level = 100.0;
prev_shock = 0.0;
for i = 1:150
  shock = randn();
  level = level + 0.3 + 0.4 * prev_shock + shock;  % integrated ARMA(0,1) increments with drift
  y(end + 1) = level;
  prev_shock = shock;
end

options = ARIMAOptions();
ARIMAOptions_p_set(options, 1);
ARIMAOptions_d_set(options, 1);
ARIMAOptions_q_set(options, 1);
ARIMAOptions_de_population_size_set(options, 80);
ARIMAOptions_de_max_generations_set(options, 400);
model = ARIMA(dv(y), options);

ar = ARIMA_ar_coefficients(model);
ma = ARIMA_ma_coefficients(model);
printf("AR coefficient: %g\n", ar{1});
printf("MA coefficient: %g\n", ma{1});
printf("sigma^2: %g, AIC: %g, BIC: %g\n", ARIMA_sigma2(model), ARIMA_aic(model), ARIMA_bic(model));

pair = ARIMA_forecast_with_intervals(model, 6);
printf("6-step forecast: %s\n", cell_join(pair{1}));
printf("forecast std. errors: %s\n", cell_join(pair{2}));

printf("\n=================== SARIMA(1,0,0)(1,1,0)_12 on a seasonal series ===================\n");
rand("seed", 7);
randn("seed", 7);
s = [];
prev = 0.0;
for i = 0:119
  prev = 0.5 * prev + randn();
  s(end + 1) = 20.0 + 0.2 * i + 5.0 * sin(2.0 * pi * i / 12.0) + prev;
end

options2 = ARIMAOptions();
ARIMAOptions_p_set(options2, 1);
ARIMAOptions_seasonal_p_set(options2, 1);
ARIMAOptions_seasonal_d_set(options2, 1);
ARIMAOptions_seasonal_period_set(options2, 12);
ARIMAOptions_de_population_size_set(options2, 100);
ARIMAOptions_de_max_generations_set(options2, 500);
model2 = ARIMA(dv(s), options2);
ar2 = ARIMA_ar_coefficients(model2);
sar2 = ARIMA_seasonal_ar_coefficients(model2);
printf("AR coefficient: %g, seasonal AR coefficient: %g\n", ar2{1}, sar2{1});
printf("12-step forecast: %s\n", cell_join(ARIMA_forecast(model2, 12)));

printf("\n=================== Exponential smoothing: Holt-Winters on retail-style seasonal data ===================\n");
seasonal_shape = [0.8, 0.75, 0.9, 0.95, 1.0, 1.05, 1.1, 1.05, 1.0, 1.1, 1.3, 1.6];
rand("seed", 11);
randn("seed", 11);
y2 = [];
for i = 0:47
  lvl = 100.0 + 2.0 * i;
  y2(end + 1) = lvl * seasonal_shape(mod(i, 12) + 1) + randn() * 3.0;
end

es_options = ExponentialSmoothingOptions();
ExponentialSmoothingOptions_trend_set(es_options, TrendType_Additive);
ExponentialSmoothingOptions_seasonal_set(es_options, SeasonalType_Multiplicative);
ExponentialSmoothingOptions_seasonal_period_set(es_options, 12);
ExponentialSmoothingOptions_de_population_size_set(es_options, 60);
ExponentialSmoothingOptions_de_max_generations_set(es_options, 300);
es_model = ExponentialSmoothing(dv(y2), es_options);
printf("alpha=%g beta=%g gamma=%g\n", ExponentialSmoothing_alpha(es_model), ExponentialSmoothing_beta(es_model), ExponentialSmoothing_gamma(es_model));
printf("sigma^2: %g, AIC: %g\n", ExponentialSmoothing_sigma2(es_model), ExponentialSmoothing_aic(es_model));
printf("12-month forecast: %s\n", cell_join(ExponentialSmoothing_forecast(es_model, 12)));

printf("\n=================== Simple exponential smoothing vs. Holt's linear trend ===================\n");
flat = [50.2, 49.8, 50.5, 49.6, 50.1, 50.3, 49.9, 50.0, 50.4, 49.7];

ses = ExponentialSmoothing(dv(flat), ExponentialSmoothingOptions());
printf("SES alpha= %g\n", ExponentialSmoothing_alpha(ses));
printf("SES 5-step forecast: %s\n", cell_join(ExponentialSmoothing_forecast(ses, 5)));

holt_options = ExponentialSmoothingOptions();
ExponentialSmoothingOptions_trend_set(holt_options, TrendType_Additive);
holt = ExponentialSmoothing(dv(flat), holt_options);
printf("Holt alpha=%g beta=%g\n", ExponentialSmoothing_alpha(holt), ExponentialSmoothing_beta(holt));
printf("Holt 5-step forecast: %s\n", cell_join(ExponentialSmoothing_forecast(holt, 5)));
