const datamunge = require("../../index.js");

function dvector(values) {
  const out = new datamunge.DVector(values.length);
  for (let i = 0; i < values.length; i++) out.set(i, values[i]);
  return out;
}

function randn() {
  const u1 = Math.random();
  const u2 = Math.random();
  return Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
}

function toArray(v) {
  const out = [];
  for (let i = 0; i < v.size(); i++) out.push(v.get(i));
  return out;
}

console.log("=================== ARIMA(1,1,1) on a simulated random walk with drift ===================");
const y = [];
let level = 100.0;
let prevShock = 0.0;
for (let i = 0; i < 150; i++) {
  const shock = randn();
  level = level + 0.3 + 0.4 * prevShock + shock;
  y.push(level);
  prevShock = shock;
}

const options = new datamunge.ARIMAOptions();
options.p = 1;
options.d = 1;
options.q = 1;
options.de_population_size = 80;
options.de_max_generations = 400;
const model = new datamunge.ARIMA(dvector(y), options);

const ar = model.ar_coefficients();
const ma = model.ma_coefficients();
console.log(`AR coefficient: ${ar.get(0)}`);
console.log(`MA coefficient: ${ma.get(0)}`);
console.log(`sigma^2: ${model.sigma2()}, AIC: ${model.aic()}, BIC: ${model.bic()}`);

const pair = model.forecast_with_intervals(6);
console.log(`6-step forecast: ${toArray(pair.first).join(", ")}`);
console.log(`forecast std. errors: ${toArray(pair.second).join(", ")}`);

console.log("\n=================== SARIMA(1,0,0)(1,1,0)_12 on a seasonal series ===================");
const s = [];
let prev = 0.0;
for (let i = 0; i < 120; i++) {
  prev = 0.5 * prev + randn();
  s.push(20.0 + 0.2 * i + 5.0 * Math.sin((2.0 * Math.PI * i) / 12.0) + prev);
}

const options2 = new datamunge.ARIMAOptions();
options2.p = 1;
options2.seasonal_p = 1;
options2.seasonal_d = 1;
options2.seasonal_period = 12;
options2.de_population_size = 100;
options2.de_max_generations = 500;
const model2 = new datamunge.ARIMA(dvector(s), options2);
const ar2 = model2.ar_coefficients();
const sar2 = model2.seasonal_ar_coefficients();
console.log(`AR coefficient: ${ar2.get(0)}, seasonal AR coefficient: ${sar2.get(0)}`);
console.log(`12-step forecast: ${toArray(model2.forecast(12)).join(", ")}`);

console.log("\n=================== Exponential smoothing: Holt-Winters on retail-style seasonal data ===================");
const seasonalShape = [0.8, 0.75, 0.9, 0.95, 1.0, 1.05, 1.1, 1.05, 1.0, 1.1, 1.3, 1.6];
const y2 = [];
for (let i = 0; i < 48; i++) {
  const lvl = 100.0 + 2.0 * i;
  y2.push(lvl * seasonalShape[i % 12] + randn() * 3.0);
}

const esOptions = new datamunge.ExponentialSmoothingOptions();
esOptions.trend = datamunge.TrendType_Additive;
esOptions.seasonal = datamunge.SeasonalType_Multiplicative;
esOptions.seasonal_period = 12;
esOptions.de_population_size = 60;
esOptions.de_max_generations = 300;
const esModel = new datamunge.ExponentialSmoothing(dvector(y2), esOptions);
console.log(`alpha=${esModel.alpha()} beta=${esModel.beta()} gamma=${esModel.gamma()}`);
console.log(`sigma^2: ${esModel.sigma2()}, AIC: ${esModel.aic()}`);
console.log(`12-month forecast: ${toArray(esModel.forecast(12)).join(", ")}`);

console.log("\n=================== Simple exponential smoothing vs. Holt's linear trend ===================");
const flat = [50.2, 49.8, 50.5, 49.6, 50.1, 50.3, 49.9, 50.0, 50.4, 49.7];

const ses = new datamunge.ExponentialSmoothing(dvector(flat), new datamunge.ExponentialSmoothingOptions());
console.log(`SES alpha= ${ses.alpha()}`);
console.log(`SES 5-step forecast: ${toArray(ses.forecast(5)).join(", ")}`);

const holtOptions = new datamunge.ExponentialSmoothingOptions();
holtOptions.trend = datamunge.TrendType_Additive;
const holt = new datamunge.ExponentialSmoothing(dvector(flat), holtOptions);
console.log(`Holt alpha=${holt.alpha()} beta=${holt.beta()}`);
console.log(`Holt 5-step forecast: ${toArray(holt.forecast(5)).join(", ")}`);
