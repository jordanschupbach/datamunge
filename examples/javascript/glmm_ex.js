const datamunge = require("../../index.js");

function dvector(values) {
  const out = new datamunge.DVector(values.length);
  for (let i = 0; i < values.length; i++) out.set(i, values[i]);
  return out;
}

function svector(values) {
  const out = new datamunge.SVector(values.length);
  for (let i = 0; i < values.length; i++) out.set(i, values[i]);
  return out;
}

function randn() {
  const u1 = Math.random();
  const u2 = Math.random();
  return Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
}

console.log("=================== Binomial (logistic) mixed model on a real dataset (penguins) ===================");
const penguins = datamunge.DataFrame.penguins();
const isMale = [];
const bodyMass = [];
const island = [];
const n = penguins.nrows();
for (let i = 0; i < n; i++) {
  if (!penguins.is_null("sex", i) && !penguins.is_null("body_mass_g", i) && !penguins.is_null("island", i)) {
    isMale.push(penguins.string_at("sex", i) === "male" ? 1.0 : 0.0);
    bodyMass.push(penguins.numeric_at("body_mass_g", i));
    island.push(penguins.string_at("island", i));
  }
}

const sexDf = new datamunge.DataFrame();
sexDf.add_numeric_column("is_male", dvector(isMale));
sexDf.add_numeric_column("body_mass_g", dvector(bodyMass));
sexDf.add_string_column("island", svector(island));

const sexModel = new datamunge.GLMM(sexDf, "is_male ~ body_mass_g + (1 | island)", "binomial");
sexModel.print_summary();

console.log("\n=================== Poisson mixed model on simulated multi-site count data ===================");
const nStores = 25;
const storeEffect = Array.from({ length: nStores }, () => randn() * 0.4);

const trueIntercept = 2.0;
const trueSlope = 0.3;
const store = [];
const promo = [];
const visits = [];
for (let s = 0; s < nStores; s++) {
  const nDays = 15 + Math.floor(Math.random() * 11);
  for (let d = 0; d < nDays; d++) {
    const promoIntensity = Math.random() * 3.0;
    const lam = Math.exp(trueIntercept + storeEffect[s] + trueSlope * promoIntensity);
    const lThresh = Math.exp(-lam);
    let k = 0;
    let p = 1.0;
    do {
      k += 1;
      p *= Math.random();
    } while (p > lThresh);
    store.push(s);
    promo.push(promoIntensity);
    visits.push(k - 1);
  }
}

const df = new datamunge.DataFrame();
df.add_numeric_column("store", dvector(store));
df.add_numeric_column("promo", dvector(promo));
df.add_numeric_column("visits", dvector(visits));

const storeModel = new datamunge.GLMM(df, "visits ~ promo + (1 | store)", "poisson");
storeModel.print_summary();

console.log(`\nTrue generating values: intercept=${trueIntercept}, slope=${trueSlope}, random-intercept SD (log scale)=0.4`);

console.log("\n--- BLUPs for a few stores ---");
const groupLabels = storeModel.group_labels();
for (let idx = 0; idx < 3; idx++) {
  const re = storeModel.random_effects_for_group(idx);
  console.log(`store ${groupLabels.get(idx)}: intercept shift=${re.get(0)}`);
}

console.log("\n--- Prediction: population-level vs. store-adjusted ---");
const newdataPopulation = new datamunge.DataFrame();
newdataPopulation.add_numeric_column("promo", dvector([1.5]));
const newdataStore0 = new datamunge.DataFrame();
newdataStore0.add_numeric_column("promo", dvector([1.5]));
newdataStore0.add_numeric_column("store", dvector([0.0]));
const predPop = storeModel.predict(newdataPopulation);
const predS0 = storeModel.predict(newdataStore0);
console.log(`promo=1.5, unseen store:   ${predPop.get(0)} expected visits (fixed effects only)`);
console.log(`promo=1.5, store 0 (known): ${predS0.get(0)} expected visits (fixed effects + store 0's BLUP)`);
