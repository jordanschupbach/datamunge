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

console.log("=================== Random intercept on a real dataset (penguins) ===================");
const penguins = datamunge.DataFrame.penguins();
const speciesModel = new datamunge.LMM(penguins, "body_mass_g ~ flipper_length_mm + bill_length_mm + (1 | species)");
speciesModel.print_summary();

console.log("\n=================== Random intercept + slope on a simulated multi-school dataset ===================");
const nSchools = 30;
const schoolIntercept = Array.from({ length: nSchools }, () => randn() * 6.0);
const schoolSlope = Array.from({ length: nSchools }, () => randn() * 1.2);

const trueIntercept = 60.0;
const trueSlope = 3.0;
const school = [];
const studyHours = [];
const score = [];
for (let s = 0; s < nSchools; s++) {
  const nStudents = 15 + Math.floor(Math.random() * 21);
  for (let j = 0; j < nStudents; j++) {
    const hours = Math.random() * 10.0;
    const noise = randn() * 4.0;
    const sVal = trueIntercept + schoolIntercept[s] + (trueSlope + schoolSlope[s]) * hours + noise;
    school.push(s);
    studyHours.push(hours);
    score.push(sVal);
  }
}

const df = new datamunge.DataFrame();
df.add_numeric_column("school", dvector(school));
df.add_numeric_column("study_hours", dvector(studyHours));
df.add_numeric_column("score", dvector(score));

const model = new datamunge.LMM(df, "score ~ study_hours + (1 + study_hours | school)");
model.print_summary();

console.log(`\nTrue generating values: intercept=${trueIntercept}, slope=${trueSlope}, random-intercept SD=6.0, random-slope SD=1.2, residual SD=4.0`);

console.log("\n--- Best Linear Unbiased Predictors (BLUPs) for a few schools ---");
const groupLabels = model.group_labels();
for (let idx = 0; idx < 3; idx++) {
  const re = model.random_effects_for_group(idx);
  console.log(`school ${groupLabels.get(idx)}: intercept shift=${re.get(0)}, slope shift=${re.get(1)}`);
}

console.log("\n--- Prediction: population-level vs. school-adjusted ---");
const newdataPopulation = new datamunge.DataFrame();
newdataPopulation.add_numeric_column("study_hours", dvector([5.0]));
const newdataSchool0 = new datamunge.DataFrame();
newdataSchool0.add_numeric_column("study_hours", dvector([5.0]));
newdataSchool0.add_numeric_column("school", dvector([0.0]));
const predPop = model.predict(newdataPopulation);
const predS0 = model.predict(newdataSchool0);
console.log(`5 study hours, unseen school:      ${predPop.get(0)} (fixed effects only)`);
console.log(`5 study hours, school 0 (known):    ${predS0.get(0)} (fixed effects + school 0's BLUP)`);
