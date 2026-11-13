// CMD: just TARGET=hypothesis_tests_ex run-javascript

const datamunge = require("../../index.js");

function printResult(label, r) {
  let line = `${label}: statistic=${r.statistic}`;
  if (r.parameter1 !== 0.0) line += `, df1=${r.parameter1}`;
  if (r.parameter2 !== 0.0) line += `, df2=${r.parameter2}`;
  const altNames = ["two.sided", "less", "greater"];
  line += `, p=${r.p_value} (${altNames[r.alternative]})`;
  if (r.has_conf_int) line += `, CI=[${r.conf_int_lower}, ${r.conf_int_upper}]`;
  line += ` -- ${r.method}`;
  console.log(line);
}

function columnForSpecies(df, column, species) {
  const out = [];
  const n = df.nrows();
  for (let i = 0; i < n; i++) {
    if (df.string_at("Species", i) === species) out.push(df.numeric_at(column, i));
  }
  return out;
}

const iris = datamunge.data.iris();

const setosaPetal = columnForSpecies(iris, "Petal.Length", "setosa");
const versicolorPetal = columnForSpecies(iris, "Petal.Length", "versicolor");
const virginicaPetal = columnForSpecies(iris, "Petal.Length", "virginica");

console.log("=================== t-tests: petal length, setosa vs. versicolor ===================");
printResult("Welch two-sample t-test", datamunge.t_test_two_sample(setosaPetal, versicolorPetal));
printResult("Wilcoxon rank-sum test", datamunge.wilcoxon_rank_sum_test(setosaPetal, versicolorPetal));

console.log("\n=================== One-way ANOVA / Kruskal-Wallis across all 3 species ===================");
const allPetal = setosaPetal.concat(versicolorPetal, virginicaPetal);
const sizes = [setosaPetal.length, versicolorPetal.length, virginicaPetal.length];
printResult("One-way ANOVA", datamunge.one_way_anova(allPetal, sizes));
printResult("Kruskal-Wallis", datamunge.kruskal_wallis_test(allPetal, sizes));

console.log("\n=================== Correlation: sepal length vs. petal length ===================");
const n = iris.nrows();
const sepalLength = [];
const petalLength = [];
for (let i = 0; i < n; i++) {
  sepalLength.push(iris.numeric_at("Sepal.Length", i));
  petalLength.push(iris.numeric_at("Petal.Length", i));
}
printResult("Pearson correlation", datamunge.pearson_correlation_test(sepalLength, petalLength));
printResult("Spearman correlation", datamunge.spearman_correlation_test(sepalLength, petalLength));

console.log("\n=================== F-test: petal length variance, setosa vs. virginica ===================");
printResult("F test", datamunge.f_test_variance(setosaPetal, virginicaPetal));

console.log('\n=================== Normality: is sepal length normally distributed within setosa? ===================');
const setosaSepal = columnForSpecies(iris, "Sepal.Length", "setosa");
printResult("Shapiro-Francia", datamunge.shapiro_francia_test(setosaSepal));
printResult("KS vs. fitted normal", datamunge.ks_test_one_sample_normal(setosaSepal, 5.006, 0.3525));

console.log('\n=================== Chi-squared / Fisher: is petal length "long" independent of species? ===================');
const sortedAll = [...allPetal].sort((a, b) => a - b);
const medianAll = sortedAll[Math.floor(allPetal.length / 2)];
const setosaLong = setosaPetal.filter((x) => x > medianAll).length;
const setosaShort = setosaPetal.length - setosaLong;
const versicolorLong = versicolorPetal.filter((x) => x > medianAll).length;
const versicolorShort = versicolorPetal.length - versicolorLong;
console.log(`table: setosa=[${setosaLong},${setosaShort}] versicolor=[${versicolorLong},${versicolorShort}]`);
const tableVec = [setosaLong, setosaShort, versicolorLong, versicolorShort];
printResult("Chi-squared independence", datamunge.chi_squared_test_independence(tableVec, 2, 2));
printResult("Fisher's exact test", datamunge.fisher_exact_test_2x2(setosaLong, setosaShort, versicolorLong, versicolorShort));

console.log('\n=================== Proportion / binomial: fraction of "long" petals overall ===================');
const longCount = allPetal.filter((x) => x > medianAll).length;
printResult("One-sample proportion test (vs 0.5)", datamunge.proportion_test_one_sample(longCount, allPetal.length, 0.5));
printResult("Exact binomial test (vs 0.5)", datamunge.binomial_test(longCount, allPetal.length, 0.5));
