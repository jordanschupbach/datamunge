local dm = require("datamunge")

local function dv(t)
  local v = dm.DVector(#t)
  for i, x in ipairs(t) do v[i - 1] = x end
  return v
end

local ALT_NAMES = {
  [dm.Alternative_Less] = "less",
  [dm.Alternative_Greater] = "greater",
  [dm.Alternative_TwoSided] = "two.sided",
}

local function print_result(label, r)
  local line = label .. ": statistic=" .. r.statistic
  if r.parameter1 ~= 0.0 then line = line .. ", df1=" .. r.parameter1 end
  if r.parameter2 ~= 0.0 then line = line .. ", df2=" .. r.parameter2 end
  line = line .. ", p=" .. r.p_value .. " (" .. ALT_NAMES[r.alternative] .. ")"
  if r.has_conf_int then line = line .. ", CI=[" .. r.conf_int_lower .. ", " .. r.conf_int_upper .. "]" end
  line = line .. " -- " .. r.method
  print(line)
end

local function column_for_species(df, column, species)
  local out = {}
  for i = 0, df:nrows() - 1 do
    if df:string_at("Species", i) == species then table.insert(out, df:numeric_at(column, i)) end
  end
  return out
end

local iris = dm.DataFrame.iris()

local setosa_petal = column_for_species(iris, "Petal.Length", "setosa")
local versicolor_petal = column_for_species(iris, "Petal.Length", "versicolor")
local virginica_petal = column_for_species(iris, "Petal.Length", "virginica")

print("=================== t-tests: petal length, setosa vs. versicolor ===================")
print_result("Welch two-sample t-test", dm.t_test_two_sample(dv(setosa_petal), dv(versicolor_petal)))
print_result("Wilcoxon rank-sum test", dm.wilcoxon_rank_sum_test(dv(setosa_petal), dv(versicolor_petal)))

print("\n=================== One-way ANOVA / Kruskal-Wallis across all 3 species ===================")
local all_petal = {}
for _, v in ipairs(setosa_petal) do table.insert(all_petal, v) end
for _, v in ipairs(versicolor_petal) do table.insert(all_petal, v) end
for _, v in ipairs(virginica_petal) do table.insert(all_petal, v) end
local sizes = dm.SizeVector(3)
sizes[0] = #setosa_petal
sizes[1] = #versicolor_petal
sizes[2] = #virginica_petal
print_result("One-way ANOVA", dm.one_way_anova(dv(all_petal), sizes))
print_result("Kruskal-Wallis", dm.kruskal_wallis_test(dv(all_petal), sizes))

print("\n=================== Correlation: sepal length vs. petal length ===================")
local sepal_length, petal_length = {}, {}
for i = 0, iris:nrows() - 1 do
  table.insert(sepal_length, iris:numeric_at("Sepal.Length", i))
  table.insert(petal_length, iris:numeric_at("Petal.Length", i))
end
print_result("Pearson correlation", dm.pearson_correlation_test(dv(sepal_length), dv(petal_length)))
print_result("Spearman correlation", dm.spearman_correlation_test(dv(sepal_length), dv(petal_length)))

print("\n=================== F-test: petal length variance, setosa vs. virginica ===================")
print_result("F test", dm.f_test_variance(dv(setosa_petal), dv(virginica_petal)))

print("\n=================== Normality: is sepal length normally distributed within setosa? ===================")
local setosa_sepal = column_for_species(iris, "Sepal.Length", "setosa")
print_result("Shapiro-Francia", dm.shapiro_francia_test(dv(setosa_sepal)))
print_result("KS vs. fitted normal", dm.ks_test_one_sample_normal(dv(setosa_sepal), 5.006, 0.3525))

print("\n=================== Chi-squared / Fisher: is petal length \"long\" independent of species? ===================")
local sorted_all = {}
for _, v in ipairs(all_petal) do table.insert(sorted_all, v) end
table.sort(sorted_all)
local median_all = sorted_all[math.floor(#all_petal / 2) + 1]
local function count_long(t) local n = 0; for _, v in ipairs(t) do if v > median_all then n = n + 1 end end; return n end
local setosa_long = count_long(setosa_petal)
local setosa_short = #setosa_petal - setosa_long
local versicolor_long = count_long(versicolor_petal)
local versicolor_short = #versicolor_petal - versicolor_long
print("table: setosa=[" .. setosa_long .. "," .. setosa_short .. "] versicolor=[" .. versicolor_long .. "," .. versicolor_short .. "]")
local table_vec = dv({setosa_long, setosa_short, versicolor_long, versicolor_short})
print_result("Chi-squared independence", dm.chi_squared_test_independence(table_vec, 2, 2))
print_result("Fisher's exact test", dm.fisher_exact_test_2x2(setosa_long, setosa_short, versicolor_long, versicolor_short))

print("\n=================== Proportion / binomial: fraction of \"long\" petals overall ===================")
local long_count = count_long(all_petal)
print_result("One-sample proportion test (vs 0.5)", dm.proportion_test_one_sample(long_count, #all_petal, 0.5))
print_result("Exact binomial test (vs 0.5)", dm.binomial_test(long_count, #all_petal, 0.5))
