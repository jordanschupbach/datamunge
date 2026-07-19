local dm = require("datamunge")

local function dv(t)
  local v = dm.DVector(#t)
  for i, x in ipairs(t) do v[i - 1] = x end
  return v
end

local function sv(t)
  local v = dm.SVector(#t)
  for i, x in ipairs(t) do v[i - 1] = x end
  return v
end

local function gauss()
  local u1, u2 = math.random(), math.random()
  return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
end

print("=================== Binomial (logistic) mixed model on a real dataset (penguins) ===================")
-- Predicting sex from body mass, grouped by island: unlike species (which is almost perfectly
-- confounded with island in this dataset -- Gentoo penguins occur on Biscoe island only), sex
-- is not tied to location, so this is a well-behaved fit.
local penguins = dm.DataFrame.penguins()
local is_male, body_mass, island = {}, {}, {}
for i = 0, penguins:nrows() - 1 do
  if not (penguins:is_null("sex", i) or penguins:is_null("body_mass_g", i) or penguins:is_null("island", i)) then
    table.insert(is_male, penguins:string_at("sex", i) == "male" and 1.0 or 0.0)
    table.insert(body_mass, penguins:numeric_at("body_mass_g", i))
    table.insert(island, penguins:string_at("island", i))
  end
end

local sex_df = dm.DataFrame()
sex_df:add_numeric_column("is_male", dv(is_male))
sex_df:add_numeric_column("body_mass_g", dv(body_mass))
sex_df:add_string_column("island", sv(island))

local sex_model = dm.GLMM(sex_df, "is_male ~ body_mass_g + (1 | island)", "binomial")
sex_model:print_summary()

print("\n=================== Poisson mixed model on simulated multi-site count data ===================")
-- 25 stores, ~20 days each: daily visit counts depend on a promo intensity score, but both
-- the baseline traffic and the promo's effectiveness vary by store.
math.randomseed(4242)
local n_stores = 25
local store_effect = {}
for s = 0, n_stores - 1 do store_effect[s] = gauss() * 0.4 end

local true_intercept = 2.0
local true_slope = 0.3
local store, promo, visits = {}, {}, {}
for s = 0, n_stores - 1 do
  local n_days = 15 + math.random(0, 10)
  for _ = 1, n_days do
    local promo_intensity = math.random() * 3.0
    local lam = math.exp(true_intercept + store_effect[s] + true_slope * promo_intensity)
    -- Knuth's Poisson sampler.
    local l_thresh = math.exp(-lam)
    local k = 0
    local p = 1.0
    while true do
      k = k + 1
      p = p * math.random()
      if p <= l_thresh then break end
    end
    table.insert(store, s * 1.0)
    table.insert(promo, promo_intensity)
    table.insert(visits, (k - 1) * 1.0)
  end
end

local df = dm.DataFrame()
df:add_numeric_column("store", dv(store))
df:add_numeric_column("promo", dv(promo))
df:add_numeric_column("visits", dv(visits))

local store_model = dm.GLMM(df, "visits ~ promo + (1 | store)", "poisson")
store_model:print_summary()

print("\nTrue generating values: intercept=" .. true_intercept .. ", slope=" .. true_slope ..
      ", random-intercept SD (log scale)=0.4")

print("\n--- BLUPs for a few stores ---")
local group_labels = store_model:group_labels()
for _, idx in ipairs({0, 1, 2}) do
  print("store " .. group_labels[idx] .. ": intercept shift=" .. store_model:random_effects_for_group(idx)[0])
end

print("\n--- Prediction: population-level vs. store-adjusted ---")
local newdata_population = dm.DataFrame()
newdata_population:add_numeric_column("promo", dv({1.5}))
local newdata_store0 = dm.DataFrame()
newdata_store0:add_numeric_column("promo", dv({1.5}))
newdata_store0:add_numeric_column("store", dv({0.0}))
print("promo=1.5, unseen store:   " .. store_model:predict(newdata_population)[0] .. " expected visits (fixed effects only)")
print("promo=1.5, store 0 (known): " .. store_model:predict(newdata_store0)[0] .. " expected visits (fixed effects + store 0's BLUP)")
