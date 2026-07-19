local dm = require("datamunge")

local function dv(t)
  local v = dm.DVector(#t)
  for i, x in ipairs(t) do v[i - 1] = x end
  return v
end

local function gauss()
  local u1, u2 = math.random(), math.random()
  return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
end

print("=================== Random intercept on a real dataset (penguins) ===================")
local penguins = dm.DataFrame.penguins()
local species_model = dm.LMM(penguins, "body_mass_g ~ flipper_length_mm + bill_length_mm + (1 | species)")
species_model:print_summary()

print("\n=================== Random intercept + slope on a simulated multi-school dataset ===================")
-- 30 schools, ~25 students each: score depends on study_hours, but both the baseline score
-- and the return on an extra hour of study vary by school -- the textbook case for a
-- random-intercept-and-slope model.
math.randomseed(2024)
local n_schools = 30
local school_intercept, school_slope = {}, {}
for s = 0, n_schools - 1 do
  school_intercept[s] = gauss() * 6.0
  school_slope[s] = gauss() * 1.2
end

local true_intercept = 60.0
local true_slope = 3.0
local school, study_hours, score = {}, {}, {}
for s = 0, n_schools - 1 do
  local n_students = 15 + math.random(0, 20)
  for _ = 1, n_students do
    local hours = math.random() * 10.0
    local noise = gauss() * 4.0
    local s_val = true_intercept + school_intercept[s] + (true_slope + school_slope[s]) * hours + noise
    table.insert(school, s * 1.0)
    table.insert(study_hours, hours)
    table.insert(score, s_val)
  end
end

local df = dm.DataFrame()
df:add_numeric_column("school", dv(school))
df:add_numeric_column("study_hours", dv(study_hours))
df:add_numeric_column("score", dv(score))

local model = dm.LMM(df, "score ~ study_hours + (1 + study_hours | school)")
model:print_summary()

print("\nTrue generating values: intercept=" .. true_intercept .. ", slope=" .. true_slope ..
      ", random-intercept SD=6.0, random-slope SD=1.2, residual SD=4.0")

print("\n--- Best Linear Unbiased Predictors (BLUPs) for a few schools ---")
local group_labels = model:group_labels()
for _, idx in ipairs({0, 1, 2}) do
  local re = model:random_effects_for_group(idx)
  print("school " .. group_labels[idx] .. ": intercept shift=" .. re[0] .. ", slope shift=" .. re[1])
end

print("\n--- Prediction: population-level vs. school-adjusted ---")
local newdata_population = dm.DataFrame()
newdata_population:add_numeric_column("study_hours", dv({5.0}))
local newdata_school0 = dm.DataFrame()
newdata_school0:add_numeric_column("study_hours", dv({5.0}))
newdata_school0:add_numeric_column("school", dv({0.0}))
print("5 study hours, unseen school:      " .. model:predict(newdata_population)[0] .. " (fixed effects only)")
print("5 study hours, school 0 (known):    " .. model:predict(newdata_school0)[0] .. " (fixed effects + school 0's BLUP)")
