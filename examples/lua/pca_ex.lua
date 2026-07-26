local dm = require("datamunge")

local function sv(t)
  local v = dm.SVector(#t)
  for i, x in ipairs(t) do v[i - 1] = x end
  return v
end

local iris = dm.DataFrame.iris()
local features = sv({ "Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width" })

-- species labels as an SVector, for the grouped (colored-by-species) scores plot
local species_t = {}
for i = 0, iris:nrows() - 1 do species_t[i + 1] = iris:string_at("Species", i) end
local species = sv(species_t)

print("=================== PCA on iris (scaled) ===================")
local pca = dm.PCA(iris, features, true, true)
pca:print_summary()

print("\nPC1 loadings (which original features drive it):")
local names = pca:feature_names()
local loadings = pca:component_loadings(0)
for i = 0, names:size() - 1 do
  print(string.format("  %s: %.4f", names[i], loadings[i]))
end

print("\nTraining scores as a DataFrame:")
print(pca:scores_frame():to_string())

local scatter = pca:plot_scores_grouped(species)
scatter:save_svg("pca_iris_scores.svg")
print("\nScores scatter (colored by species) saved as pca_iris_scores.svg")

local scree = pca:plot_scree()
scree:save_svg("pca_iris_scree.svg")
print("Scree plot saved as pca_iris_scree.svg")

print("\n=================== PCA on iris (unscaled) ===================")
local unscaled = dm.PCA(iris, features, true, false)
local scaled_ratio = pca:explained_variance_ratio()
local unscaled_ratio = unscaled:explained_variance_ratio()
print(string.format(
  "PC1 explains %.2f%% of variance (vs. %.2f%% scaled) -- Sepal.Length's larger raw variance "
    .. "dominates the unscaled covariance matrix.",
  unscaled_ratio[0] * 100, scaled_ratio[0] * 100))
