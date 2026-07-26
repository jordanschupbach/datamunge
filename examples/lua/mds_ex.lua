local dm = require("datamunge")

local function sv(t)
  local v = dm.SVector(#t)
  for i, x in ipairs(t) do v[i - 1] = x end
  return v
end

local iris = dm.DataFrame.iris()
local features = sv({ "Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width" })

local species_t = {}
for i = 0, iris:nrows() - 1 do species_t[i + 1] = iris:string_at("Species", i) end
local species = sv(species_t)

print("=================== Classical MDS on iris (euclidean) ===================")
local mds = dm.MDS(iris, features, 2, "euclidean")
mds:print_summary()

print("\nEmbedding as a DataFrame:")
print(mds:embedding_frame():to_string())

local scatter = mds:plot_embedding_grouped(species)
scatter:save_svg("mds_iris_embedding.svg")
print("\nEmbedding scatter (colored by species) saved as mds_iris_embedding.svg")

print("\n=================== Classical MDS on iris (manhattan) ===================")
local manhattan_mds = dm.MDS(iris, features, 2, "manhattan")
print(string.format("Goodness of fit: euclidean=%.4f, manhattan=%.4f",
  mds:goodness_of_fit(), manhattan_mds:goodness_of_fit()))
