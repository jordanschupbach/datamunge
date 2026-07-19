local dm = require("datamunge")

local function sv(t)
  local v = dm.SVector(#t)
  for i, x in ipairs(t) do v[i - 1] = x end
  return v
end

local function report_purity(cluster_labels, species, n_clusters)
  local votes = {}
  for c = 0, n_clusters - 1 do votes[c] = {} end
  for i = 0, cluster_labels:size() - 1 do
    local label = cluster_labels[i]
    if label >= 0 then
      votes[label][species[i]] = (votes[label][species[i]] or 0) + 1
    end
  end
  for c = 0, n_clusters - 1 do
    local total = 0
    for _, n in pairs(votes[c]) do total = total + n end
    if total > 0 then
      local best_species, best = nil, -1
      for sp, n in pairs(votes[c]) do
        if n > best then best_species, best = sp, n end
      end
      print("  cluster " .. c .. ": " .. total .. " points, majority " .. best_species .. " (" .. best .. "/" .. total .. ")")
    end
  end
end

local iris = dm.DataFrame.iris()
local species = {}
for i = 0, iris:nrows() - 1 do species[i] = iris:string_at("Species", i) end
local features = sv({"Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"})

print("=================== K-means (k=3) on iris ===================")
local kmeans = dm.KMeans(iris, features, 3)
print("Inertia: " .. kmeans:inertia() .. ", iterations (best run): " .. kmeans:iterations_used())
report_purity(kmeans:labels(), species, 3)

print("\n=================== Agglomerative clustering on iris ===================")
for _, linkage in ipairs({"ward", "complete", "average"}) do
  local model = dm.AgglomerativeClustering(iris, features, 3, linkage)
  print("--- linkage=" .. linkage .. " ---")
  report_purity(model:labels(), species, 3)
end

print("\n--- Re-cutting the ward dendrogram at k=2 without refitting ---")
local ward_model = dm.AgglomerativeClustering(iris, features, 3, "ward")
report_purity(ward_model:cut(2), species, 2)

print("\n=================== DBSCAN on iris ===================")
local dbscan = dm.DBSCAN(iris, features, 0.6, 5)
print("Clusters found: " .. dbscan:n_clusters() .. ", noise points: " .. dbscan:n_noise() .. " (of " .. dbscan:observations() .. ")")
report_purity(dbscan:labels(), species, dbscan:n_clusters())
