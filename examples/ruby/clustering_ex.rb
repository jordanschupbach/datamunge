require "octruby"

def report_purity(cluster_labels, species, n_clusters)
  votes = Array.new(n_clusters) { Hash.new(0) }
  cluster_labels.each_with_index do |label, i|
    next if label < 0
    votes[label][species[i]] += 1
  end
  n_clusters.times do |c|
    total = votes[c].values.sum
    next if total == 0
    best_species, best = votes[c].max_by { |_, v| v }
    puts "  cluster #{c}: #{total} points, majority #{best_species} (#{best}/#{total})"
  end
end

iris = Datamunge::DataFrame.iris
species = (0...iris.nrows).map { |i| iris.string_at("Species", i) }
features = ["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"]

puts "=================== K-means (k=3) on iris ==================="
kmeans = Datamunge::KMeans.new(iris, features, 3)
puts "Inertia: #{kmeans.inertia}, iterations (best run): #{kmeans.iterations_used}"
report_purity(kmeans.labels, species, 3)

puts "\n=================== Agglomerative clustering on iris ==================="
["ward", "complete", "average"].each do |linkage|
  model = Datamunge::AgglomerativeClustering.new(iris, features, 3, linkage)
  puts "--- linkage=#{linkage} ---"
  report_purity(model.labels, species, 3)
end

puts "\n--- Re-cutting the ward dendrogram at k=2 without refitting ---"
ward_model = Datamunge::AgglomerativeClustering.new(iris, features, 3, "ward")
report_purity(ward_model.cut(2), species, 2)

puts "\n=================== DBSCAN on iris ==================="
dbscan = Datamunge::DBSCAN.new(iris, features, 0.6, 5)
puts "Clusters found: #{dbscan.n_clusters}, noise points: #{dbscan.n_noise} (of #{dbscan.observations})"
report_purity(dbscan.labels, species, dbscan.n_clusters)
