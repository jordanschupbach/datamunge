# CMD: just TARGET=clustering_ex run-ruby
# Clustering iris: k-means, agglomerative (several linkages, dendrogram re-cut) and DBSCAN.
require "datamunge"

iris = Datamunge::Data.iris
species = Array.new(iris.nrows) { |i| iris.string_at("Species", i) }
feature_columns = %w[Sepal.Length Sepal.Width Petal.Length Petal.Width]

# For each cluster, print its size and its majority species (negative labels are noise).
report_purity = lambda do |cluster_labels, n_clusters|
  votes = Array.new(n_clusters) { Hash.new(0) }
  cluster_labels.zip(species).each { |label, sp| votes[label][sp] += 1 if label >= 0 }
  votes.each_with_index do |counts, c|
    total = counts.values.sum
    next if total.zero?
    best_species, best = counts.max_by { |_, count| count }
    puts "  cluster #{c}: #{total} points, majority #{best_species} (#{best}/#{total})"
  end
end

puts "=================== K-means (k=3) on iris ==================="
kmeans = Datamunge::KMeans.new(iris, feature_columns, 3)
puts "Inertia: #{kmeans.inertia}, iterations (best run): #{kmeans.iterations_used}"
report_purity.(kmeans.labels, 3)

puts "\n=================== Agglomerative clustering on iris ==================="
%w[ward complete average].each do |linkage|
  model = Datamunge::AgglomerativeClustering.new(iris, feature_columns, 3, linkage)
  puts "--- linkage=#{linkage} ---"
  report_purity.(model.labels, 3)
end

puts "\n--- Re-cutting the ward dendrogram at k=2 without refitting ---"
ward_model = Datamunge::AgglomerativeClustering.new(iris, feature_columns, 3, "ward")
report_purity.(ward_model.cut(2), 2)

puts "\n=================== DBSCAN on iris ==================="
dbscan = Datamunge::DBSCAN.new(iris, feature_columns, 0.6, 5)
puts "Clusters found: #{dbscan.n_clusters}, noise points: #{dbscan.n_noise} (of #{dbscan.observations})"
report_purity.(dbscan.labels, dbscan.n_clusters)
