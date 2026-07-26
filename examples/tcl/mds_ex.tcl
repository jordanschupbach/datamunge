package require Datamunge 0.0.1

# See pca_ex.tcl / clustering_ex.tcl: pass the feature columns as a \x1f-joined encoded string so
# the overloaded constructor resolves unambiguously.
proc encode_cols {names} {
  return [join $names [format %c 31]]
}

set iris [datamunge::DataFrame_iris]
set features [encode_cols {Sepal.Length Sepal.Width Petal.Length Petal.Width}]

set n [datamunge::DataFrame_nrows $iris]
set species {}
for {set i 0} {$i < $n} {incr i} {
  lappend species [datamunge::DataFrame_string_at $iris "Species" $i]
}

puts "=================== Classical MDS on iris (euclidean) ==================="
set mds [datamunge::new_MDS $iris $features 2 "euclidean"]
datamunge::MDS_print_summary $mds

puts "\nEmbedding as a DataFrame:"
puts [datamunge::DataFrame_to_string [datamunge::MDS_embedding_frame $mds]]

set scatter [datamunge::MDS_plot_embedding_grouped $mds $species]
datamunge::Plot_save_svg $scatter "mds_iris_embedding.svg"
puts "\nEmbedding scatter (colored by species) saved as mds_iris_embedding.svg"

puts "\n=================== Classical MDS on iris (manhattan) ==================="
set manhattan_mds [datamunge::new_MDS $iris $features 2 "manhattan"]
puts [format "Goodness of fit: euclidean=%.4f, manhattan=%.4f" \
  [datamunge::MDS_goodness_of_fit $mds] [datamunge::MDS_goodness_of_fit $manhattan_mds]]
