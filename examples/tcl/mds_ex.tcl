# CMD: just TARGET=mds_ex run-tcl
# Classical multidimensional scaling of iris: euclidean embedding, plot, and a manhattan comparison.
package require datamunge

set iris [datamunge::Data::iris]
set feature_columns {Sepal.Length Sepal.Width Petal.Length Petal.Width}
set species {}
for {set i 0} {$i < [$iris nrows]} {incr i} {
    lappend species [$iris string_at Species $i]
}

puts "=================== Classical MDS on iris (euclidean) ==================="
set mds [datamunge::MDS new $iris $feature_columns 2 euclidean]
$mds print_summary

puts "\nEmbedding as a DataFrame:\n[[$mds embedding_frame] to_string]"

[$mds plot_embedding_grouped $species] save_svg mds_iris_embedding.svg
puts "\nEmbedding scatter (colored by species) saved as mds_iris_embedding.svg"

puts "\n=================== Classical MDS on iris (manhattan) ==================="
set manhattan_mds [datamunge::MDS new $iris $feature_columns 2 manhattan]
puts [format "Goodness of fit: euclidean=%.4f, manhattan=%.4f" \
    [$mds goodness_of_fit] [$manhattan_mds goodness_of_fit]]
