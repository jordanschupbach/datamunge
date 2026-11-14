(* CMD: just TARGET=mds_ex run-ocaml *)
(* Classical multidimensional scaling of iris: euclidean embedding, plot, and a manhattan comparison. *)
open Datamunge

let () =
  let iris = Data.iris () in
  let feature_columns = [ "Sepal.Length"; "Sepal.Width"; "Petal.Length"; "Petal.Width" ] in
  let species = List.init (DataFrame.nrows iris) (DataFrame.string_at iris "Species") in

  print_endline "=================== Classical MDS on iris (euclidean) ===================";
  let mds = MDS.create ~n_components:2 ~metric:"euclidean" iris feature_columns in
  MDS.print_summary mds;

  Printf.printf "\nEmbedding as a DataFrame:\n%s\n" (DataFrame.to_string (MDS.embedding_frame mds));

  RPlot.save_svg (MDS.plot_embedding_grouped mds species) "mds_iris_embedding.svg";
  print_endline "\nEmbedding scatter (colored by species) saved as mds_iris_embedding.svg";

  print_endline "\n=================== Classical MDS on iris (manhattan) ===================";
  let manhattan_mds = MDS.create ~n_components:2 ~metric:"manhattan" iris feature_columns in
  Printf.printf "Goodness of fit: euclidean=%.4f, manhattan=%.4f\n" (MDS.goodness_of_fit mds)
    (MDS.goodness_of_fit manhattan_mds)
