(* CMD: just TARGET=pca_ex run-ocaml *)
(* PCA on iris: summary, PC1 loadings, scores frame, score/scree plots, scaled vs unscaled. *)
open Datamunge

let () =
  let iris = Data.iris () in
  let feature_columns = [ "Sepal.Length"; "Sepal.Width"; "Petal.Length"; "Petal.Width" ] in

  (* species labels, for the grouped (colored-by-species) scores plot *)
  let species = List.init (DataFrame.nrows iris) (DataFrame.string_at iris "Species") in

  print_endline "=================== PCA on iris (scaled) ===================";
  let pca = PCA.create ~center:true ~scale:true iris feature_columns in
  PCA.print_summary pca;

  print_endline "\nPC1 loadings (which original features drive it):";
  List.iteri
    (fun i name -> Printf.printf "  %s: %.4f\n" name (PCA.component_loadings pca 0).(i))
    (PCA.feature_names pca);

  Printf.printf "\nTraining scores as a DataFrame:\n%s\n"
    (DataFrame.to_string (PCA.scores_frame pca));

  RPlot.save_svg (PCA.plot_scores_grouped pca species) "pca_iris_scores.svg";
  print_endline "\nScores scatter (colored by species) saved as pca_iris_scores.svg";

  RPlot.save_svg (PCA.plot_scree pca) "pca_iris_scree.svg";
  print_endline "Scree plot saved as pca_iris_scree.svg";

  print_endline "\n=================== PCA on iris (unscaled) ===================";
  let unscaled = PCA.create ~center:true ~scale:false iris feature_columns in
  Printf.printf
    "PC1 explains %.2f%% of variance (vs. %.2f%% scaled) -- Sepal.Length's larger raw variance \
     dominates the unscaled covariance matrix.\n"
    (100. *. (PCA.explained_variance_ratio unscaled).(0))
    (100. *. (PCA.explained_variance_ratio pca).(0))
