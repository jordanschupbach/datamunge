(* CMD: just TARGET=clustering_ex run-ocaml *)
(* Clustering iris: k-means, agglomerative (several linkages, dendrogram re-cut) and DBSCAN. *)

open Datamunge

let iris = Data.iris ()
let species = List.init (DataFrame.nrows iris) (DataFrame.string_at iris "Species")
let feature_columns = [ "Sepal.Length"; "Sepal.Width"; "Petal.Length"; "Petal.Width" ]

(* For each cluster, print its size and its majority species (negative labels are noise). *)
let report_purity cluster_labels n_clusters =
  let labelled = List.combine cluster_labels species in
  for c = 0 to n_clusters - 1 do
    let members = List.filter_map (fun (l, sp) -> if l = c then Some sp else None) labelled in
    (* Species counts in order of first appearance; ties keep the first species seen. *)
    let counts =
      List.fold_left
        (fun acc sp ->
          if List.mem_assoc sp acc then
            List.map (fun (s, n) -> if s = sp then (s, n + 1) else (s, n)) acc
          else acc @ [ (sp, 1) ])
        [] members
    in
    match counts with
    | [] -> ()
    | first :: rest ->
        let best_species, best =
          List.fold_left (fun (bs, b) (s, n) -> if n > b then (s, n) else (bs, b)) first rest
        in
        let total = List.length members in
        Printf.printf "  cluster %d: %d points, majority %s (%d/%d)\n" c total best_species best
          total
  done

let () =
  print_endline "=================== K-means (k=3) on iris ===================";
  let kmeans = KMeans.create ~n_clusters:3 iris feature_columns in
  Printf.printf "Inertia: %g, iterations (best run): %d\n" (KMeans.inertia kmeans)
    (KMeans.iterations_used kmeans);
  report_purity (KMeans.labels kmeans) 3;

  print_endline "\n=================== Agglomerative clustering on iris ===================";
  List.iter
    (fun linkage ->
      let model = AgglomerativeClustering.create ~n_clusters:3 ~linkage iris feature_columns in
      Printf.printf "--- linkage=%s ---\n" linkage;
      report_purity (AgglomerativeClustering.labels model) 3;
      AgglomerativeClustering.close model)
    [ "ward"; "complete"; "average" ];

  print_endline "\n--- Re-cutting the ward dendrogram at k=2 without refitting ---";
  let ward_model =
    AgglomerativeClustering.create ~n_clusters:3 ~linkage:"ward" iris feature_columns
  in
  report_purity (AgglomerativeClustering.cut ward_model 2) 2;

  print_endline "\n=================== DBSCAN on iris ===================";
  let dbscan = DBSCAN.create ~eps:0.6 ~min_samples:5 iris feature_columns in
  Printf.printf "Clusters found: %d, noise points: %d (of %d)\n" (DBSCAN.n_clusters dbscan)
    (DBSCAN.n_noise dbscan) (DBSCAN.observations dbscan);
  report_purity (DBSCAN.labels dbscan) (DBSCAN.n_clusters dbscan)
