open Swig
open Datamunge

let sv values =
  let v = new_SVector C_void in
  List.iter (fun x -> ignore (invoke v "push_back" (C_string x))) values;
  v

let report_purity cluster_labels species n_clusters =
  let votes = Array.make n_clusters (Hashtbl.create 8) in
  for c = 0 to n_clusters - 1 do votes.(c) <- Hashtbl.create 8 done;
  let n = get_int (invoke cluster_labels "size" C_void) in
  for i = 0 to n - 1 do
    let label = get_int (invoke cluster_labels "[]" (C_int i)) in
    if label >= 0 then begin
      let sp = species.(i) in
      let m = votes.(label) in
      let cur = try Hashtbl.find m sp with Not_found -> 0 in
      Hashtbl.replace m sp (cur + 1)
    end
  done;
  for c = 0 to n_clusters - 1 do
    let m = votes.(c) in
    let total = Hashtbl.fold (fun _ v acc -> acc + v) m 0 in
    if total > 0 then begin
      let best_sp = ref "" and best = ref (-1) in
      Hashtbl.iter (fun k v -> if v > !best then begin best := v; best_sp := k end) m;
      Printf.printf "  cluster %d: %d points, majority %s (%d/%d)\n" c total !best_sp !best total
    end
  done

let () =
  let iris = _DataFrame_iris C_void in
  let n = get_int (invoke iris "nrows" C_void) in
  let species = Array.init n (fun i -> get_string (invoke iris "string_at" (C_list [C_string "Species"; C_int64 (Int64.of_int i)]))) in
  let features = sv ["Sepal.Length"; "Sepal.Width"; "Petal.Length"; "Petal.Width"] in

  print_endline "=================== K-means (k=3) on iris ===================";
  let kmeans = new_KMeans (C_list [iris; features; C_int64 3L]) in
  Printf.printf "Inertia: %g, iterations (best run): %d\n" (get_float (invoke kmeans "inertia" C_void)) (get_int (invoke kmeans "iterations_used" C_void));
  report_purity (invoke kmeans "labels" C_void) species 3;

  print_endline "\n=================== Agglomerative clustering on iris ===================";
  List.iter (fun linkage ->
    let model = new_AgglomerativeClustering (C_list [iris; features; C_int64 3L; C_string linkage]) in
    Printf.printf "--- linkage=%s ---\n" linkage;
    report_purity (invoke model "labels" C_void) species 3
  ) ["ward"; "complete"; "average"];

  print_endline "\n--- Re-cutting the ward dendrogram at k=2 without refitting ---";
  let ward_model = new_AgglomerativeClustering (C_list [iris; features; C_int64 3L; C_string "ward"]) in
  report_purity (invoke ward_model "cut" (C_int64 2L)) species 2;

  print_endline "\n=================== DBSCAN on iris ===================";
  let dbscan = new_DBSCAN (C_list [iris; features; C_double 0.6; C_int64 5L]) in
  let n_clusters = get_int (invoke dbscan "n_clusters" C_void) in
  Printf.printf "Clusters found: %d, noise points: %d (of %d)\n" n_clusters (get_int (invoke dbscan "n_noise" C_void)) (get_int (invoke dbscan "observations" C_void));
  report_purity (invoke dbscan "labels" C_void) species n_clusters
