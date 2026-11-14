(* CMD: just TARGET=dataframe_ex run-ocaml *)
(* DataFrame basics: nulls, drop_duplicates, fill_null, select/sort, group-by, join and summaries. *)
open Datamunge

let () =
  let sales = DataFrame.create () in
  DataFrame.add_string_column sales "region" [ "west"; "west"; "east"; "south"; "south"; "south" ];
  DataFrame.add_string_column sales "product"
    [ "widget"; "widget"; "widget"; "gizmo"; "gizmo"; "gizmo" ];
  DataFrame.add_numeric_column sales "sales" [| 10.0; 10.0; 14.0; 8.0; 0.0; 11.0 |]
    ~valid_mask:[ 1; 1; 1; 1; 0; 1 ];
  DataFrame.add_string_column sales "quarter" [ "Q1"; "Q1"; "Q1"; "Q2"; "Q2"; "" ]
    ~valid_mask:[ 1; 1; 1; 1; 1; 0 ];

  Printf.printf "raw data\n%s\n\n" (DataFrame.to_string sales);

  let cleaned = DataFrame.drop_duplicates sales ~subset:[ "region"; "product"; "sales"; "quarter" ] in
  DataFrame.fill_null_string cleaned "quarter" "unknown";
  DataFrame.fill_null_numeric cleaned "sales" 0.0;
  Printf.printf "after drop_duplicates + fill_null\n%s\n\n" (DataFrame.to_string cleaned);

  (* The object is each function's first positional argument: pipe into a function of it. *)
  let selected =
    DataFrame.select cleaned [ "region"; "sales"; "quarter" ]
    |> fun df -> DataFrame.sort_by ~ascending:false df "sales"
  in
  Printf.printf "selected + sorted\n%s\n\n" (DataFrame.to_string selected);

  let grouped =
    DataFrame.group_by_sum cleaned [ "region" ] [ "sales" ]
    |> fun df -> DataFrame.sort_by ~ascending:false df "sales"
  in
  Printf.printf "group_by_sum(region)\n%s\n\n" (DataFrame.to_string grouped);

  let targets = DataFrame.create () in
  DataFrame.add_string_column targets "region" [ "west"; "east"; "south" ];
  DataFrame.add_numeric_column targets "target" [| 18.0; 12.0; 25.0 |];
  let joined = DataFrame.join ~join_type:"left" grouped targets "region" "region" in
  Printf.printf "joined with targets\n%s\n\n" (DataFrame.to_string joined);

  (match DataFrame.shape cleaned with
  | [ rows; cols ] -> Printf.printf "shape = (%d, %d)\n" rows cols
  | _ -> ());
  Printf.printf "sales count = %d\n" (DataFrame.numeric_count cleaned "sales");
  Printf.printf "sales nulls = %d\n" (DataFrame.numeric_null_count cleaned "sales");
  Printf.printf "sales sum = %g\n" (DataFrame.numeric_sum cleaned "sales");
  Printf.printf "sales mean = %g\n" (DataFrame.numeric_mean cleaned "sales")
