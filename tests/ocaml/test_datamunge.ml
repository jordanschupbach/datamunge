(* Smoke tests of the datamunge OCaml binding (just test-ocaml), compiled natively against the findlib
   package datamunge (build/ocaml/prefix/lib on OCAMLPATH). Argument: the generated interface (default
   src/datamungeocaml/ocaml/datamunge/datamunge.mli), for the documentation checks. *)
open Datamunge

let checks = ref 0

let check ok message =
  incr checks;
  if not ok then (
    prerr_endline ("check failed: " ^ message);
    exit 1)

let read_file path = In_channel.with_open_bin path In_channel.input_all

(* The index of [sub] in [text] at or after [from] (searching backwards from it when [back]). *)
let find ?(back = false) text sub from =
  let n = String.length sub in
  let rec go i =
    if i < 0 || i + n > String.length text then raise Not_found
    else if String.sub text i n = sub then i
    else go (if back then i - 1 else i + 1)
  in
  go from

let contains text sub = match find text sub 0 with _ -> true | exception Not_found -> false

(* The odoc comment right before [decl] (searched from [from]), or "" when there is none. *)
let doc_before text ~from decl =
  match find text decl from with
  | exception Not_found -> ""
  | at -> (
      match find ~back:true text "(**" at with
      | exception Not_found -> ""
      | start -> String.sub text start (at - start))

let () =
  (* Smoke test: call into the native library. *)
  hello ();

  (* std::vector<double> is a float array, std::pair<double, double> a tuple. *)
  let v = make_dvector 1.0 2.0 3.0 in
  check (v = [| 1.0; 2.0; 3.0 |]) "std::vector<double> is a float array";
  check (sum_dvector v = 6.0) "sum_dvector";
  check (sum_dvector [| 0.5; 1.5 |] = 2.0) "arrays convert to std::vector";
  let p = make_dpair 1.25 2.75 in
  check (p = (1.25, 2.75)) "std::pair is a tuple";
  check (sum_dpair p = 4.0) "sum_dpair";

  (* The C++ Callback class, and an OCaml subclass overriding its virtual call() (a director). *)
  let cb = Callback.create () in
  check (call_with_callback 3.0 cb = 3.0) "native Callback";
  check (sum_dvector (map_dvector_with_callback [| 1.; 2.; 3. |] cb) = 6.0) "native map";
  let doubler = Callback.extend ~call:(fun _ x -> 2. *. x) () in
  check (call_with_callback 3.0 doubler = 6.0) "the OCaml override runs";
  check
    (sum_dvector (map_dvector_with_callback [| 1.; 2.; 3. |] doubler) = 12.0)
    "override from C++ loops";

  (* An OCaml exception inside an override reaches the caller unchanged. *)
  let failing = Callback.extend ~call:(fun _ _ -> failwith "from OCaml") () in
  check
    (match call_with_callback 1.0 failing with
     | _ -> false
     | exception Failure m -> m = "from OCaml")
    "the OCaml exception propagates";

  (* C++ exceptions raise OCaml exceptions: std::invalid_argument is Invalid_argument. *)
  check
    (match runif 1.0 0.0 3 with _ -> false | exception Invalid_argument _ -> true)
    "std::invalid_argument is Invalid_argument";

  (* Classes are submodules; close releases the C++ object. *)
  let frame = DataFrame.create () in
  DataFrame.add_numeric_column frame "x" [| 1.0; 2.0; 3.0 |];
  check (DataFrame.nrows frame = 3) "DataFrame";
  check (DataFrame.columns frame = [ "x" ]) "std::vector<std::string> is a string list";
  DataFrame.close frame;
  check
    (match DataFrame.nrows frame with _ -> false | exception Error (`Binding, _) -> true)
    "a closed object refuses calls";

  (* The committed interface carries the C++ documentation as odoc comments. *)
  let path =
    if Array.length Sys.argv > 1 then Sys.argv.(1)
    else "src/datamungeocaml/ocaml/datamunge/datamunge.mli"
  in
  let mli = read_file path in
  check
    (contains (doc_before mli ~from:0 "\nval seql :") "evenly spaced")
    "function odoc comment";
  let frame_at = find mli "module DataFrame : sig" 0 in
  check
    (contains (doc_before mli ~from:frame_at "val sort_by :") "@param column_name")
    "method odoc comment";

  Printf.printf "test_datamunge.ml: all %d checks passed\n" !checks
