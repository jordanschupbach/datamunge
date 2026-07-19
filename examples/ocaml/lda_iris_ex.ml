(* This binding uses SWIG-OCaml's untyped Swig.c_obj_t dynamic-dispatch protocol:
   - Datamunge._function_name args         -- free functions and static factories
   - new_ClassName (C_list [args])         -- constructors
   - invoke obj "method_name" args         -- instance methods (args is C_void for 0
                                               args, or C_list [a1; a2; ...] for more)
   - invoke obj "[field]" args             -- struct field get/set
   - invoke v "[]" (C_int i)               -- container element access
   - std::size_t-typed parameters need C_int64, not C_int (int/index params stay C_int). *)
open Swig
open Datamunge

let dv values =
  let v = new_DVector C_void in
  List.iter (fun x -> ignore (invoke v "push_back" (C_double x))) values;
  v

let () =
  let iris = _DataFrame_iris C_void in
  Printf.printf "iris: %d rows x %d cols\n\n"
    (get_int (invoke iris "nrows" C_void)) (get_int (invoke iris "ncols" C_void));

  let model = new_LDA (C_list [iris; C_string "Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width"]) in
  ignore (invoke model "print_summary" C_void);

  print_endline "\nConfusion matrix (rows = actual, cols = predicted):";
  print_endline (get_string (invoke (invoke model "confusion_matrix" C_void) "to_string" C_void));

  let newdata = new_DataFrame C_void in
  ignore (invoke newdata "add_numeric_column" (C_list [C_string "Sepal.Length"; dv [5.1; 6.0; 6.5; 6.2]]));
  ignore (invoke newdata "add_numeric_column" (C_list [C_string "Sepal.Width"; dv [3.5; 2.7; 3.0; 2.8]]));
  ignore (invoke newdata "add_numeric_column" (C_list [C_string "Petal.Length"; dv [1.4; 4.5; 5.5; 4.8]]));
  ignore (invoke newdata "add_numeric_column" (C_list [C_string "Petal.Width"; dv [0.2; 1.5; 2.0; 1.8]]));

  print_endline "\nPredictions for new flowers:";
  print_endline (get_string (invoke (invoke model "predict_frame" newdata) "to_string" C_void));

  ignore (invoke model "save_discriminant_plot" (C_string "lda_iris_discriminants.svg"));
  print_endline "\nSaved discriminant plot as lda_iris_discriminants.svg"
