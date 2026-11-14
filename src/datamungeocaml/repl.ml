(* Toplevel init for just repl-ocaml (utop -init, or `ocaml -init`): loads findlib's topfind and the
   datamunge package (build/ocaml/prefix/lib on OCAMLPATH), then opens it. Try
   [let df = Data.iris ()], [DataFrame.nrows df] or [seql 0. 1. 5]. *)
#use "topfind";;
#require "datamunge";;
open Datamunge;;
print_endline "datamunge loaded (open Datamunge): try `let df = Data.iris ();;` then `DataFrame.to_string ~max_rows:5 df |> print_string;;`";;
