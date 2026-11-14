(* CMD: just TARGET=gis_ex run-ocaml *)
(* ShapeLayer: write a tiny synthetic shapefile (.shp + .dbf) by hand, read it back and draw it as a map. *)
open Datamunge

(* Real .shp/.dbf files are large binary bundles that don't belong in this repo, so this example
   writes two "counties" (one plain square, one with a lake-shaped hole) using the same ESRI byte
   layout ShapeLayer.read expects, then reads them back through the public API. *)

(* A shape is a list of rings; a ring is a list of (x, y) points. *)
type shape = (float * float) list list

(* Bounding box (minx, miny, maxx, maxy) of every point in the given shapes. *)
let bounds (shapes : shape list) =
  List.fold_left
    (fun (x0, y0, x1, y1) (x, y) ->
      (Float.min x0 x, Float.min y0 y, Float.max x1 x, Float.max y1 y))
    (infinity, infinity, neg_infinity, neg_infinity)
    (List.concat (List.concat shapes))

let add_int_le buf i = Buffer.add_int32_le buf (Int32.of_int i)
let add_int_be buf i = Buffer.add_int32_be buf (Int32.of_int i)
let add_double buf x = Buffer.add_int64_le buf (Int64.bits_of_float x)

let add_bounds buf (x0, y0, x1, y1) = List.iter (add_double buf) [ x0; y0; x1; y1 ]

(* Bytes of one Polygon record's contents (without the record header). *)
let polygon_record (rings : shape) =
  let buf = Buffer.create 256 in
  (* Shape type 5 = Polygon, then bounds, part/point counts, part starts and the points. *)
  add_int_le buf 5;
  add_bounds buf (bounds [ rings ]);
  add_int_le buf (List.length rings);
  add_int_le buf (List.length (List.concat rings));
  ignore
    (List.fold_left
       (fun start ring ->
         add_int_le buf start;
         start + List.length ring)
       0 rings);
  List.iter
    (fun (x, y) ->
      add_double buf x;
      add_double buf y)
    (List.concat rings);
  Buffer.contents buf

let write_file path contents =
  Out_channel.with_open_bin path (fun oc -> Out_channel.output_string oc contents)

let write_counties_shp path shapes =
  let contents = List.map polygon_record shapes in
  let total_words =
    List.fold_left (fun acc c -> acc + 4 + (String.length c / 2)) 0 contents
  in
  let buf = Buffer.create 1024 in
  (* Big-endian file code and length. *)
  List.iter (add_int_be buf) [ 9994; 0; 0; 0; 0; 0; 50 + total_words ];
  add_int_le buf 1000; (* version *)
  add_int_le buf 5; (* Polygon *)
  add_bounds buf (bounds shapes);
  add_bounds buf (0.0, 0.0, 0.0, 0.0);
  List.iteri
    (fun i c ->
      add_int_be buf (i + 1);
      add_int_be buf (String.length c / 2);
      Buffer.add_string buf c)
    contents;
  write_file path (Buffer.contents buf)

(* One 32-byte dBASE field descriptor. *)
let dbf_field buf name kind length =
  Buffer.add_string buf (name ^ String.make (11 - String.length name) '\000');
  Buffer.add_char buf kind;
  add_int_le buf 0;
  Buffer.add_uint8 buf length;
  Buffer.add_string buf (String.make 15 '\000')

let write_counties_dbf path names populations =
  let header_size = 32 + (2 * 32) + 1 in
  let record_size = 1 + 12 + 8 in
  let buf = Buffer.create 256 in
  Buffer.add_string buf "\003\000\000\000";
  add_int_le buf (List.length names);
  Buffer.add_uint16_le buf header_size;
  Buffer.add_uint16_le buf record_size;
  Buffer.add_string buf (String.make 20 '\000');
  dbf_field buf "NAME" 'C' 12;
  dbf_field buf "POP" 'N' 8;
  Buffer.add_char buf '\r';
  let field width s = Printf.sprintf "%-*s" width (String.sub s 0 (min width (String.length s))) in
  List.iter2
    (fun name pop -> Buffer.add_string buf (" " ^ field 12 name ^ field 8 pop))
    names populations;
  write_file path (Buffer.contents buf)

(* Roughly C's %g: integral doubles print without a fractional part. *)
let fmt_g x = if x = Float.round x then Printf.sprintf "%.0f" x else Printf.sprintf "%.6f" x

let () =
  let base_path = "datamunge_gis_ex_counties_ocaml" in

  (* Per the ESRI winding convention, outer rings are clockwise, holes counterclockwise. *)
  let plain = [ [ (10., 0.); (10., 10.); (20., 10.); (20., 0.); (10., 0.) ] ] in
  let with_lake =
    [
      [ (0., 0.); (0., 10.); (10., 10.); (10., 0.); (0., 0.) ];
      [ (3., 3.); (4., 3.); (4., 4.); (3., 4.); (3., 3.) ];
    ]
  in
  write_counties_shp (base_path ^ ".shp") [ with_lake; plain ];
  write_counties_dbf (base_path ^ ".dbf") [ "Lakeside"; "Plainview" ] [ "48231"; "19876" ];

  let counties = ShapeLayer.read base_path in

  Printf.printf "shapes = %d, shape_type = %s\n" (ShapeLayer.size counties)
    (ShapeLayer.shape_type counties);
  let bounds = ShapeLayer.bounds counties |> Array.to_list |> List.map fmt_g in
  Printf.printf "bounds = [%s]\n\n" (String.concat ", " bounds);

  let attrs = ShapeLayer.attributes counties in
  Printf.printf "attributes\n%s\n\n" (DataFrame.to_string attrs);

  for i = 0 to ShapeLayer.size counties - 1 do
    Printf.printf "%s: %d ring(s)\n" (DataFrame.string_at attrs "NAME" i)
      (ShapeLayer.num_parts counties i)
  done;

  let svg_path = "datamunge_gis_ex_map_ocaml.svg" in
  Plot.save_svg (ShapeLayer.plot counties) svg_path;
  Printf.printf "\nmap saved to %s\n" svg_path
