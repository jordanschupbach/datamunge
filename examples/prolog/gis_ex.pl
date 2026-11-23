% CMD: just TARGET=gis_ex run-prolog
% ShapeLayer: write a tiny synthetic shapefile (.shp + .dbf) by hand, read it back and draw it as a map.
% Real .shp/.dbf files are large binary bundles that don't belong in this repo, so this example
% writes two "counties" (one plain square, one with a lake-shaped hole) using the same ESRI byte
% layout shape_layer_read/2 expects, then reads them back through the public API.
:- use_module(library(datamunge)).
:- initialization(main, main).

base_path("datamunge_gis_ex_counties_prolog").

main :-
    base_path(BasePath),
    % Per the ESRI winding convention, outer rings are clockwise, holes counterclockwise.
    Plain = [[10.0-0.0, 10.0-10.0, 20.0-10.0, 20.0-0.0, 10.0-0.0]],
    WithLake = [ [0.0-0.0, 0.0-10.0, 10.0-10.0, 10.0-0.0, 0.0-0.0],
                 [3.0-3.0, 4.0-3.0, 4.0-4.0, 3.0-4.0, 3.0-3.0] ],

    format(string(ShpPath), "~w.shp", [BasePath]),
    format(string(DbfPath), "~w.dbf", [BasePath]),
    write_counties_shp(ShpPath, [WithLake, Plain]),
    write_counties_dbf(DbfPath, ["Lakeside", "Plainview"], ["48231", "19876"]),

    shape_layer_read(BasePath, Counties),

    shape_layer_size(Counties, Size), shape_layer_shape_type(Counties, ShapeType),
    format("shapes = ~w, shape_type = ~w~n", [Size, ShapeType]),
    shape_layer_bounds(Counties, Bounds),
    maplist(fmt_g, Bounds, BoundsText), atomic_list_concat(BoundsText, ', ', BoundsJoined),
    format("bounds = [~w]~n~n", [BoundsJoined]),

    shape_layer_attributes(Counties, Attrs), data_frame_to_string(Attrs, AttrsText),
    format("attributes~n~w~n~n", [AttrsText]),

    Last is Size - 1,
    forall(between(0, Last, I),
           ( data_frame_string_at(Attrs, "NAME", I, Name),
             shape_layer_num_parts(Counties, I, Parts),
             format("~w: ~w ring(s)~n", [Name, Parts]) )),

    SvgPath = "datamunge_gis_ex_map_prolog.svg",
    shape_layer_plot(Counties, Plot),
    plot_save_svg(Plot, SvgPath),
    format("~nmap saved to ~w~n", [SvgPath]).

% Roughly C's %g: integral doubles print without a fractional part.
fmt_g(X, S) :-
    (   X =:= floor(X)
    ->  I is truncate(X), number_string(I, S)
    ;   format(string(S), "~6f", [X])
    ).

% Little- and big-endian packers: each gives the list of bytes of one value.
i4_le(V, Bytes) :- int_bytes_le(V, 4, Bytes).
i4_be(V, Bytes) :- int_bytes_le(V, 4, Le), reverse(Le, Bytes).
i2_le(V, Bytes) :- int_bytes_le(V, 2, Bytes).
double_le(X, Bytes) :- double_bits(X, Bits), int_bytes_le(Bits, 8, Bytes).
ascii(S, Bytes) :- string_codes(S, Bytes).
zeros(N, Bytes) :- length(Bytes, N), maplist(=(0), Bytes).

% The N little-endian bytes of V (two's complement).
int_bytes_le(V, N, Bytes) :-
    U is V /\ ((1 << (8 * N)) - 1),
    Last is N - 1,
    findall(B, ( between(0, Last, K), B is (U >> (8 * K)) /\ 255 ), Bytes).

% The IEEE 754 binary64 bit pattern of a finite, normal (or zero) double.
double_bits(X, 0) :- X =:= 0, copysign(1.0, X) > 0, !.
double_bits(X, Bits) :- X =:= 0, !, Bits is 1 << 63.
double_bits(X, Bits) :-
    ( X < 0 -> Sign = 1 ; Sign = 0 ),
    A is abs(X),
    float_parts(A, M, 2, E),                 % A = M * 2^E, M in [0.5, 1)
    BiasedExp is E - 1 + 1023,
    must_be(between(1, 2046), BiasedExp),
    Fraction is truncate((M * 2 - 1) * 2 ** 52),
    Bits is (Sign << 63) \/ (BiasedExp << 52) \/ Fraction.

write_bytes(Path, Bytes) :-
    setup_call_cleanup(open(Path, write, Out, [type(binary)]),
                       maplist(put_byte(Out), Bytes),
                       close(Out)).

% Bounding box (minx, miny, maxx, maxy) of every point in Rings.
ring_bounds(Rings, [MinX, MinY, MaxX, MaxY]) :-
    append(Rings, Points),
    pairs_keys_values(Points, Xs, Ys),
    min_list(Xs, MinX), min_list(Ys, MinY), max_list(Xs, MaxX), max_list(Ys, MaxY).

% Bytes of one Polygon record's contents (without the record header).
polygon_record(Rings, Bytes) :-
    i4_le(5, Type),  % shape type: Polygon
    ring_bounds(Rings, Bounds), maplist(double_le, Bounds, BoundsBytes),
    length(Rings, NParts), i4_le(NParts, PartsCount),
    maplist(length, Rings, Lengths), sum_list(Lengths, NPoints), i4_le(NPoints, PointsCount),
    foldl([L, S0-Acc, S-[B|Acc]]>>(i4_le(S0, B), S is S0 + L), Lengths, 0-[], _-StartsRev),
    reverse(StartsRev, Starts),
    append(Rings, Points),
    findall(PB, ( member(X-Y, Points), double_le(X, XB), double_le(Y, YB), append(XB, YB, PB) ), PointBytes),
    append([[Type], BoundsBytes, [PartsCount, PointsCount], Starts, PointBytes], Chunks),
    append(Chunks, Bytes).

write_counties_shp(Path, Shapes) :-
    maplist(polygon_record, Shapes, Contents),
    foldl([C, W0, W]>>(length(C, Len), W is W0 + 4 + Len // 2), Contents, 0, TotalWords),
    i4_be(9994, Magic),
    findall(Z, ( between(1, 5, _), i4_be(0, Z) ), Unused),
    FileWords is 50 + TotalWords, i4_be(FileWords, Length),
    i4_le(1000, Version),
    i4_le(5, Type),  % Polygon
    append(Shapes, AllRings), ring_bounds(AllRings, Bounds), maplist(double_le, Bounds, BoundsBytes),
    findall(D, ( between(1, 4, _), double_le(0.0, D) ), ZBounds),
    findall(Record,
            ( nth1(I, Contents, C),
              i4_be(I, Number), length(C, Len), Words is Len // 2, i4_be(Words, RecordLength),
              append([Number, RecordLength, C], Record) ),
            Records),
    append([[Magic], Unused, [Length, Version, Type], BoundsBytes, ZBounds, Records], Chunks),
    append(Chunks, Bytes),
    write_bytes(Path, Bytes).

ljust(S, N, Ch, Out) :-
    string_length(S, L),
    (   L >= N
    ->  sub_string(S, 0, N, _, Out)
    ;   Pad is N - L, length(Cs, Pad), maplist(=(Ch), Cs),
        string_codes(PadS, Cs), string_concat(S, PadS, Out)
    ).

dbf_field(Name, FieldType, Length, Bytes) :-
    ljust(Name, 11, 0, Padded), ascii(Padded, NameBytes),
    ascii(FieldType, TypeBytes),
    zeros(4, Z4), zeros(15, Z15),
    append([NameBytes, TypeBytes, Z4, [Length], Z15], Bytes).

write_counties_dbf(Path, Names, Populations) :-
    HeaderSize is 32 + 2 * 32 + 1,
    RecordSize is 1 + 12 + 8,
    length(Names, NRecords), i4_le(NRecords, Count),
    i2_le(HeaderSize, HS), i2_le(RecordSize, RS),
    zeros(20, Z20),
    dbf_field("NAME", "C", 12, F1),
    dbf_field("POP", "N", 8, F2),
    findall(RecordBytes,
            ( nth0(I, Names, Name), nth0(I, Populations, Pop),
              ljust(Name, 12, 0' , N12), ljust(Pop, 8, 0' , P8),
              ascii(" ", B0), ascii(N12, B1), ascii(P8, B2),
              append([B0, B1, B2], RecordBytes) ),
            Records),
    append([[[0x03, 0, 0, 0], Count, HS, RS, Z20, F1, F2, [0x0D]], Records], Chunks),
    append(Chunks, Bytes),
    write_bytes(Path, Bytes).
