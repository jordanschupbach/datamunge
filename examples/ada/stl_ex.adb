--  CMD: just TARGET=stl_ex run-ada
--  C++ std::vector and std::pair values in Ada: vectors are Ada.Containers.Vectors, pairs are records.
with Ada.Text_IO; use Ada.Text_IO;
with Datamunge; use Datamunge;
with Dm_Examples; use Dm_Examples;

procedure Stl_Ex is
   N  : constant := 100;
   V  : Long_Float_Vector;
   V2 : Integer_Vector;

   type Int_Pair is record
      First, Second : Integer;
   end record;
   P  : constant Int_Pair := (3, 4);
   P2 : constant Int_Pair := (10, 20);
   P3 : Long_Float_Long_Float_Pair;
begin
   for I in 0 .. N - 1 loop
      V.Append (Long_Float (I) * 1.5);
   end loop;
   for X of V loop
      Put_Line (Fmt (X));
   end loop;
   for I in 0 .. N - 1 loop
      V2.Append (Integer (Long_Float'Floor (Long_Float (I) * 1.5)));
   end loop;
   for X of V2 loop
      Put_Line (Img (Long_Long_Integer (X)));
   end loop;

   Put_Line ("p: (" & Img (Long_Long_Integer (P.First)) & ", " & Img (Long_Long_Integer (P.Second)) & ")");
   Put_Line ("p2: (" & Img (Long_Long_Integer (P2.First)) & ", " & Img (Long_Long_Integer (P2.Second)) & ")");

   --  A C++ std::pair<double,double> comes back as a record.
   P3 := Make_Dpair (1.25, 2.75);
   Put_Line ("make_dpair(1.25, 2.75): (" & Fmt (P3.First) & ", " & Fmt (P3.Second) & "), sum_dpair = "
             & Fmt (Sum_Dpair (P3)));
end Stl_Ex;
