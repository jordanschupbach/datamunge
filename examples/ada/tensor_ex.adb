--  CMD: just TARGET=tensor_ex run-ada
--  Tensors: construction, shape ops, broadcasting, reductions, linear algebra, masks, iris.
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with Ada.Text_IO; use Ada.Text_IO;
with Datamunge; use Datamunge;
with Dm_Examples; use Dm_Examples;

procedure Tensor_Ex is
   LF : constant Character := ASCII.LF;

   function Shape_String (T : not null access Tensor_Record'Class) return String is
      Shape  : constant Long_Long_Integer_Vector := T.Shape;
      Result : Unbounded_String := To_Unbounded_String ("[");
   begin
      for I in Shape.First_Index .. Shape.Last_Index loop
         if I > Shape.First_Index then
            Append (Result, ", ");
         end if;
         Append (Result, Img (Shape (I)));
      end loop;
      return To_String (Result) & "]";
   end Shape_String;

   --  The text of a tensor a call returned, which it then frees. (The binding treats the Tensor*
   --  these methods return as belonging to the receiver, so receivers are freed after results.)
   function Show (T : Tensor) return String is
      Owned : Tensor := T;
      Text  : constant String := Owned.Image;
   begin
      Free (Owned);
      return Text;
   end Show;

   R, Ar : Tensor;
begin
   Put_Line ("=================== Construction ===================");
   Put_Line ("zeros([2,3]): " & Show (Tensor_Zeros ([2, 3])));
   Put_Line ("eye(3): " & Show (Tensor_Eye (3)));

   Ar := Tensor_Arange (0.0, 12.0, 1.0);
   R := Ar.Reshape ([3, 4]);
   Put_Line ("arange(0,12).reshape([3,4]): " & R.Image);

   Put_Line (LF & "=================== Shape ops ===================");
   declare
      T : Tensor := R.Transpose;
   begin
      Put_Line ("transpose -> shape " & Shape_String (T));
      Free (T);
   end;
   Put_Line ("slice(axis=1, start=1, stop=3): " & Show (R.Slice (1, 1, 3)));

   Put_Line (LF & "=================== Broadcasting arithmetic ===================");
   declare
      Col : Tensor := Tensor_From_Values ([3, 1], [1.0, 2.0, 3.0]);
      Row : Tensor := Tensor_From_Values ([1, 4], [10.0, 20.0, 30.0, 40.0]);
   begin
      Put_Line ("(3,1) + (1,4) -> " & Show (Col.Add (Row)));
      Free (Row);
      Free (Col);
   end;

   Put_Line (LF & "=================== Reductions ===================");
   Put_Line ("r.sum() = " & Fmt (R.Sum) & ", r.mean() = " & Fmt (R.Mean));
   Put_Line ("column means (axis=0): " & Show (R.Mean_Axis (0)));

   Put_Line (LF & "=================== Linear algebra ===================");
   declare
      A : Tensor := Tensor_From_Values ([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
      B : Tensor := Tensor_From_Values ([3, 2], [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
   begin
      Put_Line ("matmul(2x3, 3x2) -> " & Show (A.Matmul (B)));
      Free (B);
      Free (A);
   end;
   declare
      V1 : Tensor := Tensor_From_Values ([3], [1.0, 2.0, 3.0]);
      V2 : Tensor := Tensor_From_Values ([3], [4.0, 5.0, 6.0]);
   begin
      Put_Line ("dot([1,2,3], [4,5,6]) = " & Fmt (V1.Dot (V2)));
      Put_Line ("outer(v1, v2) -> " & Show (V1.Outer (V2)));
      Free (V2);
      Free (V1);
   end;

   Put_Line (LF & "=================== Comparisons & masks ===================");
   declare
      Sixes : Tensor := Tensor_Full ([3, 4], 6.0);
      Mask  : Tensor := R.Greater_Equal (Sixes);
   begin
      Put_Line ("r >= 6 -> " & Mask.Image);
      Put_Line ("count(r >= 6) = " & Fmt (Mask.Sum));
      Free (Mask);
      Free (Sixes);
   end;

   Put_Line (LF & "=================== A real dataset as a Tensor ===================");
   declare
      Iris    : Data_Frame := Data.Iris;
      N       : constant Long_Long_Integer := Iris.Nrows;
      Columns : constant String_Vector := ["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"];
      Flat    : Long_Float_Vector;
      Species : String_Vector;
      X, Feature_Means, Means_Row, Centered, Centered_T, Scatter : Tensor;
      Species_Tensor, Setosa, Setosa_Mask : Tensor;
   begin
      for I in 0 .. N - 1 loop
         for Column of Columns loop
            Flat.Append (Iris.Numeric_At (Column, I));
         end loop;
      end loop;
      X := Tensor_From_Values ([N, 4], Flat);
      Put_Line ("iris feature tensor shape: " & Shape_String (X));

      Feature_Means := X.Mean_Axis (0);
      Means_Row := Feature_Means.Reshape ([1, 4]);
      Centered := X.Subtract (Means_Row);
      Centered_T := Centered.Transpose;
      Scatter := Centered_T.Matmul (Centered);
      Put_Line ("feature means (Sepal.Length, Sepal.Width, Petal.Length, Petal.Width): "
                & Feature_Means.Image);
      Put_Line ("(X-mean)^T (X-mean) [4x4 scatter matrix]: " & Scatter.Image (Max_Elements => 16));

      for I in 0 .. N - 1 loop
         Species.Append (Iris.String_At ("Species", I));
      end loop;
      Species_Tensor := Tensor_From_String_Values ([N], Species);
      Setosa := Tensor_From_String_Values ([1], ["setosa"]);
      Setosa_Mask := Species_Tensor.Equal (Setosa);
      Put_Line ("setosa count = " & Fmt (Setosa_Mask.Sum) & " (of " & Img (N) & " rows)");

      Free (Setosa_Mask);
      Free (Setosa);
      Free (Species_Tensor);
      Free (Scatter);
      Free (Centered_T);
      Free (Centered);
      Free (Means_Row);
      Free (Feature_Means);
      Free (X);
      Free (Iris);
   end;
   Free (R);
   Free (Ar);
end Tensor_Ex;
