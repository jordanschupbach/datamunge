--  CMD: just TARGET=bspline_ex run-ada
--  Tensor-product B-spline regression: fit z ~ bs(x, y) to a smooth surface and predict.
with Ada.Text_IO; use Ada.Text_IO;
with Datamunge; use Datamunge;
with Dm_Examples; use Dm_Examples;

procedure Bspline_Ex is
   X, Y, Z, Pred : Long_Float_Vector;
   Data_Points : Data_Frame := New_Data_Frame;
   New_Points  : Data_Frame := New_Data_Frame;
   Surface     : Lm;
begin
   --  Sample a smooth surface z = sin(x) + y^2 on an 8x8 grid over [0, 1]^2.
   for Iy in 0 .. 7 loop
      for Ix in 0 .. 7 loop
         declare
            Xv : constant Long_Float := Long_Float (Ix) / 7.0;
            Yv : constant Long_Float := Long_Float (Iy) / 7.0;
         begin
            X.Append (Xv);
            Y.Append (Yv);
            Z.Append (C_Sin (Xv) + Yv * Yv);
         end;
      end loop;
   end loop;
   Data_Points.Add_Numeric_Column ("x", X);
   Data_Points.Add_Numeric_Column ("y", Y);
   Data_Points.Add_Numeric_Column ("z", Z);

   Surface := New_Lm (Data_Points, "z ~ bs(x, y)");
   Put_Line ("Fitted z ~ bs(x, y)");
   Surface.Print_Summary;

   New_Points.Add_Numeric_Column ("x", [0.25, 0.75]);
   New_Points.Add_Numeric_Column ("y", [0.50, 0.25]);
   Pred := Surface.Predict (New_Points);
   Put_Line ("Predictions: [" & F (Pred (1), 6) & ", " & F (Pred (2), 6) & "]");

   Free (Surface);
   Free (New_Points);
   Free (Data_Points);
end Bspline_Ex;
