--  CMD: just TARGET=knn_iris_ex run-ada
--  k-nearest-neighbours classifier on iris: leave-one-out accuracy and decision regions for several k.
with Ada.Text_IO; use Ada.Text_IO;
with Datamunge; use Datamunge;
with Dm_Examples; use Dm_Examples;

procedure Knn_Iris_Ex is
   Iris   : Data_Frame := Data.Iris;
   Model  : Knn_Classifier;
   Matrix : Data_Frame;
   Fitted : String_Vector;
   N      : Long_Long_Integer;
   Misclassified : Long_Long_Integer := 0;
   Regions : R_Plot;
begin
   Put_Line ("iris: " & Img (Iris.Nrows) & " rows x " & Img (Iris.Ncols) & " cols" & ASCII.LF);

   Model := New_Knn_Classifier (Iris, "Species ~ Petal.Length + Petal.Width");
   Model.Print_Summary;

   Put_Line (ASCII.LF & "Confusion matrix (leave-one-out; rows = actual, cols = predicted):");
   Matrix := Model.Confusion_Matrix;
   Put_Line (Matrix.Image);
   Free (Matrix);

   Put_Line (ASCII.LF & "Leave-one-out misclassified rows:");
   Fitted := Model.Predict (Iris);
   N := Iris.Nrows;
   for I in 0 .. N - 1 loop
      declare
         Actual    : constant String := Iris.String_At ("Species", I);
         Predicted : constant String := Fitted (Positive (I + 1));
      begin
         if Predicted /= Actual then
            Misclassified := Misclassified + 1;
            Put_Line ("  row " & Img (I) & ": Petal.Length=" & Fmt (Iris.Numeric_At ("Petal.Length", I))
                      & " Petal.Width=" & Fmt (Iris.Numeric_At ("Petal.Width", I))
                      & "  actual=" & Actual & "  predicted=" & Predicted);
         end if;
      end;
   end loop;
   Put_Line (Img (Misclassified) & " of " & Img (N) & " misclassified ("
             & Fmt (100.0 * Long_Float (Misclassified) / Long_Float (N)) & "%)");

   Regions := Model.Plot_Decision_Regions ("Petal.Length", "Petal.Width");
   Regions.Save ("knn_iris_decision_regions_k5.svg");
   Free (Regions);
   Put_Line (ASCII.LF & "Saved knn_iris_decision_regions_k5.svg");

   --  k=1 memorizes every training point exactly (jagged, overfit boundary with an island around
   --  every point, including noise); k=25 averages over a much larger neighborhood (very smooth,
   --  underfit boundary). New_Knn_Classifier (Data, Formula, K, Metric, Weighted, Standardize)
   for K of Long_Long_Integer_Vector'[1, 25] loop
      declare
         Knn  : Knn_Classifier := New_Knn_Classifier (Iris, "Species ~ Petal.Length + Petal.Width", K);
         File : constant String := "knn_iris_decision_regions_k" & Img (K) & ".svg";
      begin
         Put_Line (ASCII.LF & "k=" & Pad_Right (Img (K), 2) & " leave-one-out accuracy: "
                   & Fmt (Knn.Training_Accuracy * 100.0) & "%");
         Regions := Knn.Plot_Decision_Regions ("Petal.Length", "Petal.Width");
         Regions.Save (File);
         Free (Regions);
         Put_Line ("Saved " & File);
         Free (Knn);
      end;
   end loop;

   Free (Model);
   Free (Iris);
end Knn_Iris_Ex;
