--  CMD: just TARGET=tree_iris_ex run-ada
--  Decision tree classifier on iris: summary, confusion matrix, misclassified rows and plots.
with Ada.Text_IO; use Ada.Text_IO;
with Datamunge; use Datamunge;
with Dm_Examples; use Dm_Examples;

procedure Tree_Iris_Ex is
   Iris          : Data_Frame := Data.Iris;
   Model         : Decision_Tree_Classifier;
   Shallow       : Decision_Tree_Classifier;
   Table         : Data_Frame;
   Predictions   : String_Vector;
   N             : Long_Long_Integer;
   Misclassified : Long_Long_Integer := 0;
   Figure        : R_Plot;
begin
   Put_Line ("iris: " & Img (Iris.Nrows) & " rows x " & Img (Iris.Ncols) & " cols" & ASCII.LF);

   Model := New_Decision_Tree_Classifier (Iris, "Species ~ Petal.Length + Petal.Width");
   Model.Print_Summary;

   Put_Line (ASCII.LF & "Confusion matrix (rows = actual, cols = predicted):");
   Table := Model.Confusion_Matrix;
   Put_Line (Table.Image);
   Free (Table);

   Put_Line (ASCII.LF & "Misclassified rows:");
   Predictions := Model.Predict (Iris);
   N := Iris.Nrows;
   for I in 0 .. N - 1 loop
      declare
         Actual    : constant String := Iris.String_At ("Species", I);
         Predicted : constant String := Predictions (Positive (I + 1));
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

   Figure := Model.Plot_Classification (Iris, "Petal.Length", "Petal.Width");
   Figure.Save ("tree_iris_classification.svg");
   Free (Figure);
   Figure := Model.Plot_Decision_Regions ("Petal.Length", "Petal.Width");
   Figure.Save ("tree_iris_decision_regions.svg");
   Free (Figure);
   Put_Line (ASCII.LF & "Saved tree_iris_classification.svg and tree_iris_decision_regions.svg");

   --  A shallower tree, for comparison, showing a coarser (but still fairly accurate) decision boundary.
   Shallow := New_Decision_Tree_Classifier (Iris, "Species ~ Petal.Length + Petal.Width", 2);
   Put_Line (ASCII.LF & "Depth-2 tree training accuracy: " & Fmt (Shallow.Training_Accuracy * 100.0) & "% ("
             & Img (Shallow.Leaf_Count) & " leaves)");
   Figure := Shallow.Plot_Decision_Regions ("Petal.Length", "Petal.Width");
   Figure.Save ("tree_iris_decision_regions_depth2.svg");
   Free (Figure);
   Put_Line ("Saved tree_iris_decision_regions_depth2.svg");

   Free (Shallow);
   Free (Model);
   Free (Iris);
end Tree_Iris_Ex;
