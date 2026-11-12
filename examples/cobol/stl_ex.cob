*> C++ standard containers from COBOL: std::vector<double> and std::vector<int> handles,
*> and pairs, which the bindings flatten into -FIRST/-SECOND fields.
*> Run with `just run-cobol TARGET=stl_ex`.
IDENTIFICATION DIVISION.
PROGRAM-ID. STL-EX.
DATA DIVISION.
WORKING-STORAGE SECTION.
COPY "datamunge.cpy".

01 N            USAGE BINARY-LONG VALUE 100.
01 I            USAGE BINARY-LONG.
01 V            USAGE BINARY-DOUBLE UNSIGNED.
01 V2           USAGE BINARY-DOUBLE UNSIGNED.
01 DOUBLE-TEXT  PIC ZZZ9.9.
01 INT-TEXT     PIC -(9)9.
01 INT-TEXT-2   PIC -(9)9.
*> A std::pair<int, int> and a std::pair<double, double>, flattened like the bindings do.
01 P.
   05 P-FIRST   USAGE BINARY-LONG VALUE 3.
   05 P-SECOND  USAGE BINARY-LONG VALUE 4.
01 P2.
   05 P2-FIRST  USAGE COMP-2 VALUE 10.
   05 P2-SECOND USAGE COMP-2 VALUE 20.

PROCEDURE DIVISION.
    *> std::vector<double>: push i * 1.5, then read each element back.
    CALL "dm_vector_double_new" USING DM-VECTOR-DOUBLE DM-STATUS
    PERFORM CHECK
    MOVE DM-VECTOR-DOUBLE-SELF TO V
    PERFORM VARYING I FROM 0 BY 1 UNTIL I >= N
        COMPUTE DM-VECTOR-DOUBLE-VALUE = I * 1.5
        CALL "dm_vector_double_push" USING DM-VECTOR-DOUBLE DM-STATUS
        PERFORM CHECK
    END-PERFORM
    PERFORM VARYING I FROM 1 BY 1 UNTIL I > N
        MOVE I TO DM-VECTOR-DOUBLE-INDEX
        CALL "dm_vector_double_get" USING DM-VECTOR-DOUBLE DM-STATUS
        PERFORM CHECK
        MOVE DM-VECTOR-DOUBLE-VALUE TO DOUBLE-TEXT
        DISPLAY FUNCTION TRIM(DOUBLE-TEXT)
    END-PERFORM

    *> std::vector<int>: the same values truncated to integers.
    CALL "dm_vector_int_new" USING DM-VECTOR-INT DM-STATUS
    PERFORM CHECK
    MOVE DM-VECTOR-INT-SELF TO V2
    PERFORM VARYING I FROM 0 BY 1 UNTIL I >= N
        COMPUTE DM-VECTOR-INT-VALUE = I * 1.5
        CALL "dm_vector_int_push" USING DM-VECTOR-INT DM-STATUS
        PERFORM CHECK
    END-PERFORM
    PERFORM VARYING I FROM 1 BY 1 UNTIL I > N
        MOVE I TO DM-VECTOR-INT-INDEX
        CALL "dm_vector_int_get" USING DM-VECTOR-INT DM-STATUS
        PERFORM CHECK
        MOVE DM-VECTOR-INT-VALUE TO INT-TEXT
        DISPLAY FUNCTION TRIM(INT-TEXT)
    END-PERFORM

    MOVE P-FIRST TO INT-TEXT
    MOVE P-SECOND TO INT-TEXT-2
    DISPLAY "p: (" FUNCTION TRIM(INT-TEXT) ", " FUNCTION TRIM(INT-TEXT-2) ")"
    DISPLAY "p2: (" P2-FIRST ", " P2-SECOND ")"

    MOVE V TO DM-HANDLE-SELF
    CALL "dm_close" USING DM-HANDLE DM-STATUS
    MOVE V2 TO DM-HANDLE-SELF
    CALL "dm_close" USING DM-HANDLE DM-STATUS
    STOP RUN.

*> Stops with the error after a failed call.
CHECK.
    IF NOT DM-STATUS-OK
        DISPLAY "datamunge error (" FUNCTION TRIM(DM-STATUS-KIND) "): "
            DM-STATUS-MESSAGE(1:DM-STATUS-LENGTH) UPON SYSERR
        STOP RUN RETURNING 1
    END-IF.
