#include <datamunge/algorithms/exact_cover.hpp>

#include <cstddef>
#include <vector>

namespace datamunge::algorithms {

namespace {

// Dancing Links (DLX) over index-based nodes. Node 0 is the root header `h`.
struct Dlx {
    std::vector<int> L, R, U, D, C, ROW; // links, column-header index, original row id
    std::vector<int> S;                  // column sizes (only meaningful for headers)
    int              h = 0;

    int make(int col, int row) {
        const int id = static_cast<int>(L.size());
        L.push_back(id);
        R.push_back(id);
        U.push_back(id);
        D.push_back(id);
        C.push_back(col);
        ROW.push_back(row);
        S.push_back(0);
        return id;
    }

    void cover(int c) {
        L[R[c]] = L[c];
        R[L[c]] = R[c];
        for (int i = D[c]; i != c; i = D[i])
            for (int j = R[i]; j != i; j = R[j]) {
                U[D[j]] = U[j];
                D[U[j]] = D[j];
                --S[C[j]];
            }
    }

    void uncover(int c) {
        for (int i = U[c]; i != c; i = U[i])
            for (int j = L[i]; j != i; j = L[j]) {
                ++S[C[j]];
                U[D[j]] = j;
                D[U[j]] = j;
            }
        L[R[c]] = c;
        R[L[c]] = c;
    }

    bool search(std::vector<int>& solution) {
        if (R[h] == h) return true; // all columns covered
        // Choose the column with the fewest rows (S-heuristic).
        int c = R[h];
        for (int j = R[h]; j != h; j = R[j])
            if (S[j] < S[c]) c = j;
        if (S[c] == 0) return false; // a column no row can cover

        cover(c);
        for (int r = D[c]; r != c; r = D[r]) {
            solution.push_back(ROW[r]);
            for (int j = R[r]; j != r; j = R[j]) cover(C[j]);
            if (search(solution)) return true;
            for (int j = L[r]; j != r; j = L[j]) uncover(C[j]);
            solution.pop_back();
        }
        uncover(c);
        return false;
    }
};

} // namespace

ExactCoverResult exact_cover(int num_columns, const std::vector<std::vector<int>>& rows) {
    Dlx dlx;
    dlx.h = dlx.make(-1, -1); // root, index 0

    // Column headers, linked into the header row after the root.
    std::vector<int> colhdr(num_columns);
    int              prev = dlx.h;
    for (int j = 0; j < num_columns; ++j) {
        const int c = dlx.make(j, -1);
        dlx.C[c]    = c; // a header's column is itself
        dlx.R[prev] = c;
        dlx.L[c]    = prev;
        prev        = c;
        colhdr[j]   = c;
    }
    dlx.R[prev]   = dlx.h;
    dlx.L[dlx.h]  = prev;

    // Data nodes, one per 1 in the matrix.
    for (std::size_t r = 0; r < rows.size(); ++r) {
        int first = -1, pr = -1;
        for (int col : rows[r]) {
            const int c  = colhdr[col];
            const int nd = dlx.make(c, static_cast<int>(r));
            // insert nd at the bottom of column c (just above the header)
            dlx.U[nd]      = dlx.U[c];
            dlx.D[nd]      = c;
            dlx.D[dlx.U[c]] = nd;
            dlx.U[c]       = nd;
            ++dlx.S[c];
            if (first == -1) {
                first = nd;
            } else {
                dlx.R[pr] = nd;
                dlx.L[nd] = pr;
            }
            pr = nd;
        }
        if (first != -1) {
            dlx.R[pr]    = first;
            dlx.L[first] = pr;
        }
    }

    ExactCoverResult result;
    result.solved = dlx.search(result.rows);
    if (!result.solved) result.rows.clear();
    return result;
}

} // namespace datamunge::algorithms
