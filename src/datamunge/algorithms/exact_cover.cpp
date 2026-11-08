/// @file exact_cover.cpp
/// @brief Checked iterative Algorithm X, actual reversible mesh witnesses and native domino encodings.
#include <datamunge/algorithms/exact_cover.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
namespace datamunge::algorithms {
/// @brief Checked incidence geometry and one live index-linked search implementation.
namespace exact_cover_detail {
/// @brief Add logical geometry or work counts without wrapping.
/// @param a Existing count.
/// @param b Additional count.
/// @return Checked sum.
/// @throws std::length_error On size_t overflow.
std::size_t add(std::size_t a,std::size_t b){if(b>std::numeric_limits<std::size_t>::max()-a)throw std::length_error("exact-cover count overflow");return a+b;}
/// @brief Multiply logical geometry without wrapping.
/// @param a First factor.
/// @param b Second factor.
/// @return Checked product.
/// @throws std::length_error On size_t overflow.
std::size_t multiply(std::size_t a,std::size_t b){if(a&&b>std::numeric_limits<std::size_t>::max()/a)throw std::length_error("exact-cover geometry overflow");return a*b;}
/// @brief Increment one actual work counter with overflow protection.
/// @param counter Borrowed counter to update.
/// @param amount Additional logical operations, default one.
/// @throws std::length_error On counter overflow.
void increment(std::size_t& counter,std::size_t amount=1){counter=add(counter,amount);}
/// @brief Checked complete input geometry.
struct Shape {
    std::size_t rows; ///< Input row records.
    std::size_t incidences; ///< Total nonzero memberships.
    std::size_t nodes; ///< Root plus headers plus incidences.
};
/// @brief Count all incidence entries and check complete node geometry.
/// @param columns Signed required column count.
/// @param rows Borrowed input incidence lists.
/// @return Checked input and node sizes.
/// @throws std::invalid_argument On negative columns.
/// @throws std::length_error On checked sums, IDs or container limits.
Shape shape(int columns,const std::vector<std::vector<int>>& rows){
    if(columns<0)throw std::invalid_argument("exact-cover columns must be nonnegative");
    std::size_t incidences=0;for(const auto& row:rows)incidences=add(incidences,row.size());
    return {rows.size(),incidences,exact_cover_node_count(columns,rows.size(),incidences)};
}
/// @brief Validate every required column and within-row uniqueness without reordering input.
/// @param columns Valid nonnegative required count.
/// @param rows Input rows with representable original IDs.
/// @throws std::invalid_argument On an invalid or repeated column within a row.
/// @throws std::bad_alloc If stamp storage is unavailable.
void validate_rows(int columns,const std::vector<std::vector<int>>& rows){
    std::vector<int> seen(static_cast<std::size_t>(columns),-1);
    for(std::size_t i=0;i<rows.size();++i)for(int col:rows[i]){
        if(col<0||col>=columns)throw std::invalid_argument("exact-cover column out of range");
        if(seen[col]==static_cast<int>(i))throw std::invalid_argument("exact-cover repeated column within row");
        seen[col]=static_cast<int>(i);
    }
}
/// @brief Reserve logical trace entries before optional copies or event captures.
/// @param trace Optional actual owned trace.
/// @param amount Additional logical entries.
/// @param cap Optional quota, zero disables.
/// @throws std::length_error On count or quota limits.
void capture(ExactCoverTrace* trace,std::size_t amount,std::size_t cap){
    if(!trace)return;const auto total=add(trace->captured_entries,amount);
    if(cap&&total>cap)throw std::length_error("exact-cover capture limit exceeded");trace->captured_entries=total;
}
/// @brief One explicit suspended Algorithm X level.
struct Frame {
    int column; ///< Covered pivot header node ID.
    int next_row; ///< Next actual vertical-list node to try after restoration.
    int active_row=-1; ///< Selected pivot-row node, or -1 between candidates.
};
/// @brief Live indexed dancing-links mesh with one actual iterative search and optional observations.
class Dlx {
public:
    std::vector<ExactCoverNode> nodes; ///< Actual root, header and data records in stable original order.
    ExactCoverWork work; ///< Actual checked search and restoration operations.

    /// @brief Construct the complete mesh in original node and input-entry order.
    /// @param columns Valid nonnegative required count.
    /// @param rows Validated original incidence rows.
    /// @param size Checked complete input geometry.
    /// @param trace Optional trace with fixed base already reserved.
    /// @param cap Optional retained-history quota.
    /// @throws std::bad_alloc If node/selected-row storage is unavailable.
    Dlx(int columns,const std::vector<std::vector<int>>& rows,const Shape& size,ExactCoverTrace* trace,std::size_t cap)
      :columns_(columns),trace_(trace),cap_(cap){
        nodes.reserve(size.nodes);selected_.reserve(std::min(static_cast<std::size_t>(columns),rows.size()));make(-1,-1);int previous=0;
        for(int c=0;c<columns;++c){const int id=make(c+1,-1);nodes[previous].right=id;nodes[id].left=previous;previous=id;}
        nodes[previous].right=0;nodes[0].left=previous;
        for(std::size_t r=0;r<rows.size();++r){int first=-1,last=-1;
            for(int col:rows[r]){const int c=col+1,id=make(c,static_cast<int>(r));
                nodes[id].up=nodes[c].up;nodes[id].down=c;nodes[nodes[c].up].down=id;nodes[c].up=id;++nodes[c].size;
                if(first==-1)first=id;else{nodes[last].right=id;nodes[id].left=last;}last=id;
            }
            if(first!=-1){nodes[last].right=first;nodes[first].left=last;}
        }
    }
    /// @brief Find the first cover using explicit frames, then restore every active frame.
    /// @return Owned first verdict and actual original row-selection order.
    /// @throws std::length_error On actual work/history/container limits.
    /// @throws std::bad_alloc If frame, result or history storage is unavailable.
    ExactCoverResult search(){
        ExactCoverResult result;std::vector<Frame> frames;bool entering=true;
        while(true){
            if(entering){increment(work.search_entries);
                if(nodes[0].right==0){result.solved=true;result.rows=selected_;event(ExactCoverEvent::solution,-1,-1);break;}
                const int c=choose();event(ExactCoverEvent::choose_column,c-1,-1);
                if(nodes[c].size==0){increment(work.dead_ends);event(ExactCoverEvent::dead_end,c-1,-1);entering=false;}
                else{cover(c);frames.push_back({c,nodes[c].down,-1});entering=false;}
            }
            if(frames.empty())break;
            auto& frame=frames.back();if(frame.active_row!=-1)restore_row(frame);
            const int r=frame.next_row;
            if(r==frame.column){uncover(frame.column);frames.pop_back();entering=false;continue;}
            frame.next_row=nodes[r].down;frame.active_row=r;selected_.push_back(nodes[r].row);increment(work.row_attempts);
            work.maximum_depth=std::max(work.maximum_depth,selected_.size());
            for(int j=nodes[r].right;j!=r;j=nodes[j].right)cover(nodes[j].column);
            event(ExactCoverEvent::select_row,frame.column-1,nodes[r].row);entering=true;
        }
        while(!frames.empty()){auto& frame=frames.back();if(frame.active_row!=-1)restore_row(frame);uncover(frame.column);frames.pop_back();}
        event(ExactCoverEvent::finished,-1,-1);return result;
    }
private:
    int columns_; ///< Nonnegative problem-column count.
    ExactCoverTrace* trace_; ///< Optional borrowed capture valid only during this call.
    std::size_t cap_; ///< Optional logical quota, zero disables.
    std::vector<int> selected_; ///< Actual currently selected original rows.

    /// @brief Append a self-linked node using prechecked representable IDs.
    /// @param column Header node ID or -1 for root.
    /// @param row Original row ID or -1 for root/header.
    /// @return New node ID.
    /// @throws std::bad_alloc If storage is unavailable.
    int make(int column,int row){const int id=static_cast<int>(nodes.size());nodes.push_back({id,id,id,id,column,row,0});return id;}
    /// @brief Select the earliest active column of minimum actual size.
    /// @return Actual header node ID; root is known nonempty.
    /// @throws std::length_error On scan counter overflow.
    int choose(){int c=nodes[0].right;for(int j=nodes[0].right;j!=0;j=nodes[j].right){increment(work.column_scans);if(nodes[j].size<nodes[c].size)c=j;}return c;}
    /// @brief Cover a live column and unlink conflicting row incidences.
    /// @param c Actual header node ID to cover.
    /// @throws std::length_error On actual operation counter overflow.
    void cover(int c){
        increment(work.covers);increment(work.header_link_writes,2);nodes[nodes[c].right].left=nodes[c].left;nodes[nodes[c].left].right=nodes[c].right;
        for(int i=nodes[c].down;i!=c;i=nodes[i].down)for(int j=nodes[i].right;j!=i;j=nodes[j].right){
            increment(work.data_link_writes,2);increment(work.column_size_updates);nodes[nodes[j].down].up=nodes[j].up;nodes[nodes[j].up].down=nodes[j].down;--nodes[nodes[j].column].size;
        }
    }
    /// @brief Undo a covered column in the exact inverse traversal order.
    /// @param c Actual header node ID to uncover.
    /// @throws std::length_error On actual operation counter overflow.
    void uncover(int c){
        increment(work.uncovers);
        for(int i=nodes[c].up;i!=c;i=nodes[i].up)for(int j=nodes[i].left;j!=i;j=nodes[j].left){
            increment(work.data_link_writes,2);increment(work.column_size_updates);++nodes[nodes[j].column].size;nodes[nodes[j].down].up=j;nodes[nodes[j].up].down=j;
        }
        increment(work.header_link_writes,2);nodes[nodes[c].right].left=c;nodes[nodes[c].left].right=c;
    }
    /// @brief Restore one selected row while retaining its frame's covered pivot column.
    /// @param frame Actual active frame, changed to no active row.
    /// @throws std::length_error On operation/history limits.
    /// @throws std::bad_alloc If event capture is unavailable.
    void restore_row(Frame& frame){
        const int r=frame.active_row;for(int j=nodes[r].left;j!=r;j=nodes[j].left)uncover(nodes[j].column);
        selected_.pop_back();increment(work.row_restorations);event(ExactCoverEvent::unselect_row,frame.column-1,nodes[r].row);frame.active_row=-1;
    }
    /// @brief Capture actual current headers and selected rows at one live event.
    /// @param kind Actual event kind.
    /// @param column Problem column ID, or -1.
    /// @param row Original row ID, or -1.
    /// @throws std::length_error On quota, count or step container limits.
    /// @throws std::bad_alloc If snapshot storage is unavailable.
    void event(ExactCoverEvent kind,int column,int row){
        if(!trace_)return;capture(trace_,add(add(1,static_cast<std::size_t>(columns_)),selected_.size()),cap_);
        if(trace_->steps.size()==trace_->steps.max_size())throw std::length_error("exact-cover event container exceeded");
        ExactCoverStep step;step.kind=kind;step.depth=selected_.size();step.column=column;step.row=row;
        step.column_sizes.assign(static_cast<std::size_t>(columns_),-1);for(int c=nodes[0].right;c!=0;c=nodes[c].right)step.column_sizes[c-1]=nodes[c].size;
        step.partial_rows=selected_;trace_->steps.push_back(std::move(step));
    }
};
/// @brief Validate/build/search once and optionally capture actual input and terminal meshes.
/// @param columns Requested required-column count.
/// @param rows Borrowed original incidence lists.
/// @param trace Optional owned output, with no externally borrowed values.
/// @param cap Optional retained-history quota.
/// @return Owned first result and actual fully restored search work.
/// @throws std::invalid_argument On invalid input incidence.
/// @throws std::length_error On geometry, work, container or history limits.
/// @throws std::bad_alloc If required runtime/output storage is unavailable.
ExactCoverOutcome run(int columns,const std::vector<std::vector<int>>& rows,ExactCoverTrace* trace,std::size_t cap){
    const auto size=shape(columns,rows);
    if(trace)capture(trace,add(add(size.rows,size.incidences),multiply(2,size.nodes)),cap);
    validate_rows(columns,rows);Dlx dlx(columns,rows,size,trace,cap);
    if(trace){trace->num_columns=columns;trace->input_rows=rows;trace->initial_nodes=dlx.nodes;}
    auto result=dlx.search();if(trace)trace->final_nodes=dlx.nodes;return {std::move(result),dlx.work};
}
/// @brief Validated complete rectangular domino geometry.
struct BoardShape {
    std::size_t cells; ///< Required board cells.
    std::size_t placements; ///< Complete horizontal/vertical pair count.
};
/// @brief Check all board/candidate/node counts before generation or rendering.
/// @param height Nonnegative requested board rows.
/// @param width Nonnegative requested board columns.
/// @return Checked cell and full placement counts.
/// @throws std::invalid_argument On negative dimensions.
/// @throws std::length_error On arithmetic/int/node/container limits.
BoardShape board_shape(int height,int width){
    if(height<0||width<0)throw std::invalid_argument("domino board dimensions must be nonnegative");
    const auto h=static_cast<std::size_t>(height),w=static_cast<std::size_t>(width),cells=multiply(h,w);
    if(cells>static_cast<std::size_t>(std::numeric_limits<int>::max()))throw std::length_error("domino cell IDs exceed int");
    const auto placements=h&&w?add(multiply(h,w-1),multiply(h-1,w)):0;
    (void)exact_cover_node_count(static_cast<int>(cells),placements,multiply(2,placements));return {cells,placements};
}
} // namespace exact_cover_detail
std::size_t exact_cover_node_count(int num_columns,std::size_t row_count,std::size_t nonzeros){
    if(num_columns<0)throw std::invalid_argument("exact-cover columns must be nonnegative");
    const auto nodes=exact_cover_detail::add(exact_cover_detail::add(1,static_cast<std::size_t>(num_columns)),nonzeros);
    const auto integer_limit=static_cast<std::size_t>(std::numeric_limits<int>::max());
    if(nodes>integer_limit||row_count>integer_limit||nodes>std::vector<ExactCoverNode>{}.max_size()||row_count>std::vector<std::vector<int>>{}.max_size()||static_cast<std::size_t>(num_columns)>std::vector<int>{}.max_size())
        throw std::length_error("exact-cover node/row/container geometry exceeded");
    return nodes;
}
ExactCoverOutcome exact_cover_work(int num_columns,const std::vector<std::vector<int>>& rows){return exact_cover_detail::run(num_columns,rows,nullptr,0);}
ExactCoverResult exact_cover(int num_columns,const std::vector<std::vector<int>>& rows){return exact_cover_work(num_columns,rows).result;}
ExactCoverTrace exact_cover_trace(int num_columns,const std::vector<std::vector<int>>& rows,std::size_t max_entries){
    ExactCoverTrace trace;trace.outcome=exact_cover_detail::run(num_columns,rows,&trace,max_entries);return trace;
}
DominoCoverProblem domino_exact_cover(int board_rows,int board_columns){
    const auto size=exact_cover_detail::board_shape(board_rows,board_columns);DominoCoverProblem p;p.board_rows=board_rows;p.board_columns=board_columns;p.num_columns=static_cast<int>(size.cells);p.placements.reserve(size.placements);
    if(!size.cells)return p;
    for(int r=0;r<board_rows;++r)for(int c=0;c<board_columns;++c){const int cell=r*board_columns+c;if(c+1<board_columns)p.placements.push_back({cell,cell+1});if(r+1<board_rows)p.placements.push_back({cell,cell+board_columns});}
    return p;
}
graphics::Diagram domino_tiling_diagram(const DominoCoverProblem& problem,const ExactCoverResult& result,double unit_pixels,double max_width,std::size_t max_cells){
    graphics::Diagram d(unit_pixels,max_width);const auto size=exact_cover_detail::board_shape(problem.board_rows,problem.board_columns);
    if(problem.num_columns!=static_cast<int>(size.cells))throw std::invalid_argument("domino problem cell count mismatch");
    if(max_cells&&size.cells>max_cells)throw std::length_error("domino render cell limit exceeded");
    for(const auto& row:problem.placements){if(row.size()!=2)throw std::invalid_argument("domino placement needs two cells");for(int cell:row)if(cell<0||cell>=problem.num_columns)throw std::invalid_argument("domino cell out of range");
        const int a=row[0],b=row[1];if(std::abs(a/problem.board_columns-b/problem.board_columns)+std::abs(a%problem.board_columns-b%problem.board_columns)!=1)throw std::invalid_argument("domino placement cells must be adjacent");}
    if(!result.solved&&!result.rows.empty())throw std::invalid_argument("unsolved domino result must have no rows");
    std::vector<unsigned char> covered(size.cells,0);
    for(int id:result.rows){if(id<0||static_cast<std::size_t>(id)>=problem.placements.size())throw std::invalid_argument("domino selected row out of range");for(int cell:problem.placements[id])if(covered[cell]++)throw std::invalid_argument("domino selected placements overlap");}
    if(result.solved&&std::any_of(covered.begin(),covered.end(),[](auto x){return x!=1;}))throw std::invalid_argument("domino solved result must cover every cell");
    if(!size.cells){d.text({0,0},"Empty board: no required cells",24);return d;}
    graphics::DiagramStroke grid;grid.color={148,163,184};grid.width=1;
    for(int r=0;r<problem.board_rows;++r)for(int c=0;c<problem.board_columns;++c)d.polygon({{double(c),double(r)},{double(c+1),double(r)},{double(c+1),double(r+1)},{double(c),double(r+1)}},{248,250,252},grid);
    const std::array<graphics::RGB,4> palette{{{191,219,254},{187,247,208},{254,240,138},{254,205,211}}};graphics::DiagramStroke outline;outline.color={51,65,85};outline.width=2;
    for(std::size_t i=0;i<result.rows.size();++i){const int id=result.rows[i],a=problem.placements[id][0],b=problem.placements[id][1];
        const double x0=std::min(a%problem.board_columns,b%problem.board_columns),x1=std::max(a%problem.board_columns,b%problem.board_columns)+1.;
        const double y0=problem.board_rows-1-std::max(a/problem.board_columns,b/problem.board_columns),y1=problem.board_rows-std::min(a/problem.board_columns,b/problem.board_columns);
        d.polygon({{x0,y0},{x1,y0},{x1,y1},{x0,y1}},palette[i%palette.size()],outline).text({(x0+x1)/2,(y0+y1)/2},std::to_string(id),20);
    }
    d.text({problem.board_columns/2.,problem.board_rows+.7},result.solved?"Selected domino placement row IDs":"Reported unsolved: no placements selected",24);return d;
}
} // namespace datamunge::algorithms
