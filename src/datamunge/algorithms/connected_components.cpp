/// @file connected_components.cpp
/// @brief Single balanced two-pass kernel, actual history capture and checked blob statistics.
#include <datamunge/algorithms/connected_components.hpp>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace datamunge::algorithms {
/// @brief Checked internal labelling and measurement helpers.
namespace {
using Image = std::vector<std::vector<int>>; ///< Owned row vectors used internally.

/// @brief Increment a work or measurement counter with a range check.
/// @param value Counter to increment.
/// @throws std::length_error size_t exhaustion.
void tick(std::size_t& value) {
    if (value == std::numeric_limits<std::size_t>::max()) throw std::length_error("components: counter overflow");
    ++value;
}
/// @brief Validate complete rectangular shape and compute its checked cell count.
/// @param image Image to inspect without mutation.
/// @return Complete cell count; zero-width row shapes remain valid.
/// @throws std::invalid_argument Ragged rows, including after an empty first row.
/// @throws std::length_error Shape product overflow.
std::size_t shape(const Image& image) {
    const auto cols = image.empty() ? 0 : image.front().size();
    for (const auto& row : image)
        if (row.size() != cols) throw std::invalid_argument("components: ragged image");
    if (cols && image.size() > std::numeric_limits<std::size_t>::max()/cols)
        throw std::length_error("components: shape overflow");
    return image.size()*cols;
}
/// @brief Checked balanced forest used solely by the producing kernel.
struct Forest {
    std::vector<int> parent{0};          ///< Actual forest, background dummy at zero.
    std::vector<unsigned int> rank{0};   ///< Root rank; bounded by log2(number of sets).
    ComponentLabelWork& work;            ///< Actual work counter owner.

    /// @brief Bind the executing kernel's counters.
    /// @param counters Counters living longer than this forest.
    explicit Forest(ComponentLabelWork& counters) : work(counters) {}
    /// @brief Create the next checked positive provisional singleton.
    /// @return New positive int label.
    /// @throws std::length_error More than INT_MAX positive labels or counter exhaustion.
    int make() {
        if (parent.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
            throw std::length_error("components: int label exhaustion");
        const int label = static_cast<int>(parent.size());
        parent.push_back(label);rank.push_back(0);tick(work.labels_created);
        return label;
    }
    /// @brief Find one valid internal label's root using counted path halving.
    /// @param x Valid allocated positive provisional label.
    /// @return Actual current representative.
    /// @throws std::length_error Counter exhaustion.
    int find(int x) {
        tick(work.find_calls);
        while (parent[x] != x) {
            tick(work.find_steps);tick(work.parent_writes);
            parent[x] = parent[parent[x]];x = parent[x];
        }
        return x;
    }
    /// @brief Attempt an actual rank-based union and return its observed transition.
    /// @param a First valid provisional label.
    /// @param b Second valid provisional label.
    /// @return Actual operands, roots and optional attachment; equal ranks keep smaller root.
    /// @throws std::length_error Counter exhaustion.
    ComponentLabelUnion unite(int a, int b) {
        tick(work.union_calls);
        const int ra = find(a), rb = find(b);
        if (ra == rb) return {a,b,ra,rb,ra,0,false};
        int winner = ra, loser = rb;
        if (rank[winner] < rank[loser] || (rank[winner] == rank[loser] && winner > loser))
            std::swap(winner,loser);
        parent[loser] = winner;tick(work.parent_writes);tick(work.successful_unions);
        if (rank[winner] == rank[loser]) ++rank[winner];
        return {a,b,ra,rb,winner,loser,true};
    }
};

/// @brief Execute both labelling scans, capturing actual history only when requested.
/// @tparam Record Retain snapshots and decisions when true.
/// @param image Source rectangular image; validated here before allocation.
/// @param eight Select diagonal earlier neighbours when true.
/// @param options Complete cell and online record caps.
/// @return Owned result, work and optional actual history.
/// @throws std::invalid_argument Ragged image.
/// @throws std::length_error Caps, counters, shape or int labels exceed their range.
template<bool Record>
ComponentLabelTrace run(const Image& image, bool eight, ComponentLabelOptions options) {
    const auto cells = shape(image);
    if (options.max_cells && cells > options.max_cells) throw std::length_error("components: complete cell cap");
    const auto rows = image.size(), cols = rows ? image[0].size() : 0;
    ComponentLabelTrace out;
    auto& result = out.analysis.result;auto& work = out.analysis.work;
    work.validated_cells = cells;result.labels.assign(rows,std::vector<int>(cols));
    Forest forest(work);std::size_t records = 0;
    /// @brief Charge one actual foreground/union record, never a background position.
    /// @throws std::length_error Actual event cap or record counter exhausted.
    const auto record = [&]() {
        if constexpr (Record) {
            if (options.max_events && records >= options.max_events) throw std::length_error("components: actual event cap");
            tick(records);
        }
    };
    for (std::size_t r=0;r<rows;++r) for (std::size_t c=0;c<cols;++c) {
        tick(work.first_pass_cells);if (!image[r][c]) continue;tick(work.foreground);
        ComponentLabelCell event;event.row=r;event.col=c;
        /// @brief Inspect one guarded in-bounds predecessor and retain its provisional label.
        /// @param nr Earlier row index.
        /// @param nc Earlier column index.
        /// @throws std::length_error Counter exhaustion.
        const auto consider = [&](std::size_t nr,std::size_t nc) {
            tick(work.neighbor_probes);const int value=result.labels[nr][nc];
            if (value) { tick(work.neighbor_labels);event.earlier_labels[event.earlier_count++]=value; }
        };
        if (r) consider(r-1,c);
        if (c) consider(r,c-1);
        if (eight && r && c) consider(r-1,c-1);
        if (eight && r && c+1<cols) consider(r-1,c+1);
        record();
        if constexpr (Record) event.union_begin=out.unions.size();
        if (!event.earlier_count) { event.assigned=forest.make();event.created=true; }
        else {
            event.assigned=*std::min_element(event.earlier_labels.begin(),event.earlier_labels.begin()+event.earlier_count);
            for (std::size_t k=0;k<event.earlier_count;++k) {
                record();const auto attempt=forest.unite(event.assigned,event.earlier_labels[k]);
                if constexpr (Record) out.unions.push_back(attempt);
            }
        }
        result.labels[r][c]=event.assigned;
        if constexpr (Record) {event.union_end=out.unions.size();out.cells.push_back(event);}
    }
    if constexpr (Record) out.provisional=result.labels;
    std::vector<int> remap(forest.parent.size());
    for (auto& row:result.labels) for (auto& label:row) {
        tick(work.second_pass_cells);if (!label) continue;
        const int root=forest.find(label);
        if (!remap[root]) {
            if (result.count==std::numeric_limits<int>::max()) throw std::length_error("components: dense label exhaustion");
            remap[root]=++result.count;
        }
        label=remap[root];tick(work.remapped_cells);
    }
    if constexpr (Record) {out.parents=std::move(forest.parent);out.root_to_dense=std::move(remap);}
    return out;
}
} // namespace

ConnectedComponents connected_components(const std::vector<std::vector<int>>& image, bool eight_connected) {
    return run<false>(image,eight_connected,{0,0}).analysis.result;
}
ComponentLabelAnalysis connected_components_analyze(const std::vector<std::vector<int>>& image,
    bool eight_connected, ComponentLabelOptions options) {
    return run<false>(image,eight_connected,options).analysis;
}
ComponentLabelTrace connected_components_trace(const std::vector<std::vector<int>>& image,
    bool eight_connected, ComponentLabelOptions options) {
    return run<true>(image,eight_connected,options);
}
std::vector<ConnectedComponentStats> connected_component_stats(const ConnectedComponents& components) {
    (void)shape(components.labels);
    if (components.count<0) throw std::invalid_argument("components: negative count");
    std::size_t foreground=0;
    for (const auto& row:components.labels) for (int label:row) {
        if (label<0 || label>components.count) throw std::invalid_argument("components: label outside 0..count");
        if (label) tick(foreground);
    }
    const auto count=static_cast<std::size_t>(components.count);
    if (count>foreground) throw std::invalid_argument("components: more IDs than foreground");
    std::vector<ConnectedComponentStats> out(count);
    std::vector<long double> row_sum(count),col_sum(count);
    const auto rows=components.labels.size(),cols=rows?components.labels[0].size():0;
    for (std::size_t r=0;r<rows;++r) for (std::size_t c=0;c<cols;++c) {
        const int label=components.labels[r][c];if (!label) continue;
        const auto id=static_cast<std::size_t>(label-1);auto& s=out[id];
        if (!s.area) {s.label=label;s.min_row=s.max_row=r;s.min_col=s.max_col=c;}
        tick(s.area);s.min_row=std::min(s.min_row,r);s.max_row=std::max(s.max_row,r);
        s.min_col=std::min(s.min_col,c);s.max_col=std::max(s.max_col,c);
        row_sum[id]+=static_cast<long double>(r);col_sum[id]+=static_cast<long double>(c);
        if (!r || components.labels[r-1][c]!=label) tick(s.perimeter4);
        if (!c || components.labels[r][c-1]!=label) tick(s.perimeter4);
        if (r+1==rows || components.labels[r+1][c]!=label) tick(s.perimeter4);
        if (c+1==cols || components.labels[r][c+1]!=label) tick(s.perimeter4);
    }
    for (std::size_t i=0;i<count;++i) {
        if (!out[i].area) throw std::invalid_argument("components: missing dense ID");
        out[i].centroid_row=static_cast<double>(row_sum[i]/out[i].area);
        out[i].centroid_col=static_cast<double>(col_sum[i]/out[i].area);
    }
    return out;
}
std::vector<std::vector<int>> component_blobs_example() {
    Image image(12,std::vector<int>(12));
    for (const auto& rect:std::array<std::array<std::size_t,4>,4>{{{1,1,3,3},{1,7,2,10},{6,2,9,3},{7,7,10,10}}})
        for (auto r=rect[0];r<=rect[2];++r) for(auto c=rect[1];c<=rect[3];++c) image[r][c]=1;
    return image;
}
std::vector<std::vector<int>> component_checkerboard(std::size_t rows,std::size_t cols,std::size_t max_cells) {
    if (!rows || !cols) throw std::invalid_argument("checkerboard: dimensions must be positive");
    if (rows>std::numeric_limits<std::size_t>::max()/cols) throw std::length_error("checkerboard: shape overflow");
    if (max_cells && rows*cols>max_cells) throw std::length_error("checkerboard: complete cell cap");
    Image image(rows,std::vector<int>(cols));
    for (std::size_t r=0;r<rows;++r) for (std::size_t c=0;c<cols;++c) image[r][c]=(r%2)==(c%2);
    return image;
}
} // namespace datamunge::algorithms
