/// @file blast.cpp
/// @brief One indexed seed traversal and one checked ungapped extension loop for search and inspection.
#include <datamunge/algorithms/blast.hpp>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <unordered_map>

namespace datamunge::algorithms {
/// @brief Private checked coordinates, indexed seeding and shared extension mechanics.
namespace {
/// @brief Validate the representable coordinate range and positive word length.
/// @param query Borrowed query bytes.
/// @param db Borrowed database bytes.
/// @param word_size Requested word length.
/// @throws std::length_error If an input exceeds INT_MAX bytes.
/// @throws std::invalid_argument If word_size<=0.
void validate_words(const std::string& query,const std::string& db,int word_size) {
    if (word_size<=0) throw std::invalid_argument("BLAST word_size must be positive");
    if (query.size()>std::numeric_limits<int>::max() || db.size()>std::numeric_limits<int>::max())
        throw std::length_error("BLAST byte coordinates exceed INT_MAX");
}
/// @brief Validate the signed match/mismatch convention and nonnegative drop allowance.
/// @param match Positive reward.
/// @param mismatch Negative penalty, including INT_MIN.
/// @param xdrop Nonnegative loss allowance.
/// @throws std::invalid_argument For an unsupported scoring sign or drop.
void validate_scoring(int match,int mismatch,int xdrop) {
    if (match<=0 || mismatch>=0 || xdrop<0)
        throw std::invalid_argument("BLAST requires match>0, mismatch<0 and xdrop>=0");
}
/// @brief Emit all exact seed pairs in deterministic query/database order.
/// @tparam Emit Callable receiving a BlastSeed.
/// @param query Validated query bytes.
/// @param db Validated database bytes.
/// @param word_size Positive word length.
/// @param emit Seed consumer; exceptions propagate.
template<class Emit>
void visit_seeds(const std::string& query,const std::string& db,int word_size,Emit emit) {
    const int nq=static_cast<int>(query.size()),nd=static_cast<int>(db.size());
    if (word_size>nq || word_size>nd) return;
    std::unordered_map<std::string,std::vector<int>> words;
    for (int di=0;di<=nd-word_size;++di) words[db.substr(di,word_size)].push_back(di);
    for (int qi=0;qi<=nq-word_size;++qi) {
        const auto found=words.find(query.substr(qi,word_size));
        if (found!=words.end()) for (int di:found->second) emit(BlastSeed{qi,di});
    }
}
/// @brief Extend one validated exact seed, optionally recording actual pair inspections.
/// @tparam Observe Callable receiving a BlastExtensionStep.
/// @param query Validated query bytes.
/// @param db Validated database bytes.
/// @param seed In-range exact word starts.
/// @param word_size Positive word length.
/// @param match Positive reward.
/// @param mismatch Negative penalty.
/// @param xdrop Nonnegative allowed peak loss.
/// @param observe Event consumer; exceptions propagate.
/// @return Best inspected seed-containing HSP, with checked int score.
/// @throws std::overflow_error If its score exceeds INT_MAX.
template<class Observe>
HSP extend(const std::string& query,const std::string& db,BlastSeed seed,int word_size,
           int match,int mismatch,int xdrop,Observe observe) {
    std::int64_t gains[2]{0,0};int lengths[2]{0,0};
    for (int side=0;side<2;++side) {
        const bool left=side==1;
        const int available=left?std::min(seed.query_start,seed.db_start):
            std::min(static_cast<int>(query.size())-seed.query_start-word_size,
                     static_cast<int>(db.size())-seed.db_start-word_size);
        std::int64_t running=0;
        for (int offset=0;offset<available;++offset) {
            const int qi=left?seed.query_start-1-offset:seed.query_start+word_size+offset;
            const int di=left?seed.db_start-1-offset:seed.db_start+word_size+offset;
            const int delta=query[qi]==db[di]?match:mismatch;
            running+=delta;
            if (running>gains[side]) {gains[side]=running;lengths[side]=offset+1;}
            const bool stopped=gains[side]-running>xdrop;
            observe(BlastExtensionStep{left,qi,di,delta,running,gains[side],stopped});
            if (stopped) break;
        }
    }
    const auto score=static_cast<std::int64_t>(word_size)*match+gains[0]+gains[1];
    if (score>std::numeric_limits<int>::max()) throw std::overflow_error("BLAST HSP score exceeds INT_MAX");
    return {seed.query_start-lengths[1],seed.db_start-lengths[1],
            word_size+lengths[0]+lengths[1],static_cast<int>(score)};
}
} // namespace

std::vector<BlastSeed> blast_seeds(const std::string& query,const std::string& db,int word_size) {
    validate_words(query,db,word_size);
    std::vector<BlastSeed> result;
    visit_seeds(query,db,word_size,[&](BlastSeed seed){result.push_back(seed);});
    return result;
}

BlastExtensionTrace blast_extend_trace(const std::string& query,const std::string& db,
    BlastSeed seed,int word_size,int match,int mismatch,int xdrop) {
    validate_words(query,db,word_size);validate_scoring(match,mismatch,xdrop);
    if (seed.query_start<0 || seed.db_start<0 || word_size>static_cast<int>(query.size()) ||
        word_size>static_cast<int>(db.size()) || seed.query_start>static_cast<int>(query.size())-word_size ||
        seed.db_start>static_cast<int>(db.size())-word_size ||
        query.compare(seed.query_start,word_size,db,seed.db_start,word_size)!=0)
        throw std::invalid_argument("BLAST trace requires an in-range exact seed");
    BlastExtensionTrace result;result.seed=seed;
    result.hsp=extend(query,db,seed,word_size,match,mismatch,xdrop,
                      [&](const BlastExtensionStep& step){result.steps.push_back(step);});
    return result;
}

std::vector<HSP> blast_search(const std::string& query,const std::string& db,
    int word_size,int match,int mismatch,int xdrop,int min_score) {
    validate_words(query,db,word_size);validate_scoring(match,mismatch,xdrop);
    std::vector<HSP> hsps;
    std::unordered_map<std::int64_t,std::size_t> best_on_diagonal;
    /// @brief Extend every seed, threshold its raw score and retain the diagonal's strict best.
    /// @param seed Exact word emitted by the shared index traversal.
    auto consume=[&](BlastSeed seed) {
        const auto hsp=extend(query,db,seed,word_size,match,mismatch,xdrop,[](const BlastExtensionStep&){});
        if (hsp.score<min_score) return;
        const auto diagonal=static_cast<std::int64_t>(hsp.db_start)-hsp.query_start;
        const auto found=best_on_diagonal.find(diagonal);
        if (found==best_on_diagonal.end()) {
            best_on_diagonal[diagonal]=hsps.size();hsps.push_back(hsp);
        } else if (hsp.score>hsps[found->second].score) hsps[found->second]=hsp;
    };
    visit_seeds(query,db,word_size,consume);
    std::sort(hsps.begin(),hsps.end(),[](const HSP& a,const HSP& b) {
        if (a.score!=b.score) return a.score>b.score;
        return std::tie(a.query_start,a.db_start,a.length)<std::tie(b.query_start,b.db_start,b.length);
    });
    return hsps;
}
} // namespace datamunge::algorithms
