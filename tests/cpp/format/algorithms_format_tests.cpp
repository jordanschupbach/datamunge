// std::format / std::print support for datamunge::algorithms types.
#include <dm/dm.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <format>
#include <random>
#include <string>
#include <tuple>
#include <vector>

namespace alg = datamunge::algorithms;

// ---- enums and stateful classes ------------------------------------------------------------------

static_assert(std::formattable<alg::PegOp, char>);
static_assert(std::formattable<alg::RoundMode, char>);
static_assert(std::formattable<alg::SimplexStatus, char>);
static_assert(std::formattable<alg::NagleEvent::Type, char>);
static_assert(std::formattable<alg::Point2D, char>);
static_assert(std::formattable<alg::PlanarPoint, char>);
static_assert(std::formattable<alg::Pixel, char>);
static_assert(std::formattable<alg::NagleEvent, char>);
static_assert(std::formattable<alg::Production, char>);
static_assert(std::formattable<alg::ContextFreeGrammar, char>);
static_assert(std::formattable<alg::MorrisCounter, char>);
static_assert(std::formattable<alg::BloomFilter, char>);
static_assert(std::formattable<alg::BuddyAllocator, char>);
static_assert(std::formattable<alg::KarnEstimator, char>);
static_assert(std::formattable<alg::LamportClock, char>);
static_assert(std::formattable<alg::OnlineVariance, char>);
static_assert(std::formattable<alg::OnlineCovariance, char>);
static_assert(std::formattable<alg::LL1Parser, char>);
static_assert(std::formattable<alg::SLRParser, char>);
static_assert(std::formattable<alg::PackratParser, char>);
static_assert(std::formattable<alg::SummedAreaTable, char>);
static_assert(std::formattable<alg::LeakyBucket, char>);
static_assert(std::formattable<alg::TokenBucket, char>);
static_assert(std::formattable<alg::CosineLSH, char>);
static_assert(std::formattable<alg::Rc4, char>);
static_assert(std::formattable<alg::ZigguratNormal, char>);

TEST(AlgorithmsFormat, EnumsPrintEnumeratorNames) {
    EXPECT_EQ(std::format("{}", alg::PegOp::Choice), "Choice");
    EXPECT_EQ(std::format("{}", alg::PegOp::Not), "Not");
    EXPECT_EQ(std::format("{}", static_cast<alg::PegOp>(42)), "PegOp(42)");
    EXPECT_EQ(std::format("{}", alg::RoundMode::HalfToEven), "HalfToEven");
    EXPECT_EQ(std::format("{}", alg::RoundMode::Floor), "Floor");
    EXPECT_EQ(std::format("{}", static_cast<alg::RoundMode>(-1)), "RoundMode(-1)");
    EXPECT_EQ(std::format("{}", alg::SimplexStatus::Unbounded), "Unbounded");
    EXPECT_EQ(std::format("{}", alg::NagleEvent::Ack), "Ack");
}

TEST(AlgorithmsFormat, SmallValueTypes) {
    EXPECT_EQ(std::format("{}", alg::Point2D{1.5, -2}), "Point2D(1.5, -2)");
    EXPECT_EQ(std::format("{}", alg::PlanarPoint{3, 4}), "PlanarPoint(3, 4)");
    EXPECT_EQ(std::format("{}", alg::Pixel{7, 8}), "Pixel(7, 8)");
    EXPECT_EQ(std::format("{}", alg::NagleEvent{alg::NagleEvent::Write, 100}), "NagleEvent(Write, bytes=100)");
    EXPECT_EQ(std::format("{}", alg::Circle2D{{1, 2}, 3}), "Circle2D(center=Point2D(1, 2), radius=3)");
    EXPECT_EQ(std::format("{:>18}", alg::Point2D{1, 2}), "     Point2D(1, 2)");
}

TEST(AlgorithmsFormat, GrammarsAndParsers) {
    const alg::ContextFreeGrammar g{{{"S", {"a", "S", "b"}}, {"S", {}}}, "S"};
    EXPECT_EQ(std::format("{}", g.productions[0]), "Production(S -> a S b)");
    EXPECT_EQ(std::format("{}", g.productions[1]), "Production(S -> ε)");
    EXPECT_EQ(std::format("{}", g), "ContextFreeGrammar(start=S, productions=2)\n  S -> a S b\n  S -> ε");
    EXPECT_EQ(std::format("{}", alg::LL1Parser(g)), "LL1Parser(ll1=true)");
    const std::string slr = std::format("{}", alg::SLRParser(g));
    EXPECT_TRUE(slr.starts_with("SLRParser(slr=true, states=")) << slr;

    alg::PackratParser peg({{"S", alg::peg_plus(alg::peg_lit("a"))}}, "S");
    EXPECT_EQ(std::format("{}", peg), "PackratParser(rule_calls=0, memo_hits=0)");
    EXPECT_TRUE(peg.parse("aaa"));
    EXPECT_EQ(std::format("{}", peg),
              std::format("PackratParser(rule_calls={}, memo_hits={})", peg.rule_calls(), peg.memo_hits()));
    EXPECT_EQ(std::format("{}", alg::peg_lit("x")), R"x(PegExpr(op=Literal, text="x", children=[]))x");
}

TEST(AlgorithmsFormat, CountersClocksAndAllocators) {
    alg::MorrisCounter morris;
    EXPECT_EQ(std::format("{}", morris), "MorrisCounter(exponent=0, estimate=0)");

    alg::BloomFilter bloom(8, 2);
    EXPECT_EQ(std::format("{}", bloom), "BloomFilter(bits=8, hashes=2, fill_ratio=0)");
    bloom.add(1);
    EXPECT_EQ(std::format("{}", bloom), std::format("BloomFilter(bits=8, hashes=2, fill_ratio={})", bloom.fill_ratio()));

    alg::BuddyAllocator buddy(64, 8);
    EXPECT_EQ(std::format("{}", buddy),
              "BuddyAllocator(size=64, min_block=8, allocated=0, requested=0, free_blocks=[0, 0, 0, 1])");
    buddy.allocate(5);
    EXPECT_EQ(std::format("{}", buddy),
              "BuddyAllocator(size=64, min_block=8, allocated=8, requested=5, free_blocks=[1, 1, 1, 0])");

    alg::KarnEstimator karn(2.0);
    EXPECT_EQ(std::format("{}", karn), "KarnEstimator(rto=2, srtt=0, rttvar=0, initialized=false)");

    alg::LamportClock clock;
    clock.local_event();
    clock.receive(5);
    EXPECT_EQ(std::format("{}", clock), "LamportClock(time=6)");

    alg::SummedAreaTable sat({{1, 2}, {3, 4}});
    EXPECT_EQ(std::format("{}", sat), "SummedAreaTable(rows=2, cols=2, total=10)");
    EXPECT_EQ(std::format("{}", alg::SummedAreaTable({})), "SummedAreaTable(rows=0, cols=0, total=0)");
}

TEST(AlgorithmsFormat, OnlineStatistics) {
    alg::OnlineVariance v;
    EXPECT_EQ(std::format("{}", v), "OnlineVariance(n=0, mean=0, variance=0, sample_variance=0)");
    for (double x : {1.0, 2.0, 3.0, 4.0}) v.add(x);
    EXPECT_EQ(std::format("{}", v), "OnlineVariance(n=4, mean=2.5, variance=1.25, sample_variance=1.6666666666666667)");

    alg::OnlineCovariance c;
    EXPECT_EQ(std::format("{}", c), "OnlineCovariance(n=0, mean_x=0, mean_y=0, covariance=0, correlation=0)");
    c.add(1, 2);
    c.add(3, 6);
    const std::string cov = std::format("{}", c);
    EXPECT_TRUE(cov.starts_with("OnlineCovariance(n=2, mean_x=2, mean_y=4, covariance=2, correlation=")) << cov;
    EXPECT_TRUE(cov.ends_with(std::format("correlation={})", c.correlation()))) << cov;
}

TEST(AlgorithmsFormat, RateLimitersSamplersAndIndexes) {
    alg::LeakyBucket leaky(10, 1);
    EXPECT_EQ(std::format("{}", leaky), "LeakyBucket(capacity=10, leak_rate=1, level=0, last_time=0)");
    leaky.add(0.0, 4.0);
    leaky.add(2.0, 1.0);
    EXPECT_EQ(std::format("{}", leaky), "LeakyBucket(capacity=10, leak_rate=1, level=3, last_time=2)");

    alg::TokenBucket tokens(10, 1);
    EXPECT_EQ(std::format("{}", tokens), "TokenBucket(capacity=10, rate=1, tokens=10, last_time=0)");
    tokens.allow(0.0, 6.0);
    tokens.allow(2.0, 1.0);
    EXPECT_EQ(std::format("{}", tokens), "TokenBucket(capacity=10, rate=1, tokens=5, last_time=2)");

    const std::vector<std::vector<double>> data{{1, 0}, {0, 1}};
    EXPECT_EQ(std::format("{}", alg::CosineLSH(data, 2, 4)), "CosineLSH(dim=2, tables=2, bits=4)");
    EXPECT_EQ(std::format("{}", alg::Rc4("Key")), "Rc4");
    EXPECT_EQ(std::format("{}", alg::ZigguratNormal()), "ZigguratNormal(layers=256, tail=3.654152885361009)");
    EXPECT_EQ(std::format("{}", alg::ZigguratNormal(128)), "ZigguratNormal(layers=128, tail=3.442619855899)");
}

// ---- records built by real algorithm calls -------------------------------------------------------

TEST(AlgorithmsFormat, RecordsFromAlgorithms) {
    EXPECT_EQ(std::format("{}", alg::extended_gcd(240, 46)), "ExtendedGcd(gcd=2, x=-9, y=47)");
    EXPECT_EQ(std::format("{}", alg::computus_gregorian(2024)), "EasterDate(month=3, day=31)");

    const std::vector<std::tuple<std::size_t, std::size_t, double>> edges{{0, 1, 1.0}, {1, 2, 2.5}, {0, 2, 5.0}};
    EXPECT_EQ(std::format("{}", alg::dijkstra(3, edges, 0)),
              std::format("ShortestPathResult(distance=[0, 1, 3.5], predecessor=[{}, 0, 1])", static_cast<std::size_t>(-1)));
    EXPECT_EQ(std::format("{}", alg::kruskal(3, edges)),
              "MinimumSpanningTree(edges=[(0, 1, 1), (1, 2, 2.5)], total_weight=3.5, is_connected=true)");

    const auto runs = alg::rle_encode("aaab");
    ASSERT_EQ(runs.size(), 2u);
    EXPECT_EQ(std::format("{}", runs[0]), "Run(symbol='a', count=3)");
    EXPECT_EQ(std::format("{}", runs[1]), "Run(symbol='b', count=1)");
    EXPECT_EQ(std::format("{}", alg::bwt_transform("banana")), R"x(BWTResult(transformed="nnbaaa", index=3))x");

    const auto huff = alg::huffman_build("aab");
    const std::string h = std::format("{}", huff);
    EXPECT_TRUE(h.starts_with("HuffmanCode(codes={'a': ")) << h;
}

TEST(AlgorithmsFormat, LongAndNestedFieldsArePreviewed) {
    alg::BellmanFordResult r;
    for (int i = 0; i < 100; ++i) r.distance.push_back(i);
    EXPECT_EQ(std::format("{}", r),
              "BellmanFordResult(distance=[0, 1, 2, 3, 4, ..., 99] (n=100), predecessor=[], has_negative_cycle=false)");

    alg::OdeSolution ode{{0, 1}, {{1, 2}, {3, 4}}};
    EXPECT_EQ(std::format("{}", ode), "OdeSolution(t=[0, 1], y=[[1, 2], [3, 4]])");

    alg::UPGMAResult upgma;
    for (int i = 0; i < 5; ++i) upgma.merges.push_back({i, i + 1, 0.5, 10 + i});
    EXPECT_EQ(std::format("{}", upgma),
              "UPGMAResult(merges=[UPGMAMerge(left=0, right=1, height=0.5, id=10), "
              "UPGMAMerge(left=1, right=2, height=0.5, id=11), ..., "
              "UPGMAMerge(left=4, right=5, height=0.5, id=14)] (n=5))");

    alg::ShannonFanoCode sf;
    sf.codes = {{'b', "1"}, {'a', "0"}};
    EXPECT_EQ(std::format("{}", sf), R"x(ShannonFanoCode(codes={'a': "0", 'b': "1"}))x");

    alg::RansacLine line;
    line.inliers = {1, 0, 1};
    EXPECT_EQ(std::format("{}", line), "RansacLine(a=0, b=0, c=0, inliers=[1, 0, 1], inlier_count=0, trials=0)");

    alg::PellSolution pell;
    pell.x = static_cast<__int128>(1) << 100;
    pell.y = -7;
    EXPECT_EQ(std::format("{}", pell), "PellSolution(x=1267650600228229401496703205376, y=-7, found=false)");

    alg::Alignment al{3, std::string(100, 'A'), "AC"};
    EXPECT_EQ(std::format("{}", al), std::format(R"x(Alignment(score=3, a_aligned="{}..." (n=100), b_aligned="AC"))x",
                                                 std::string(57, 'A')));
}

// ---- every plain record struct: formattable, and its value-initialised text -------------------

static_assert(std::formattable<alg::DecisionStump, char>);
static_assert(std::formattable<alg::AdaBoostModel, char>);
static_assert(std::formattable<alg::MinCutResult, char>);
static_assert(std::formattable<alg::DirectedBranchingResult, char>);
static_assert(std::formattable<alg::AhoCorasickMatch, char>);
static_assert(std::formattable<alg::AlmeidaPinedaNetwork, char>);
static_assert(std::formattable<alg::AlmeidaPinedaResult, char>);
static_assert(std::formattable<alg::AlopexResult, char>);
static_assert(std::formattable<alg::AlopexParameters, char>);
static_assert(std::formattable<alg::Ac3Result, char>);
static_assert(std::formattable<alg::ArnoldiResult, char>);
static_assert(std::formattable<alg::FrequentItemset, char>);
static_assert(std::formattable<alg::AssociationRule, char>);
static_assert(std::formattable<alg::NFA, char>);
static_assert(std::formattable<alg::DFA, char>);
static_assert(std::formattable<alg::BakeryTicket, char>);
static_assert(std::formattable<alg::SafetyResult, char>);
static_assert(std::formattable<alg::BellmanFordResult, char>);
static_assert(std::formattable<alg::BerlekampMasseyResult, char>);
static_assert(std::formattable<alg::BirkhoffConstraint, char>);
static_assert(std::formattable<alg::HSP, char>);
static_assert(std::formattable<alg::GeneralMatchingResult, char>);
static_assert(std::formattable<alg::KnapsackResult, char>);
static_assert(std::formattable<alg::BullyResult, char>);
static_assert(std::formattable<alg::BWTResult, char>);
static_assert(std::formattable<alg::Canopy, char>);
static_assert(std::formattable<alg::CheneyResult, char>);
static_assert(std::formattable<alg::CrtSolution, char>);
static_assert(std::formattable<alg::BerkeleyResult, char>);
static_assert(std::formattable<alg::ClippedSegment, char>);
static_assert(std::formattable<alg::EasterDate, char>);
static_assert(std::formattable<alg::ConnectedComponents, char>);
static_assert(std::formattable<alg::SchedProcess, char>);
static_assert(std::formattable<alg::SchedMetrics, char>);
static_assert(std::formattable<alg::GanttSegment, char>);
static_assert(std::formattable<alg::ScheduleResult, char>);
static_assert(std::formattable<alg::CycleResult, char>);
static_assert(std::formattable<alg::CNFGrammar, char>);
static_assert(std::formattable<alg::DecisionTreeNode, char>);
static_assert(std::formattable<alg::DecisionTree, char>);
static_assert(std::formattable<alg::DeutschJozsaResult, char>);
static_assert(std::formattable<alg::DifferenceMapResult, char>);
static_assert(std::formattable<alg::ShortestPathResult, char>);
static_assert(std::formattable<alg::DivResultEx, char>);
static_assert(std::formattable<alg::SatResult, char>);
static_assert(std::formattable<alg::MaxFlowResult, char>);
static_assert(std::formattable<alg::BinSplitResult, char>);
static_assert(std::formattable<alg::DivResult, char>);
static_assert(std::formattable<alg::EllipsoidResult, char>);
static_assert(std::formattable<alg::ExactCoverResult, char>);
static_assert(std::formattable<alg::PancakeResult, char>);
static_assert(std::formattable<alg::ExtendedGcd, char>);
static_assert(std::formattable<alg::AllPairsShortestPaths, char>);
static_assert(std::formattable<alg::FrankWolfeResult, char>);
static_assert(std::formattable<alg::FuzzyCMeansResult, char>);
static_assert(std::formattable<alg::GameNode, char>);
static_assert(std::formattable<alg::GameResult, char>);
static_assert(std::formattable<alg::GaussNewtonResult, char>);
static_assert(std::formattable<alg::GenerationalResult, char>);
static_assert(std::formattable<alg::GoldenResult, char>);
static_assert(std::formattable<alg::ColoringResult, char>);
static_assert(std::formattable<alg::SubgraphMatch, char>);
static_assert(std::formattable<alg::Circle2D, char>);
static_assert(std::formattable<alg::GraspMaxCutResult, char>);
static_assert(std::formattable<alg::GroverResult, char>);
static_assert(std::formattable<alg::Hamming74Decoded, char>);
static_assert(std::formattable<alg::HiddenMarkovModel, char>);
static_assert(std::formattable<alg::ViterbiResult, char>);
static_assert(std::formattable<alg::ForwardBackwardResult, char>);
static_assert(std::formattable<alg::BaumWelchResult, char>);
static_assert(std::formattable<alg::BipartiteMatchingResult, char>);
static_assert(std::formattable<alg::HopfieldNetwork, char>);
static_assert(std::formattable<alg::HuffmanCode, char>);
static_assert(std::formattable<alg::HuffmanCode::Node, char>);
static_assert(std::formattable<alg::AssignmentResult, char>);
static_assert(std::formattable<alg::DiscreteLogResult, char>);
static_assert(std::formattable<alg::CubicSpline, char>);
static_assert(std::formattable<alg::KMedoidsResult, char>);
static_assert(std::formattable<alg::KabschResult, char>);
static_assert(std::formattable<alg::RewriteRule, char>);
static_assert(std::formattable<alg::KnuthBendixResult, char>);
static_assert(std::formattable<alg::LanczosResult, char>);
static_assert(std::formattable<alg::LBGResult, char>);
static_assert(std::formattable<alg::EigenPair, char>);
static_assert(std::formattable<alg::LinearSolution, char>);
static_assert(std::formattable<alg::IterativeSolution, char>);
static_assert(std::formattable<alg::GaussJordanResult, char>);
static_assert(std::formattable<alg::RegressionStump, char>);
static_assert(std::formattable<alg::LogitBoostModel, char>);
static_assert(std::formattable<alg::LZ77Token, char>);
static_assert(std::formattable<alg::PalindromeResult, char>);
static_assert(std::formattable<alg::MarkCompactResult, char>);
static_assert(std::formattable<alg::MarkSweepResult, char>);
static_assert(std::formattable<alg::MatrixChainResult, char>);
static_assert(std::formattable<alg::ParsimonyTree, char>);
static_assert(std::formattable<alg::MemeticMaxCutResult, char>);
static_assert(std::formattable<alg::NQueensResult, char>);
static_assert(std::formattable<alg::MiserResult, char>);
static_assert(std::formattable<alg::MinimumSpanningTree, char>);
static_assert(std::formattable<alg::BoothResult, char>);
static_assert(std::formattable<alg::NestedSamplingResult, char>);
static_assert(std::formattable<alg::NestingResult, char>);
static_assert(std::formattable<alg::HITSResult, char>);
static_assert(std::formattable<alg::ModSqrt, char>);
static_assert(std::formattable<alg::DiscreteLog, char>);
static_assert(std::formattable<alg::PellSolution, char>);
static_assert(std::formattable<alg::ObjectGraph, char>);
static_assert(std::formattable<alg::OddsResult, char>);
static_assert(std::formattable<alg::OdeSolution, char>);
static_assert(std::formattable<alg::VerletSolution, char>);
static_assert(std::formattable<alg::PerceptronModel, char>);
static_assert(std::formattable<alg::WinnowModel, char>);
static_assert(std::formattable<alg::OpticsResult, char>);
static_assert(std::formattable<alg::PegExpr, char>);
static_assert(std::formattable<alg::DepthRect, char>);
static_assert(std::formattable<alg::PCNNResult, char>);
static_assert(std::formattable<alg::PCNNParameters, char>);
static_assert(std::formattable<alg::YoungTableaux, char>);
static_assert(std::formattable<alg::PetrickResult, char>);
static_assert(std::formattable<alg::PhaseEstimationResult, char>);
static_assert(std::formattable<alg::MatchRatingResult, char>);
static_assert(std::formattable<alg::SinCos, char>);
static_assert(std::formattable<alg::KangarooResult, char>);
static_assert(std::formattable<alg::PolyDivMod, char>);
static_assert(std::formattable<alg::Implicant, char>);
static_assert(std::formattable<alg::QuineMcCluskeyResult, char>);
static_assert(std::formattable<alg::HillClimbResult, char>);
static_assert(std::formattable<alg::RansacLine, char>);
static_assert(std::formattable<alg::RBFNetwork, char>);
static_assert(std::formattable<alg::RTTask, char>);
static_assert(std::formattable<alg::RTSlice, char>);
static_assert(std::formattable<alg::RTScheduleResult, char>);
static_assert(std::formattable<alg::RefCountResult, char>);
static_assert(std::formattable<alg::RootResult, char>);
static_assert(std::formattable<alg::LongestPathResult, char>);
static_assert(std::formattable<alg::JohnsonAllPairsResult, char>);
static_assert(std::formattable<alg::ZeroRModel, char>);
static_assert(std::formattable<alg::OneRModel, char>);
static_assert(std::formattable<alg::Run, char>);
static_assert(std::formattable<alg::RVMModel, char>);
static_assert(std::formattable<alg::BSGS, char>);
static_assert(std::formattable<alg::UnweightedSearchResult, char>);
static_assert(std::formattable<alg::BasicPathResult, char>);
static_assert(std::formattable<alg::GridPathResult, char>);
static_assert(std::formattable<alg::PlanningAction, char>);
static_assert(std::formattable<alg::PlanningResult, char>);
static_assert(std::formattable<alg::SelfOrganizingMap, char>);
static_assert(std::formattable<alg::SOMParameters, char>);
static_assert(std::formattable<alg::AlignmentScoring, char>);
static_assert(std::formattable<alg::Alignment, char>);
static_assert(std::formattable<alg::SecretShare, char>);
static_assert(std::formattable<alg::ShannonFanoCode, char>);
static_assert(std::formattable<alg::SignedReversalResult, char>);
static_assert(std::formattable<alg::SimonResult, char>);
static_assert(std::formattable<alg::SimplexResult, char>);
static_assert(std::formattable<alg::SobelResult, char>);
static_assert(std::formattable<alg::SearchResult, char>);
static_assert(std::formattable<alg::StableMatchingResult, char>);
static_assert(std::formattable<alg::StoneResult, char>);
static_assert(std::formattable<alg::StructuredSVMModel, char>);
static_assert(std::formattable<alg::StructuredSVMParameters, char>);
static_assert(std::formattable<alg::ScoringSegment, char>);
static_assert(std::formattable<alg::SubsetSumResult, char>);
static_assert(std::formattable<alg::MaxSubarray, char>);
static_assert(std::formattable<alg::SuffixTreeNode, char>);
static_assert(std::formattable<alg::SuffixTree, char>);
static_assert(std::formattable<alg::StepResult, char>);
static_assert(std::formattable<alg::TabularEnvironment, char>);
static_assert(std::formattable<alg::RLParameters, char>);
static_assert(std::formattable<alg::QLearningResult, char>);
static_assert(std::formattable<alg::TDPredictionResult, char>);
static_assert(std::formattable<alg::TernaryResult, char>);
static_assert(std::formattable<alg::TopologicalSortResult, char>);
static_assert(std::formattable<alg::TourResult, char>);
static_assert(std::formattable<alg::VehicleRoutes, char>);
static_assert(std::formattable<alg::InformedSearchResult, char>);
static_assert(std::formattable<alg::UniformCostResult, char>);
static_assert(std::formattable<alg::UPGMAMerge, char>);
static_assert(std::formattable<alg::UPGMAResult, char>);
static_assert(std::formattable<alg::VectorClock, char>);
static_assert(std::formattable<alg::VincentyResult, char>);
static_assert(std::formattable<alg::WuPixel, char>);
static_assert(std::formattable<alg::ZobristTable, char>);

TEST(AlgorithmsFormat, ValueInitialisedRecords) {
    EXPECT_EQ(std::format("{}", alg::DecisionStump{}),
              R"x(DecisionStump(feature=0, threshold=0, polarity=1, alpha=0))x");
    EXPECT_EQ(std::format("{}", alg::AdaBoostModel{}),
              R"x(AdaBoostModel(stumps=[], training_error=[], weak_error=[]))x");
    EXPECT_EQ(std::format("{}", alg::MinCutResult{}),
              R"x(MinCutResult(cut_size=0, side_a=[], side_b=[]))x");
    EXPECT_EQ(std::format("{}", alg::DirectedBranchingResult{}),
              R"x(DirectedBranchingResult(total_weight=0, exists=false))x");
    EXPECT_EQ(std::format("{}", alg::AhoCorasickMatch{}),
              R"x(AhoCorasickMatch(position=0, pattern_index=0))x");
    EXPECT_EQ(std::format("{}", alg::AlmeidaPinedaNetwork{}),
              R"x(AlmeidaPinedaNetwork(num_units=0, num_inputs=0, output_units=[], W=[]))x");
    EXPECT_EQ(std::format("{}", alg::AlmeidaPinedaResult{}),
              R"x(AlmeidaPinedaResult(net=AlmeidaPinedaNetwork(num_units=0, num_inputs=0, output_units=[], W=[]), error_history=[]))x");
    EXPECT_EQ(std::format("{}", alg::AlopexResult{}),
              R"x(AlopexResult(best_parameters=[], best_cost=0, cost_history=[]))x");
    EXPECT_EQ(std::format("{}", alg::AlopexParameters{}),
              R"x(AlopexParameters(step=0.05, temperature=1, cooling=0.995, iterations=2000, seed=0))x");
    EXPECT_EQ(std::format("{}", alg::Ac3Result{}),
              R"x(Ac3Result(domains=[], consistent=true))x");
    EXPECT_EQ(std::format("{}", alg::ArnoldiResult{}),
              R"x(ArnoldiResult(V=[], H=[]))x");
    EXPECT_EQ(std::format("{}", alg::FrequentItemset{}),
              R"x(FrequentItemset(items=[], support=0))x");
    EXPECT_EQ(std::format("{}", alg::AssociationRule{}),
              R"x(AssociationRule(antecedent=[], consequent=[], support=0, confidence=0, lift=0))x");
    EXPECT_EQ(std::format("{}", alg::NFA{}),
              R"x(NFA(num_states=0, num_symbols=0, delta=[], start_states=[], accept=[]))x");
    EXPECT_EQ(std::format("{}", alg::DFA{}),
              R"x(DFA(num_states=0, num_symbols=0, delta=[], start=0, accept=[]))x");
    EXPECT_EQ(std::format("{}", alg::BakeryTicket{}),
              R"x(BakeryTicket(number=0, id=0))x");
    EXPECT_EQ(std::format("{}", alg::SafetyResult{}),
              R"x(SafetyResult(safe=false, sequence=[]))x");
    EXPECT_EQ(std::format("{}", alg::BellmanFordResult{}),
              R"x(BellmanFordResult(distance=[], predecessor=[], has_negative_cycle=false))x");
    EXPECT_EQ(std::format("{}", alg::BerlekampMasseyResult{}),
              R"x(BerlekampMasseyResult(length=0, polynomial=[]))x");
    EXPECT_EQ(std::format("{}", alg::BirkhoffConstraint{}),
              R"x(BirkhoffConstraint(node=0, order=0, value=0))x");
    EXPECT_EQ(std::format("{}", alg::HSP{}),
              R"x(HSP(query_start=0, db_start=0, length=0, score=0))x");
    EXPECT_EQ(std::format("{}", alg::GeneralMatchingResult{}),
              R"x(GeneralMatchingResult(mate=[], size=0))x");
    EXPECT_EQ(std::format("{}", alg::KnapsackResult{}),
              R"x(KnapsackResult(value=0, chosen=[], nodes_explored=0))x");
    EXPECT_EQ(std::format("{}", alg::BullyResult{}),
              R"x(BullyResult(leader=-1, election_messages=0, answer_messages=0, coordinator_messages=0))x");
    EXPECT_EQ(std::format("{}", alg::BWTResult{}),
              R"x(BWTResult(transformed="", index=0))x");
    EXPECT_EQ(std::format("{}", alg::Canopy{}),
              R"x(Canopy(center=0, members=[]))x");
    EXPECT_EQ(std::format("{}", alg::CheneyResult{}),
              R"x(CheneyResult(to_space_order=[], forwarding=[]))x");
    EXPECT_EQ(std::format("{}", alg::CrtSolution{}),
              R"x(CrtSolution(solvable=false, remainder=0, modulus=1))x");
    EXPECT_EQ(std::format("{}", alg::BerkeleyResult{}),
              R"x(BerkeleyResult(synchronized_time=0, adjustments=[]))x");
    EXPECT_EQ(std::format("{}", alg::ClippedSegment{}),
              R"x(ClippedSegment(visible=false, x0=0, y0=0, x1=0, y1=0))x");
    EXPECT_EQ(std::format("{}", alg::EasterDate{}),
              R"x(EasterDate(month=0, day=0))x");
    EXPECT_EQ(std::format("{}", alg::ConnectedComponents{}),
              R"x(ConnectedComponents(labels=[], count=0))x");
    EXPECT_EQ(std::format("{}", alg::SchedProcess{}),
              R"x(SchedProcess(id=0, arrival=0, burst=0))x");
    EXPECT_EQ(std::format("{}", alg::SchedMetrics{}),
              R"x(SchedMetrics(id=0, completion=0, turnaround=0, waiting=0))x");
    EXPECT_EQ(std::format("{}", alg::GanttSegment{}),
              R"x(GanttSegment(id=0, start=0, end=0))x");
    EXPECT_EQ(std::format("{}", alg::ScheduleResult{}),
              R"x(ScheduleResult(metrics=[], gantt=[], avg_waiting=0, avg_turnaround=0))x");
    EXPECT_EQ(std::format("{}", alg::CycleResult{}),
              R"x(CycleResult(lambda=0, mu=0))x");
    EXPECT_EQ(std::format("{}", alg::CNFGrammar{}),
              R"x(CNFGrammar(start=0, unary={}, binary=[]))x");
    EXPECT_EQ(std::format("{}", alg::DecisionTreeNode{}),
              R"x(DecisionTreeNode(leaf=true, prediction=0, feature=0, threshold=0, children={}, left=-1, right=-1))x");
    EXPECT_EQ(std::format("{}", alg::DecisionTree{}),
              R"x(DecisionTree(nodes=[]))x");
    EXPECT_EQ(std::format("{}", alg::DeutschJozsaResult{}),
              R"x(DeutschJozsaResult(constant=false, prob_all_zero=0))x");
    EXPECT_EQ(std::format("{}", alg::DifferenceMapResult{}),
              R"x(DifferenceMapResult(solution=[], iterations=0, residual=0, converged=false))x");
    EXPECT_EQ(std::format("{}", alg::ShortestPathResult{}),
              R"x(ShortestPathResult(distance=[], predecessor=[]))x");
    EXPECT_EQ(std::format("{}", alg::DivResultEx{}),
              R"x(DivResultEx(quotient=0, remainder=0, iterations=0))x");
    EXPECT_EQ(std::format("{}", alg::SatResult{}),
              R"x(SatResult(satisfiable=false, assignment=[]))x");
    EXPECT_EQ(std::format("{}", alg::MaxFlowResult{}),
              R"x(MaxFlowResult(max_flow=0, edge_flow=[], min_cut_source_side=[]))x");
    EXPECT_EQ(std::format("{}", alg::BinSplitResult{}),
              R"x(BinSplitResult(num=0, den=1, value=0))x");
    EXPECT_EQ(std::format("{}", alg::DivResult{}),
              R"x(DivResult(quotient=0, remainder=0, iterations=0))x");
    EXPECT_EQ(std::format("{}", alg::EllipsoidResult{}),
              R"x(EllipsoidResult(x=[], value=0, iterations=0))x");
    EXPECT_EQ(std::format("{}", alg::ExactCoverResult{}),
              R"x(ExactCoverResult(solved=false, rows=[]))x");
    EXPECT_EQ(std::format("{}", alg::PancakeResult{}),
              R"x(PancakeResult(sorted=[], flips=[]))x");
    EXPECT_EQ(std::format("{}", alg::ExtendedGcd{}),
              R"x(ExtendedGcd(gcd=0, x=0, y=0))x");
    EXPECT_EQ(std::format("{}", alg::AllPairsShortestPaths{}),
              R"x(AllPairsShortestPaths(distance=[], next=[], has_negative_cycle=false))x");
    EXPECT_EQ(std::format("{}", alg::FrankWolfeResult{}),
              R"x(FrankWolfeResult(x=[], objective=[]))x");
    EXPECT_EQ(std::format("{}", alg::FuzzyCMeansResult{}),
              R"x(FuzzyCMeansResult(centers=[], membership=[], objective=0, iterations=0))x");
    EXPECT_EQ(std::format("{}", alg::GameNode{}),
              R"x(GameNode(leaf=false, value=0, children=[]))x");
    EXPECT_EQ(std::format("{}", alg::GameResult{}),
              R"x(GameResult(value=0, leaves_evaluated=0, nodes_visited=0, cutoffs=0, best_child=-1))x");
    EXPECT_EQ(std::format("{}", alg::GaussNewtonResult{}),
              R"x(GaussNewtonResult(params=[], iterations=0, sse=0))x");
    EXPECT_EQ(std::format("{}", alg::GenerationalResult{}),
              R"x(GenerationalResult(promoted=[], young_freed=[], remembered_set=[], minor_scanned=0, full_scanned=0))x");
    EXPECT_EQ(std::format("{}", alg::GoldenResult{}),
              R"x(GoldenResult(argmax=0, value=0, iterations=0))x");
    EXPECT_EQ(std::format("{}", alg::ColoringResult{}),
              R"x(ColoringResult(color=[], num_colors=0))x");
    EXPECT_EQ(std::format("{}", alg::SubgraphMatch{}),
              R"x(SubgraphMatch(found=false, mapping=[]))x");
    EXPECT_EQ(std::format("{}", alg::Circle2D{}),
              R"x(Circle2D(center=Point2D(0, 0), radius=1))x");
    EXPECT_EQ(std::format("{}", alg::GraspMaxCutResult{}),
              R"x(GraspMaxCutResult(value=0, side=[]))x");
    EXPECT_EQ(std::format("{}", alg::GroverResult{}),
              R"x(GroverResult(measured=0, success_prob=0, iterations=0))x");
    EXPECT_EQ(std::format("{}", alg::Hamming74Decoded{}),
              R"x(Hamming74Decoded(data=[0, 0, 0, 0], error_position=0, corrected=false))x");
    EXPECT_EQ(std::format("{}", alg::HiddenMarkovModel{}),
              R"x(HiddenMarkovModel(initial=[], transition=[], emission=[]))x");
    EXPECT_EQ(std::format("{}", alg::ViterbiResult{}),
              R"x(ViterbiResult(states=[], log_probability=0))x");
    EXPECT_EQ(std::format("{}", alg::ForwardBackwardResult{}),
              R"x(ForwardBackwardResult(alpha=[], beta=[], gamma=[], scale=[], log_likelihood=0))x");
    EXPECT_EQ(std::format("{}", alg::BaumWelchResult{}),
              R"x(BaumWelchResult(model=HiddenMarkovModel(initial=[], transition=[], emission=[]), log_likelihood_history=[], iterations=0))x");
    EXPECT_EQ(std::format("{}", alg::BipartiteMatchingResult{}),
              R"x(BipartiteMatchingResult(match_left=[], match_right=[], size=0))x");
    EXPECT_EQ(std::format("{}", alg::HopfieldNetwork{}),
              R"x(HopfieldNetwork(weights=[]))x");
    EXPECT_EQ(std::format("{}", alg::HuffmanCode{}),
              R"x(HuffmanCode(codes={}, tree=[], root=-1))x");
    EXPECT_EQ(std::format("{}", alg::HuffmanCode::Node{}),
              R"x(HuffmanCode::Node(symbol=0, leaf=false, left=-1, right=-1))x");
    EXPECT_EQ(std::format("{}", alg::AssignmentResult{}),
              R"x(AssignmentResult(assignment=[], cost=0))x");
    EXPECT_EQ(std::format("{}", alg::DiscreteLogResult{}),
              R"x(DiscreteLogResult(x=0, found=false))x");
    EXPECT_EQ(std::format("{}", alg::CubicSpline{}),
              R"x(CubicSpline(xs=[], a=[], b=[], c=[], d=[]))x");
    EXPECT_EQ(std::format("{}", alg::KMedoidsResult{}),
              R"x(KMedoidsResult(medoids=[], assignment=[], total_cost=0, iterations=0))x");
    EXPECT_EQ(std::format("{}", alg::KabschResult{}),
              R"x(KabschResult(rotation=[[0, 0, 0], [0, 0, 0], [0, 0, 0]], translation=[0, 0, 0], rmsd=0))x");
    EXPECT_EQ(std::format("{}", alg::RewriteRule{}),
              R"x(RewriteRule(lhs="", rhs=""))x");
    EXPECT_EQ(std::format("{}", alg::KnuthBendixResult{}),
              R"x(KnuthBendixResult(rules=[], complete=false))x");
    EXPECT_EQ(std::format("{}", alg::LanczosResult{}),
              R"x(LanczosResult(alpha=[], beta=[], V=[]))x");
    EXPECT_EQ(std::format("{}", alg::LBGResult{}),
              R"x(LBGResult(codebook=[], assignment=[], distortion=0, distortion_history=[]))x");
    EXPECT_EQ(std::format("{}", alg::EigenPair{}),
              R"x(EigenPair(value=0, vector=[]))x");
    EXPECT_EQ(std::format("{}", alg::LinearSolution{}),
              R"x(LinearSolution(solved=false, x=[]))x");
    EXPECT_EQ(std::format("{}", alg::IterativeSolution{}),
              R"x(IterativeSolution(converged=false, x=[], iterations=0, residual=0))x");
    EXPECT_EQ(std::format("{}", alg::GaussJordanResult{}),
              R"x(GaussJordanResult(solved=false, x=[], inverse=[]))x");
    EXPECT_EQ(std::format("{}", alg::RegressionStump{}),
              R"x(RegressionStump(feature=0, threshold=0, left_value=0, right_value=0))x");
    EXPECT_EQ(std::format("{}", alg::LogitBoostModel{}),
              R"x(LogitBoostModel(stumps=[], log_likelihood=[]))x");
    EXPECT_EQ(std::format("{}", alg::LZ77Token{}),
              R"x(LZ77Token(offset=0, length=0, next=0, literal=true))x");
    EXPECT_EQ(std::format("{}", alg::PalindromeResult{}),
              R"x(PalindromeResult(start=0, length=0))x");
    EXPECT_EQ(std::format("{}", alg::MarkCompactResult{}),
              R"x(MarkCompactResult(marked=[], new_address=[], order=[], updated_refs=[]))x");
    EXPECT_EQ(std::format("{}", alg::MarkSweepResult{}),
              R"x(MarkSweepResult(marked=[], live=[], freed=[]))x");
    EXPECT_EQ(std::format("{}", alg::MatrixChainResult{}),
              R"x(MatrixChainResult(cost=0, parenthesization=""))x");
    EXPECT_EQ(std::format("{}", alg::ParsimonyTree{}),
              R"x(ParsimonyTree(root=0, children=[]))x");
    EXPECT_EQ(std::format("{}", alg::MemeticMaxCutResult{}),
              R"x(MemeticMaxCutResult(value=0, side=[]))x");
    EXPECT_EQ(std::format("{}", alg::NQueensResult{}),
              R"x(NQueensResult(queens=[], solved=false, steps=0))x");
    EXPECT_EQ(std::format("{}", alg::MiserResult{}),
              R"x(MiserResult(value=0, variance=0))x");
    EXPECT_EQ(std::format("{}", alg::MinimumSpanningTree{}),
              R"x(MinimumSpanningTree(edges=[], total_weight=0, is_connected=false))x");
    EXPECT_EQ(std::format("{}", alg::BoothResult{}),
              R"x(BoothResult(product=0, additions=0, subtractions=0))x");
    EXPECT_EQ(std::format("{}", alg::NestedSamplingResult{}),
              R"x(NestedSamplingResult(log_evidence=-inf, log_evidence_error=0, information=0, iterations=0, posterior_samples=[], posterior_log_weights=[]))x");
    EXPECT_EQ(std::format("{}", alg::NestingResult{}),
              R"x(NestingResult(bins=0, assignment=[], loads=[]))x");
    EXPECT_EQ(std::format("{}", alg::HITSResult{}),
              R"x(HITSResult(hubs=[], authorities=[], iterations=0))x");
    EXPECT_EQ(std::format("{}", alg::ModSqrt{}),
              R"x(ModSqrt(exists=false, root=0))x");
    EXPECT_EQ(std::format("{}", alg::DiscreteLog{}),
              R"x(DiscreteLog(exists=false, exponent=0))x");
    EXPECT_EQ(std::format("{}", alg::PellSolution{}),
              R"x(PellSolution(x=0, y=0, found=false))x");
    EXPECT_EQ(std::format("{}", alg::ObjectGraph{}),
              R"x(ObjectGraph(num_objects=0, refs=[], roots=[], sizes=[]))x");
    EXPECT_EQ(std::format("{}", alg::OddsResult{}),
              R"x(OddsResult(threshold=0, win_probability=0))x");
    EXPECT_EQ(std::format("{}", alg::OdeSolution{}),
              R"x(OdeSolution(t=[], y=[]))x");
    EXPECT_EQ(std::format("{}", alg::VerletSolution{}),
              R"x(VerletSolution(t=[], x=[], v=[]))x");
    EXPECT_EQ(std::format("{}", alg::PerceptronModel{}),
              R"x(PerceptronModel(weights=[], bias=0, epochs_run=0, converged=false, total_mistakes=0))x");
    EXPECT_EQ(std::format("{}", alg::WinnowModel{}),
              R"x(WinnowModel(weights=[], threshold=0, alpha=2, epochs_run=0, converged=false, total_mistakes=0))x");
    EXPECT_EQ(std::format("{}", alg::OpticsResult{}),
              R"x(OpticsResult(ordering=[], reachability=[], core_distance=[]))x");
    EXPECT_EQ(std::format("{}", alg::PegExpr{}),
              R"x(PegExpr(op=Literal, text="", children=[]))x");
    EXPECT_EQ(std::format("{}", alg::DepthRect{}),
              R"x(DepthRect(x0=0, y0=0, x1=0, y1=0, depth=0, color=0))x");
    EXPECT_EQ(std::format("{}", alg::PCNNResult{}),
              R"x(PCNNResult(pulses=[], time_signature=[], first_fire=[]))x");
    EXPECT_EQ(std::format("{}", alg::PCNNParameters{}),
              R"x(PCNNParameters(iterations=20, beta=0.2, threshold_decay=0.8, threshold_gain=20, initial_threshold=1))x");
    EXPECT_EQ(std::format("{}", alg::YoungTableaux{}),
              R"x(YoungTableaux(p=[], q=[]))x");
    EXPECT_EQ(std::format("{}", alg::PetrickResult{}),
              R"x(PetrickResult(minimal_covers=[]))x");
    EXPECT_EQ(std::format("{}", alg::PhaseEstimationResult{}),
              R"x(PhaseEstimationResult(estimate=0, measured=0, probability=0))x");
    EXPECT_EQ(std::format("{}", alg::MatchRatingResult{}),
              R"x(MatchRatingResult(comparable=false, similarity=0, min_rating=0, match=false))x");
    EXPECT_EQ(std::format("{}", alg::SinCos{}),
              R"x(SinCos(sin=0, cos=0))x");
    EXPECT_EQ(std::format("{}", alg::KangarooResult{}),
              R"x(KangarooResult(x=0, found=false, jumps=0))x");
    EXPECT_EQ(std::format("{}", alg::PolyDivMod{}),
              R"x(PolyDivMod(quotient=[], remainder=[]))x");
    EXPECT_EQ(std::format("{}", alg::Implicant{}),
              R"x(Implicant(bits=0, mask=0))x");
    EXPECT_EQ(std::format("{}", alg::QuineMcCluskeyResult{}),
              R"x(QuineMcCluskeyResult(prime_implicants=[], essential=[], minimal_cover=[]))x");
    EXPECT_EQ(std::format("{}", alg::HillClimbResult{}),
              R"x(HillClimbResult(x=[], value=0))x");
    EXPECT_EQ(std::format("{}", alg::RansacLine{}),
              R"x(RansacLine(a=0, b=0, c=0, inliers=[], inlier_count=0, trials=0))x");
    EXPECT_EQ(std::format("{}", alg::RBFNetwork{}),
              R"x(RBFNetwork(centers=[], width=1, weights=[], bias=0))x");
    EXPECT_EQ(std::format("{}", alg::RTTask{}),
              R"x(RTTask(id=0, period=0, wcet=0))x");
    EXPECT_EQ(std::format("{}", alg::RTSlice{}),
              R"x(RTSlice(id=0, start=0, end=0))x");
    EXPECT_EQ(std::format("{}", alg::RTScheduleResult{}),
              R"x(RTScheduleResult(timeline=[], deadline_missed=false))x");
    EXPECT_EQ(std::format("{}", alg::RefCountResult{}),
              R"x(RefCountResult(refcount=[], freed=[], leaked=[]))x");
    EXPECT_EQ(std::format("{}", alg::RootResult{}),
              R"x(RootResult(converged=false, root=0, iterations=0, residual=0))x");
    EXPECT_EQ(std::format("{}", alg::LongestPathResult{}),
              R"x(LongestPathResult(distance=[], predecessor=[]))x");
    EXPECT_EQ(std::format("{}", alg::JohnsonAllPairsResult{}),
              R"x(JohnsonAllPairsResult(distance=[], has_negative_cycle=false))x");
    EXPECT_EQ(std::format("{}", alg::ZeroRModel{}),
              R"x(ZeroRModel(prediction=0, majority_count=0, total=0, accuracy=0))x");
    EXPECT_EQ(std::format("{}", alg::OneRModel{}),
              R"x(OneRModel(feature=0, value_to_class={}, default_class=0, errors=0, total=0, accuracy=0))x");
    EXPECT_EQ(std::format("{}", alg::Run{}),
              R"x(Run(symbol=0, count=0))x");
    EXPECT_EQ(std::format("{}", alg::RVMModel{}),
              R"x(RVMModel(centers=[], weights=[], relevant=[], gamma=1, beta=1, iterations=0, num_relevance_vectors=0))x");
    EXPECT_EQ(std::format("{}", alg::BSGS{}),
              R"x(BSGS(n=0, base=[], transversal=[], order=1))x");
    EXPECT_EQ(std::format("{}", alg::UnweightedSearchResult{}),
              R"x(UnweightedSearchResult(path=[], order=[], found=false))x");
    EXPECT_EQ(std::format("{}", alg::BasicPathResult{}),
              R"x(BasicPathResult(path=[], expanded=0, found=false))x");
    EXPECT_EQ(std::format("{}", alg::GridPathResult{}),
              R"x(GridPathResult(path=[], cost=0, found=false))x");
    EXPECT_EQ(std::format("{}", alg::PlanningAction{}),
              R"x(PlanningAction(name="", require=0, add=0, remove=0))x");
    EXPECT_EQ(std::format("{}", alg::PlanningResult{}),
              R"x(PlanningResult(actions=[], final_state=0, found=false))x");
    EXPECT_EQ(std::format("{}", alg::SelfOrganizingMap{}),
              R"x(SelfOrganizingMap(rows=0, cols=0, dim=0, weights=[]))x");
    EXPECT_EQ(std::format("{}", alg::SOMParameters{}),
              R"x(SOMParameters(rows=5, cols=5, iterations=1000, learning_rate=0.5, radius=0, seed=0))x");
    EXPECT_EQ(std::format("{}", alg::AlignmentScoring{}),
              R"x(AlignmentScoring(match=2, mismatch=-1, gap=-2))x");
    EXPECT_EQ(std::format("{}", alg::Alignment{}),
              R"x(Alignment(score=0, a_aligned="", b_aligned=""))x");
    EXPECT_EQ(std::format("{}", alg::SecretShare{}),
              R"x(SecretShare(x=0, y=0))x");
    EXPECT_EQ(std::format("{}", alg::ShannonFanoCode{}),
              R"x(ShannonFanoCode(codes={}))x");
    EXPECT_EQ(std::format("{}", alg::SignedReversalResult{}),
              R"x(SignedReversalResult(distance=0, reversals=[]))x");
    EXPECT_EQ(std::format("{}", alg::SimonResult{}),
              R"x(SimonResult(recovered_period=0, observed_y=[]))x");
    EXPECT_EQ(std::format("{}", alg::SimplexResult{}),
              R"x(SimplexResult(status=Optimal, value=0, x=[]))x");
    EXPECT_EQ(std::format("{}", alg::SobelResult{}),
              R"x(SobelResult(gx=[], gy=[], magnitude=[]))x");
    EXPECT_EQ(std::format("{}", alg::SearchResult{}),
              R"x(SearchResult(found=false, index=-1, probes=0))x");
    EXPECT_EQ(std::format("{}", alg::StableMatchingResult{}),
              R"x(StableMatchingResult(proposer_match=[], reviewer_match=[], proposals=0))x");
    EXPECT_EQ(std::format("{}", alg::StoneResult{}),
              R"x(StoneResult(x=[], iterations=0, residual=0))x");
    EXPECT_EQ(std::format("{}", alg::StructuredSVMModel{}),
              R"x(StructuredSVMModel(weights=[], num_classes=0, dim=0))x");
    EXPECT_EQ(std::format("{}", alg::StructuredSVMParameters{}),
              R"x(StructuredSVMParameters(lambda=0.01, epochs=50, seed=0))x");
    EXPECT_EQ(std::format("{}", alg::ScoringSegment{}),
              R"x(ScoringSegment(begin=0, end=0, score=0))x");
    EXPECT_EQ(std::format("{}", alg::SubsetSumResult{}),
              R"x(SubsetSumResult(found=false, indices=[]))x");
    EXPECT_EQ(std::format("{}", alg::MaxSubarray{}),
              R"x(MaxSubarray(sum=0, begin=0, end=0))x");
    EXPECT_EQ(std::format("{}", alg::SuffixTreeNode{}),
              R"x(SuffixTreeNode(start=-1, end=-1, suffix_link=-1, children={}))x");
    EXPECT_EQ(std::format("{}", alg::SuffixTree{}),
              R"x(SuffixTree(text="", nodes=[]))x");
    EXPECT_EQ(std::format("{}", alg::StepResult{}),
              R"x(StepResult(next_state=0, reward=0, terminal=false))x");
    EXPECT_EQ(std::format("{}", alg::TabularEnvironment{}),
              R"x(TabularEnvironment(num_states=0, num_actions=0))x");
    EXPECT_EQ(std::format("{}", alg::RLParameters{}),
              R"x(RLParameters(learning_rate=0.1, discount=0.99, epsilon=0.1, episodes=500, max_steps=1000, seed=0))x");
    EXPECT_EQ(std::format("{}", alg::QLearningResult{}),
              R"x(QLearningResult(q=[], greedy_policy=[], episode_returns=[]))x");
    EXPECT_EQ(std::format("{}", alg::TDPredictionResult{}),
              R"x(TDPredictionResult(value=[], episode_returns=[]))x");
    EXPECT_EQ(std::format("{}", alg::TernaryResult{}),
              R"x(TernaryResult(x=0, value=0, iterations=0))x");
    EXPECT_EQ(std::format("{}", alg::TopologicalSortResult{}),
              R"x(TopologicalSortResult(order=[], is_dag=true))x");
    EXPECT_EQ(std::format("{}", alg::TourResult{}),
              R"x(TourResult(tour=[], cost=0))x");
    EXPECT_EQ(std::format("{}", alg::VehicleRoutes{}),
              R"x(VehicleRoutes(routes=[], total_distance=0))x");
    EXPECT_EQ(std::format("{}", alg::InformedSearchResult{}),
              R"x(InformedSearchResult(cost=0, path=[], expanded=0, found=false))x");
    EXPECT_EQ(std::format("{}", alg::UniformCostResult{}),
              R"x(UniformCostResult(cost=0, path=[], found=false, expanded=0))x");
    EXPECT_EQ(std::format("{}", alg::UPGMAMerge{}),
              R"x(UPGMAMerge(left=0, right=0, height=0, id=0))x");
    EXPECT_EQ(std::format("{}", alg::UPGMAResult{}),
              R"x(UPGMAResult(merges=[]))x");
    EXPECT_EQ(std::format("{}", alg::VectorClock{}),
              R"x(VectorClock(v=[]))x");
    EXPECT_EQ(std::format("{}", alg::VincentyResult{}),
              R"x(VincentyResult(distance_m=0, iterations=0, converged=false))x");
    EXPECT_EQ(std::format("{}", alg::WuPixel{}),
              R"x(WuPixel(x=0, y=0, brightness=0))x");
    EXPECT_EQ(std::format("{}", alg::ZobristTable{}),
              R"x(ZobristTable(num_cells=0, num_pieces=0, keys=[]))x");
}
