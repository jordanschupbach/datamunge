#include <datamunge/algorithms/cheney.hpp>
#include <gtest/gtest.h>
#include <algorithm>
#include <deque>
#include <limits>
#include <numeric>
#include <random>
namespace {
namespace a=datamunge::algorithms;
a::ObjectGraph scrambled() {return {6,{{2},{3},{0},{4},{1,3},{}},{4,4},{2,3,1,4,2,5}};}
struct Oracle {std::vector<int> order,forward;};
Oracle queue_oracle(const a::ObjectGraph& g) {
  Oracle result;result.forward.assign(g.num_objects,-1);std::deque<int> queue;
  const auto discover=[&](int old) {
    if(result.forward[old]<0) {result.forward[old]=static_cast<int>(result.order.size());result.order.push_back(old);queue.push_back(old);}
  };
  for(int root:g.roots)discover(root);
  while(!queue.empty()) {const int old=queue.front();queue.pop_front();for(int child:g.refs[old])discover(child);}
  return result;
}
void verify(const a::ObjectGraph& g) {
  const auto expected=queue_oracle(g);const auto actual=a::cheney_copy(g);const auto copy=a::cheney_evacuate(g);
  const auto trace=a::cheney_copy_trace(g);const auto& r=trace.copy;
  EXPECT_EQ(actual.to_space_order,expected.order);EXPECT_EQ(actual.forwarding,expected.forward);
  EXPECT_EQ(copy.relocation.to_space_order,expected.order);EXPECT_EQ(r.relocation.forwarding,expected.forward);
  EXPECT_EQ(copy.heap.refs,r.heap.refs);EXPECT_EQ(copy.heap.roots,r.heap.roots);EXPECT_EQ(copy.offsets,r.offsets);
  EXPECT_EQ(r.heap.num_objects,static_cast<int>(expected.order.size()));EXPECT_EQ(r.heap.refs.size(),expected.order.size());
  EXPECT_EQ(r.work.validation.rows,g.refs.size());EXPECT_EQ(r.work.validation.roots,g.roots.size());EXPECT_EQ(r.work.validation.sizes,g.sizes.size());
  std::size_t total_edges=0,live_edges=0,units=0;
  for(const auto& row:g.refs)total_edges+=row.size();
  EXPECT_EQ(r.work.validation.references,total_edges);
  for(std::size_t slot=0;slot<expected.order.size();++slot) {
    const int old=expected.order[slot];const int size=static_cast<std::size_t>(old)<g.sizes.size()?g.sizes[old]:1;
    EXPECT_EQ(r.offsets[slot],units);EXPECT_EQ(r.heap.sizes[slot],size);units+=size;
    std::vector<int> refs;for(int child:g.refs[old])refs.push_back(expected.forward[child]);
    EXPECT_EQ(r.heap.refs[slot],refs);live_edges+=g.refs[old].size();
  }
  std::vector<int> roots;for(int old:g.roots)roots.push_back(expected.forward[old]);EXPECT_EQ(r.heap.roots,roots);
  EXPECT_EQ(r.used_units,units);EXPECT_EQ(r.work.root_probes,g.roots.size());EXPECT_EQ(r.work.reference_probes,live_edges);
  EXPECT_EQ(r.work.scans,expected.order.size());EXPECT_EQ(r.work.copies,expected.order.size());
  EXPECT_EQ(r.work.reuses,g.roots.size()+live_edges-expected.order.size());
  ASSERT_EQ(trace.events.size(),g.roots.size()+live_edges+expected.order.size());
  // Replay only the recorded decisions, independently checking exact event history.
  std::vector<int> forwarded(g.num_objects,-1),order;std::size_t scan=0,used=0,root_index=0,field=0,scan_free=0,scan_used=0;bool in_scans=false;
  for(const auto& e:trace.events) {
    EXPECT_EQ(e.scan,scan);
    if(e.phase!=a::CheneyPhase::Root && !in_scans) {in_scans=true;scan_free=order.size();scan_used=used;}
    if(e.phase==a::CheneyPhase::Scan) {
      ASSERT_LT(scan,order.size());EXPECT_EQ(e.source,order[scan]);EXPECT_EQ(e.target,order[scan]);EXPECT_EQ(e.destination,static_cast<int>(scan));
      EXPECT_EQ(field,g.refs[order[scan]].size());EXPECT_FALSE(e.copied);EXPECT_EQ(e.field,0u);
      EXPECT_EQ(e.free_before,scan_free);EXPECT_EQ(e.free_after,order.size());EXPECT_EQ(e.used_before,scan_used);EXPECT_EQ(e.used_after,used);
      ++scan;field=0;scan_free=order.size();scan_used=used;
    } else {
      if(e.phase==a::CheneyPhase::Root) {
        EXPECT_FALSE(in_scans);EXPECT_EQ(e.source,-1);EXPECT_EQ(e.field,root_index);ASSERT_LT(root_index,g.roots.size());EXPECT_EQ(e.target,g.roots[root_index++]);
      } else {
        ASSERT_LT(scan,order.size());EXPECT_EQ(e.source,order[scan]);EXPECT_EQ(e.field,field);ASSERT_LT(field,g.refs[order[scan]].size());EXPECT_EQ(e.target,g.refs[order[scan]][field++]);
      }
      EXPECT_EQ(e.free_before,order.size());EXPECT_EQ(e.used_before,used);
      const bool fresh=forwarded[e.target]<0;EXPECT_EQ(e.copied,fresh);
      if(fresh) {forwarded[e.target]=static_cast<int>(order.size());order.push_back(e.target);used+=static_cast<std::size_t>(e.target)<g.sizes.size()?g.sizes[e.target]:1;}
      EXPECT_EQ(e.destination,forwarded[e.target]);EXPECT_EQ(e.free_after,order.size());EXPECT_EQ(e.used_after,used);
    }
  }
  EXPECT_EQ(root_index,g.roots.size());EXPECT_EQ(scan,expected.order.size());EXPECT_EQ(order,expected.order);EXPECT_EQ(forwarded,expected.forward);
  const auto again=a::cheney_evacuate(r.heap,units);
  EXPECT_EQ(again.heap.refs,r.heap.refs);EXPECT_EQ(again.heap.roots,r.heap.roots);EXPECT_EQ(again.offsets,r.offsets);
}
}
TEST(CheneyTrace, OriginalNineObjectTreePreservesLegacyLayout) {
  const a::ObjectGraph g{9,{{1,2},{3,4},{5,6},{},{},{},{},{8},{}},{0},{}};
  verify(g);const auto r=a::cheney_copy(g);EXPECT_EQ(r.to_space_order,(std::vector<int>{0,1,2,3,4,5,6}));
  EXPECT_EQ(r.forwarding,(std::vector<int>{0,1,2,3,4,5,6,-1,-1}));
}
TEST(CheneyTrace, ScrambledCycleSharingDuplicateRootsAndWeightedOffsetsAreActual) {
  const auto g=scrambled();verify(g);const auto t=a::cheney_copy_trace(g,9,9);
  EXPECT_EQ(t.copy.relocation.to_space_order,(std::vector<int>{4,1,3}));
  EXPECT_EQ(t.copy.relocation.forwarding,(std::vector<int>{-1,1,-1,2,0,-1}));
  EXPECT_EQ(t.copy.heap.refs,(std::vector<std::vector<int>>{{1,2},{2},{0}}));
  EXPECT_EQ(t.copy.heap.roots,(std::vector<int>{0,0}));EXPECT_EQ(t.copy.offsets,(std::vector<std::size_t>{0,2,5}));
  EXPECT_EQ(t.copy.used_units,9u);EXPECT_EQ(t.copy.work.reference_probes,4u);EXPECT_EQ(t.copy.work.reuses,3u);EXPECT_EQ(t.events.size(),9u);
}
TEST(CheneyTrace, AllSmallDirectedGraphsAndOrderedRootPairsMatchQueueAndClosureOracles) {
  std::size_t cases=0;
  for(int n=0;n<=3;++n)for(unsigned bits=0;bits<(1u<<(n*n));++bits) {
    a::ObjectGraph g;g.num_objects=n;g.refs.resize(n);g.sizes.resize(n);std::iota(g.sizes.begin(),g.sizes.end(),1);
    std::vector<std::vector<bool>> closure(n,std::vector<bool>(n));
    for(int i=0;i<n;++i)for(int j=0;j<n;++j)if(bits&(1u<<(i*n+j))){g.refs[i].push_back(j);closure[i][j]=true;}
    for(int i=0;i<n;++i)closure[i][i]=true;
    for(int k=0;k<n;++k)for(int i=0;i<n;++i)for(int j=0;j<n;++j)closure[i][j]=closure[i][j]||(closure[i][k]&&closure[k][j]);
    std::vector<std::vector<int>> roots{{}};for(int i=0;i<n;++i)roots.push_back({i});for(int i=0;i<n;++i)for(int j=0;j<n;++j)roots.push_back({i,j});
    for(const auto& r:roots) {
      g.roots=r;verify(g);const auto copy=a::cheney_copy(g);const auto marks=a::reachable_from_roots(g);
      for(int old=0;old<n;++old) {bool live=false;for(int root:r)live=live||closure[root][old];EXPECT_EQ(copy.forwarding[old]>=0,live);EXPECT_EQ(marks[old]!=0,live);}
      ++cases;
    }
  }
  EXPECT_EQ(cases,6775u);
}
TEST(CheneyTrace, ThousandRandomMultigraphsPreserveEveryReferenceSlotAndSizePrefix) {
  std::mt19937 random(1970);
  for(int trial=0;trial<1000;++trial) {
    const int n=1+random()%64;a::ObjectGraph g;g.num_objects=n;g.refs.resize(n);
    for(auto& row:g.refs)for(unsigned i=0,count=random()%12;i<count;++i)row.push_back(random()%n);
    for(unsigned i=0,count=random()%10;i<count;++i)g.roots.push_back(random()%n);
    for(unsigned i=0,count=random()%(n+1);i<count;++i)g.sizes.push_back(1+random()%50);
    verify(g);const auto full=a::cheney_evacuate(g);
    EXPECT_EQ(a::cheney_evacuate(g,full.used_units).heap.refs,full.heap.refs);
    if(full.used_units) {EXPECT_THROW(a::cheney_evacuate(g,full.used_units-1),std::length_error);}
  }
}
TEST(CheneyTrace, CapacityAndExactEventLimitsReturnNoPartialResultOrInputMutation) {
  const auto g=scrambled();const auto before=g;
  EXPECT_THROW(a::cheney_evacuate(g,8),std::length_error);
  EXPECT_THROW(a::cheney_copy_trace(g,9,8),std::length_error);
  EXPECT_THROW(a::cheney_copy_trace(g,0),std::length_error);
  EXPECT_EQ(a::cheney_copy_trace(g,9,9).events.size(),9u);
  EXPECT_EQ(a::cheney_evacuate(g,10).used_units,9u);
  EXPECT_EQ(g.refs,before.refs);EXPECT_EQ(g.roots,before.roots);EXPECT_EQ(g.sizes,before.sizes);EXPECT_EQ(g.num_objects,before.num_objects);
  EXPECT_EQ(a::cheney_copy(g).to_space_order,(std::vector<int>{4,1,3}));
}
TEST(CheneyTrace, DefaultAndGarbageOnlyHeapsNeedZeroCapacityButStillValidate) {
  a::ObjectGraph empty;EXPECT_EQ(empty.num_objects,0);verify(empty);
  EXPECT_TRUE(a::cheney_copy_trace(empty,0,1).events.empty());EXPECT_TRUE(a::reachable_from_roots(empty).empty());
  const a::ObjectGraph garbage{3,{{1},{0},{2}},{},{2,3,4}};verify(garbage);
  const auto t=a::cheney_copy_trace(garbage,0,1);EXPECT_EQ(t.copy.used_units,0u);EXPECT_EQ(t.copy.work.validation.references,3u);EXPECT_EQ(t.copy.work.reference_probes,0u);
}
TEST(CheneyTrace, PartialSizesDefaultToOneAndLegacyCopyIgnoresUnusedSizeMetadata) {
  a::ObjectGraph g{3,{{1},{2},{}},{0},{7}};verify(g);
  EXPECT_EQ(a::cheney_evacuate(g,9).heap.sizes,(std::vector<int>{7,1,1}));
  g.sizes={0,-1,0,999};EXPECT_EQ(a::cheney_copy(g).to_space_order,(std::vector<int>{0,1,2}));
  EXPECT_EQ(a::reachable_from_roots(g),(std::vector<char>{1,1,1}));
  EXPECT_THROW(a::cheney_evacuate(g),std::invalid_argument);
}
TEST(CheneyTrace, DuplicateFieldsAndRootSlotsAreRetainedRatherThanDeduplicated) {
  const a::ObjectGraph g{2,{{1,1,0},{0,1}},{0,0,1},{}};verify(g);
  const auto c=a::cheney_evacuate(g);EXPECT_EQ(c.heap.roots,(std::vector<int>{0,0,1}));EXPECT_EQ(c.heap.refs,g.refs);
  EXPECT_EQ(c.work.root_probes,3u);EXPECT_EQ(c.work.reference_probes,5u);EXPECT_EQ(c.work.reuses,6u);
}
TEST(CheneyTrace, RootAndFieldOrderChooseDiscoverySlotsWithoutChangingLiveness) {
  a::ObjectGraph g{4,{{2,1},{3},{3},{}},{0},{}};verify(g);
  EXPECT_EQ(a::cheney_copy(g).to_space_order,(std::vector<int>{0,2,1,3}));
  std::reverse(g.refs[0].begin(),g.refs[0].end());verify(g);
  EXPECT_EQ(a::cheney_copy(g).to_space_order,(std::vector<int>{0,1,2,3}));
  g.roots={3,0};verify(g);EXPECT_EQ(a::cheney_copy(g).to_space_order,(std::vector<int>{3,0,1,2}));
}
TEST(CheneyTrace, DeepChainIsIterativeAndWideFrontierSurvivesOutputReallocation) {
  const int n=100000;a::ObjectGraph chain;chain.num_objects=n;chain.refs.resize(n);chain.roots={0};
  for(int i=0;i+1<n;++i)chain.refs[i].push_back(i+1);
  const auto copied=a::cheney_evacuate(chain,n);EXPECT_EQ(copied.work.scans,static_cast<std::size_t>(n));EXPECT_EQ(copied.heap.refs[n-2][0],n-1);
  a::ObjectGraph star;star.num_objects=50001;star.refs.resize(star.num_objects);star.roots={0};
  star.refs[0].resize(50000);std::iota(star.refs[0].begin(),star.refs[0].end(),1);
  const auto wide=a::cheney_copy_trace(star,50001);EXPECT_EQ(wide.copy.heap.refs[0],star.refs[0]);
  const auto first_scan=std::find_if(wide.events.begin(),wide.events.end(),[](const auto& e){return e.phase==a::CheneyPhase::Scan;});
  ASSERT_NE(first_scan,wide.events.end());EXPECT_EQ(first_scan->free_before,1u);EXPECT_EQ(first_scan->free_after,50001u);EXPECT_EQ(first_scan->scan,0u);
}
TEST(CheneyTrace, ReturnedGraphAndTraceCopiesOwnTheirStorage) {
  const auto g=scrambled();const auto original=a::cheney_copy_trace(g);auto copy=original;
  copy.copy.heap.refs[0][0]=99;copy.copy.heap.roots[0]=99;copy.events[0].destination=99;
  EXPECT_EQ(original.copy.heap.refs[0][0],1);EXPECT_EQ(original.copy.heap.roots[0],0);EXPECT_EQ(original.events[0].destination,0);EXPECT_EQ(g.refs[4][0],1);
}
TEST(ObjectGraph, InvalidShapesAndIdsRejectIncludingUnreachableDanglingReferences) {
  std::vector<a::ObjectGraph> invalid{{-1,{},{},{}},{1,{},{},{}},{0,{{}},{},{}},{1,{{}},{-1},{}},{1,{{}},{1},{}},{2,{{},{2}},{0},{}},{2,{{},{-1}},{0},{}}};
  for(const auto& g:invalid) {
    EXPECT_THROW(a::validate_object_graph(g),std::invalid_argument);
    EXPECT_THROW(a::reachable_from_roots(g),std::invalid_argument);
    EXPECT_THROW(a::cheney_copy(g),std::invalid_argument);
    EXPECT_THROW(a::cheney_evacuate(g),std::invalid_argument);
    EXPECT_THROW(a::cheney_copy_trace(g),std::invalid_argument);
  }
}
TEST(ObjectGraph, CheckedSizesRejectEveryBadSuppliedEntryIncludingGarbage) {
  for(const auto& sizes:std::vector<std::vector<int>>{{0},{1,-1},{1,2,0},{1,2,3,4}}) {
    const a::ObjectGraph g{3,{{},{},{}},{0},sizes};
    EXPECT_THROW(a::validate_object_graph(g,true),std::invalid_argument);
    EXPECT_THROW(a::cheney_evacuate(g),std::invalid_argument);
    EXPECT_NO_THROW(a::validate_object_graph(g,false));
  }
}
TEST(ObjectGraph, SizeAccessorChecksOnlyItsRequestedValueAndDenseId) {
  const a::ObjectGraph g{3,{{},{},{}},{},{7,-1}};
  EXPECT_EQ(a::object_size(g,0),7);EXPECT_EQ(a::object_size(g,2),1);
  EXPECT_THROW(a::object_size(g,1),std::invalid_argument);
  EXPECT_THROW(a::object_size(g,-1),std::out_of_range);
  EXPECT_THROW(a::object_size(g,3),std::out_of_range);
  EXPECT_THROW(a::object_size(a::ObjectGraph{},0),std::out_of_range);
}
TEST(ObjectGraph, ValidationCountsIncludeGarbageAndDoNotCountOmittedSizes) {
  const auto g=scrambled();const auto w=a::validate_object_graph(g,true);
  EXPECT_EQ(w.rows,6u);EXPECT_EQ(w.references,6u);EXPECT_EQ(w.roots,2u);EXPECT_EQ(w.sizes,6u);
  EXPECT_EQ(a::validate_object_graph(g).sizes,0u);
}
