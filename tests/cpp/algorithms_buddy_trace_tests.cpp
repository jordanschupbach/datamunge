#include <gtest/gtest.h>
#include <datamunge/algorithms/buddy_allocator.hpp>
#include <algorithm>
#include <climits>
#include <deque>
#include <iostream>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace datamunge::algorithms;

// Independent occupancy bitmap: derive maximal free dyadic intervals afresh.
struct Reference {
  int total,minimum;
  std::vector<bool> busy;
  std::vector<int> requested,length;
  Reference(int n,int m):total(n),minimum(m),busy(n/m),requested(n/m),length(n/m) {}
  std::vector<BuddyBlock> blocks() const {
    std::vector<BuddyBlock> result;
    walk(0,static_cast<int>(busy.size()),result);
    return result;
  }
  void walk(int start,int count,std::vector<BuddyBlock>& output) const {
    int order=0;for (int x=count;x>1;x/=2) ++order;
    bool empty=true;for (int i=start;i<start+count;++i) empty=empty && !busy[i];
    if (empty) output.push_back({start*minimum,order,count*minimum,0,false});
    else if (length[start]==count) output.push_back({start*minimum,order,count*minimum,requested[start],true});
    else {walk(start,count/2,output);walk(start+count/2,count/2,output);}
  }
  int allocate(int size) {
    int wanted=minimum;while (wanted<size) wanted*=2;
    const auto partition=blocks();
    int best=-1,best_size=INT_MAX;
    for (const auto& block:partition) if (!block.allocated && block.size>=wanted && block.size<best_size) {
      best=block.offset;best_size=block.size;
    }
    if (best<0) return -1;
    const int start=best/minimum,count=wanted/minimum;
    for (int i=start;i<start+count;++i) busy[i]=true;
    requested[start]=size;length[start]=count;
    return best;
  }
  void free(int offset) {
    if (offset<0 || offset>=total || offset%minimum) return;
    const int start=offset/minimum;
    if (!length[start]) return;
    for (int i=start;i<start+length[start];++i) busy[i]=false;
    requested[start]=length[start]=0;
  }
  std::vector<int> key() const {
    auto result=requested;result.insert(result.end(),length.begin(),length.end());return result;
  }
};

void check_state(const BuddyAllocator& actual,const Reference& reference) {
  const auto snapshot=actual.snapshot();
  const auto expected=reference.blocks();
  ASSERT_EQ(snapshot.blocks.size(),expected.size());
  int allocated=0,requested=0,largest=0,end=0;
  std::vector<std::vector<int>> free_offsets(actual.max_order()+1);
  for (std::size_t i=0;i<expected.size();++i) {
    const auto& got=snapshot.blocks[i];const auto& want=expected[i];
    EXPECT_EQ(got.offset,want.offset);EXPECT_EQ(got.order,want.order);EXPECT_EQ(got.size,want.size);
    EXPECT_EQ(got.requested,want.requested);EXPECT_EQ(got.allocated,want.allocated);
    EXPECT_EQ(got.offset,end);EXPECT_EQ(got.offset%got.size,0);end+=got.size;
    if (got.allocated) {allocated+=got.size;requested+=got.requested;}
    else {free_offsets[got.order].push_back(got.offset);largest=std::max(largest,got.size);}
  }
  EXPECT_EQ(end,reference.total);EXPECT_EQ(snapshot.total_bytes,reference.total);EXPECT_EQ(snapshot.min_block,reference.minimum);
  EXPECT_EQ(snapshot.allocated_bytes,allocated);EXPECT_EQ(snapshot.requested_bytes,requested);
  EXPECT_EQ(snapshot.internal_fragmentation,allocated-requested);EXPECT_EQ(snapshot.free_bytes,reference.total-allocated);
  EXPECT_EQ(snapshot.largest_free_block,largest);EXPECT_EQ(snapshot.free_offsets,free_offsets);
  EXPECT_EQ(actual.allocated_bytes(),allocated);EXPECT_EQ(actual.requested_bytes(),requested);
  EXPECT_EQ(actual.internal_fragmentation(),allocated-requested);
  const auto counts=actual.free_blocks_per_order();ASSERT_EQ(counts.size(),free_offsets.size());
  for (std::size_t order=0;order<counts.size();++order) {
    EXPECT_EQ(counts[order],free_offsets[order].size());
    for (const int off:free_offsets[order])
      EXPECT_FALSE(std::binary_search(free_offsets[order].begin(),free_offsets[order].end(),off^actual.block_size(static_cast<int>(order))));
  }
}

void check_allocate(const BuddyAllocationTrace& trace,const Reference& before,int request,int returned,int max_order) {
  EXPECT_EQ(trace.requested,request);EXPECT_EQ(trace.offset,returned);
  int desired=0;for (int block=before.minimum;block<request;block*=2) ++desired;
  ASSERT_EQ(trace.desired_order,desired);
  if (returned<0) {
    EXPECT_EQ(trace.source_order,-1);EXPECT_TRUE(trace.splits.empty());
    EXPECT_EQ(trace.orders_examined,desired>max_order?0:max_order-desired+1);
    return;
  }
  const auto blocks=before.blocks();
  const auto it=std::find_if(blocks.begin(),blocks.end(),[returned](const BuddyBlock& b){return !b.allocated && b.offset<=returned && returned<b.offset+b.size;});
  ASSERT_NE(it,blocks.end());
  EXPECT_EQ(it->offset,returned);EXPECT_EQ(trace.source_order,it->order);
  EXPECT_EQ(trace.orders_examined,it->order-desired+1);
  ASSERT_EQ(trace.splits.size(),it->order-desired);
  for (std::size_t i=0;i<trace.splits.size();++i) {
    EXPECT_EQ(trace.splits[i].offset,returned);EXPECT_EQ(trace.splits[i].parent_order,it->order-static_cast<int>(i));
    EXPECT_EQ(trace.splits[i].half_size,it->size>>(i+1));
  }
}

void check_free(const BuddyFreeTrace& trace,const Reference& before,int offset,int max_order) {
  EXPECT_EQ(trace.requested_offset,offset);
  const auto partition=before.blocks();
  const auto live=std::find_if(partition.begin(),partition.end(),[offset](const BuddyBlock& b){return b.allocated && b.offset==offset;});
  if (live==partition.end()) {
    EXPECT_FALSE(trace.released);EXPECT_EQ(trace.final_offset,-1);EXPECT_EQ(trace.final_order,-1);
    EXPECT_EQ(trace.merges,0);EXPECT_TRUE(trace.probes.empty());return;
  }
  ASSERT_TRUE(trace.released);
  int start=offset,order=live->order,size=live->size;std::size_t probes=0,merges=0;
  while (order<max_order) {
    const int buddy=start^size;
    const bool free=std::any_of(partition.begin(),partition.end(),[buddy,order](const BuddyBlock& b){return !b.allocated && b.offset==buddy && b.order==order;});
    ASSERT_LT(probes,trace.probes.size());const auto& got=trace.probes[probes++];
    EXPECT_EQ(got.offset,start);EXPECT_EQ(got.order,order);EXPECT_EQ(got.buddy_offset,buddy);EXPECT_EQ(got.buddy_free,free);
    if (!free) break;
    ++merges;start=std::min(start,buddy);++order;size*=2;
  }
  EXPECT_EQ(trace.probes.size(),probes);EXPECT_EQ(trace.merges,merges);
  EXPECT_EQ(trace.final_offset,start);EXPECT_EQ(trace.final_order,order);
}
}

TEST(BuddyTrace, OriginalRequestsOffsetsAccountingAndCoalescence) {
  BuddyAllocator a(1024,32);Reference reference(1024,32);
  std::vector<int> offsets;
  for (const int request:{30,100,50}) {
    const auto before=reference;const int expected=reference.allocate(request);
    const auto trace=a.allocate_trace(request);EXPECT_EQ(trace.offset,expected);
    check_allocate(trace,before,request,expected,a.max_order());check_state(a,reference);offsets.push_back(trace.offset);
  }
  EXPECT_EQ(offsets,(std::vector<int>{0,128,64}));EXPECT_EQ(a.allocated_bytes(),224);EXPECT_EQ(a.requested_bytes(),180);
  EXPECT_EQ(a.internal_fragmentation(),44);EXPECT_EQ(a.free_blocks_per_order(),(std::vector<int>{1,0,0,1,1,0}));
  const auto preserved=a.snapshot();
  const std::vector<std::size_t> expected_merges{1,0,4};
  for (std::size_t i=0;i<offsets.size();++i) {
    const auto before=reference;const auto trace=a.free_trace(offsets[i]);reference.free(offsets[i]);
    EXPECT_EQ(trace.merges,expected_merges[i]);check_free(trace,before,offsets[i],a.max_order());check_state(a,reference);
  }
  EXPECT_EQ(a.free_blocks_per_order(),(std::vector<int>{0,0,0,0,0,1}));
  EXPECT_EQ(preserved.allocated_bytes,224);EXPECT_EQ(preserved.free_bytes,800);
}

TEST(BuddyTrace, ExhaustsAllReachableEightByteStatesAgainstBitmapModel) {
  struct State {BuddyAllocator allocator;Reference reference;};
  std::deque<State> queue;queue.push_back({BuddyAllocator(8,1),Reference(8,1)});
  std::set<std::vector<int>> visited;visited.insert(queue.front().reference.key());
  std::size_t transitions=0;
  while (!queue.empty()) {
    auto state=std::move(queue.front());queue.pop_front();check_state(state.allocator,state.reference);
    for (int request=1;request<=9;++request) {
      auto next=state;const auto trace=next.allocator.allocate_trace(request);
      const int expected=next.reference.allocate(request);
      ASSERT_EQ(trace.offset,expected);check_allocate(trace,state.reference,request,expected,state.allocator.max_order());
      auto plain=state.allocator;EXPECT_EQ(plain.allocate(request),expected);check_state(plain,next.reference);
      if (visited.insert(next.reference.key()).second) queue.push_back(std::move(next));
      ++transitions;
    }
    for (int offset=-1;offset<=8;++offset) {
      auto next=state;const auto trace=next.allocator.free_trace(offset);next.reference.free(offset);
      check_free(trace,state.reference,offset,state.allocator.max_order());
      auto plain=state.allocator;plain.free(offset);check_state(plain,next.reference);
      if (visited.insert(next.reference.key()).second) queue.push_back(std::move(next));
      ++transitions;
    }
  }
  std::cout<<"Exhaustive buddy states: "<<visited.size()<<"; transitions: "<<transitions<<'\n';
  // Free-root/allocated-root/independent-child recurrence: f(0)=2, f(h)=f(h-1)^2+2^(h-1).
  EXPECT_EQ(visited.size(),733);EXPECT_EQ(transitions,visited.size()*19);
}

TEST(BuddyTrace, RandomLargerArenasMatchBitmapAndActualPaths) {
  std::mt19937 rng(87131);
  for (int arena:{64,256,1024}) for (int minimum:{1,8,32}) {
    BuddyAllocator a(arena,minimum);Reference reference(arena,minimum);
    for (int step=0;step<1000;++step) {
      const auto before=reference;
      if (rng()%3) {
        const int request=1+static_cast<int>(rng()%(arena+100));
        const int expected=reference.allocate(request);const auto trace=a.allocate_trace(request);
        ASSERT_EQ(trace.offset,expected);check_allocate(trace,before,request,expected,a.max_order());
      } else {
        int offset=-1;const auto blocks=before.blocks();
        std::vector<int> live;for (const auto& b:blocks) if (b.allocated) live.push_back(b.offset);
        if (!live.empty() && rng()%4) offset=live[rng()%live.size()];
        else offset=static_cast<int>(rng()%(arena+20))-10;
        const auto trace=a.free_trace(offset);reference.free(offset);check_free(trace,before,offset,a.max_order());
      }
      check_state(a,reference);
    }
  }
}

TEST(BuddyTrace, AdjacentNonBuddiesStillCauseExternalFragmentation) {
  BuddyAllocator a(256,32);std::vector<int> offsets;
  for (int i=0;i<8;++i) offsets.push_back(a.allocate(32));
  EXPECT_EQ(offsets,(std::vector<int>{0,32,64,96,128,160,192,224}));
  EXPECT_EQ(a.free_trace(32).merges,0);EXPECT_EQ(a.free_trace(64).merges,0);
  const auto split=a.snapshot();EXPECT_EQ(split.free_bytes,64);EXPECT_EQ(split.largest_free_block,32);
  EXPECT_EQ(split.free_offsets[0],(std::vector<int>{32,64}));
  const auto failure=a.allocate_trace(64);EXPECT_EQ(failure.offset,-1);EXPECT_EQ(failure.orders_examined,3);
  EXPECT_TRUE(failure.splits.empty());EXPECT_EQ(a.snapshot().free_offsets,split.free_offsets);
  const auto release=a.free_trace(0);ASSERT_EQ(release.probes.size(),2);
  EXPECT_TRUE(release.probes[0].buddy_free);EXPECT_FALSE(release.probes[1].buddy_free);
  EXPECT_EQ(release.merges,1);EXPECT_EQ(a.allocate(64),0);
}

TEST(BuddyTrace, RoundingBoundHasMinimumBlockException) {
  BuddyAllocator a(1024,32);
  for (int request=1;request<=1024;++request) {
    const int order=a.order_for_size(request),size=a.block_size(order);
    EXPECT_GE(size,request);if (order>0) EXPECT_LT(size/2,request);
    const auto trace=a.allocate_trace(request);ASSERT_EQ(trace.offset,0);
    EXPECT_EQ(a.internal_fragmentation(),size-request);
    if (request>32) EXPECT_LT(2*a.internal_fragmentation(),size);
    a.free(trace.offset);
  }
  const auto trace=a.allocate_trace(1);EXPECT_EQ(a.internal_fragmentation(),31);a.free(trace.offset);
}

TEST(BuddyTrace, CopyAndSnapshotOwnershipAndUnknownFrees) {
  BuddyAllocator a(64,8);const int off=a.allocate(5);auto copy=a;const auto snapshot=a.snapshot();
  for (int unknown:{-1,1,7,64,INT_MIN,INT_MAX}) EXPECT_FALSE(a.free_trace(unknown).released);
  EXPECT_EQ(a.allocated_bytes(),8);copy.free(off);EXPECT_EQ(copy.allocated_bytes(),0);EXPECT_EQ(a.allocated_bytes(),8);
  EXPECT_EQ(snapshot.allocated_bytes,8);EXPECT_EQ(snapshot.free_offsets[0],(std::vector<int>{8}));
  a.free(off);const auto ignored=a.free_trace(off);EXPECT_FALSE(ignored.released);EXPECT_TRUE(ignored.probes.empty());
}

TEST(BuddyTrace, OrderWorkTracksArenaHeightWithoutPayloadAllocation) {
  for (int height=0;height<=30;++height) {
    BuddyAllocator a(1<<height,1);const auto allocation=a.allocate_trace(1);
    EXPECT_EQ(allocation.offset,0);EXPECT_EQ(allocation.orders_examined,height+1);EXPECT_EQ(allocation.splits.size(),height);
    const auto release=a.free_trace(allocation.offset);EXPECT_EQ(release.merges,height);EXPECT_EQ(release.probes.size(),height);
    EXPECT_EQ(release.final_order,height);EXPECT_EQ(a.snapshot().largest_free_block,1<<height);
  }
}

TEST(BuddyTrace, ValidatesDimensionsRequestsOrdersAndOversizeSafely) {
  for (const auto pair:std::vector<std::pair<int,int>>{{0,1},{1,0},{-1,1},{1,-1},{3,1},{8,3},{4,8},{INT_MIN,1},{INT_MAX,1}})
    EXPECT_THROW((BuddyAllocator(pair.first,pair.second)),std::invalid_argument);
  BuddyAllocator a(64,8);const auto before=a.snapshot();
  EXPECT_THROW(a.allocate(0),std::invalid_argument);
  EXPECT_THROW(a.allocate(-1),std::invalid_argument);
  EXPECT_THROW(a.allocate_trace(INT_MIN),std::invalid_argument);
  EXPECT_THROW(a.order_for_size(0),std::invalid_argument);
  EXPECT_THROW(a.block_size(-1),std::out_of_range);
  EXPECT_THROW(a.block_size(4),std::out_of_range);
  EXPECT_THROW(a.block_size(INT_MAX),std::out_of_range);
  const auto large=a.allocate_trace(INT_MAX);EXPECT_EQ(large.offset,-1);EXPECT_EQ(large.desired_order,28);
  EXPECT_EQ(large.orders_examined,0);EXPECT_TRUE(large.splits.empty());EXPECT_EQ(a.snapshot().free_offsets,before.free_offsets);
  BuddyAllocator maximum(1<<30,1);EXPECT_EQ(maximum.order_for_size(INT_MAX),31);
  EXPECT_EQ(maximum.allocate(INT_MAX),-1);EXPECT_EQ(maximum.allocate(1<<30),0);EXPECT_EQ(maximum.allocate(1),-1);
  maximum.free(0);EXPECT_EQ(maximum.snapshot().free_bytes,1<<30);
}
