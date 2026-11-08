#include <gtest/gtest.h>
#include <datamunge/algorithms/bully_election.hpp>
#include <algorithm>
#include <climits>
#include <deque>
#include <functional>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

namespace {
using namespace datamunge::algorithms;
void same_result(const BullyResult& got,const BullyResult& expected) {
  EXPECT_EQ(got.leader,expected.leader);EXPECT_EQ(got.election_messages,expected.election_messages);
  EXPECT_EQ(got.answer_messages,expected.answer_messages);EXPECT_EQ(got.coordinator_messages,expected.coordinator_messages);
}

// Independent delayed FIFO delivery model, rather than the production phase kernel.
BullyResult event_reference(const std::vector<char>& alive,int initiator) {
  const int n=static_cast<int>(alive.size());BullyResult result;
  if (initiator<0) return result;
  struct Event {BullyMessageKind kind;int sender,receiver;};
  std::deque<Event> pending;std::vector<bool> started(n),answered(n);
  auto start=[&](int process) {
    started[process]=true;
    for (int other=process+1;other<n;++other) {
      pending.push_back({BullyMessageKind::Election,process,other});++result.election_messages;
    }
  };
  start(initiator);
  while (!pending.empty()) {
    const auto event=pending.front();pending.pop_front();
    if (!alive[event.receiver]) continue;
    if (event.kind==BullyMessageKind::Election) {
      pending.push_back({BullyMessageKind::Answer,event.receiver,event.sender});++result.answer_messages;
      if (!started[event.receiver]) start(event.receiver);
    } else answered[event.receiver]=true;
  }
  for (int process=0;process<n;++process) if (started[process] && !answered[process]) {
    EXPECT_EQ(result.leader,-1);result.leader=process;
  }
  if (result.leader>=0) result.coordinator_messages=static_cast<std::size_t>(result.leader);
  return result;
}

void check_trace(const std::vector<char>& alive,int initiator,bool fanout) {
  const auto preserved=alive;const auto n=alive.size();
  const auto trace=fanout?bully_elect_fanout_trace(n,alive,initiator):bully_elect_trace(n,alive,initiator);
  EXPECT_EQ(alive,preserved);EXPECT_EQ(trace.fanout,fanout);EXPECT_EQ(trace.initiator,initiator);
  const auto ordinary=fanout?bully_elect_fanout(n,alive,initiator):bully_elect(n,alive,initiator);
  same_result(trace.result,ordinary);
  int leader=-1;for (std::size_t i=0;i<n;++i) if (alive[i]) leader=static_cast<int>(i);
  ASSERT_EQ(trace.result.leader,leader);ASSERT_EQ(trace.learned_leader.size(),n);
  std::vector<int> candidates;
  if (initiator>=0) {
    candidates.push_back(initiator);
    if (fanout) {for (int id=initiator+1;id<=leader;++id) if (alive[id]) candidates.push_back(id);}
    else if (leader!=initiator) candidates.push_back(leader);
  }
  ASSERT_EQ(trace.rounds.size(),candidates.size());
  std::size_t expected_elections=0,expected_answers=0,delivered=0;
  for (std::size_t phase=0;phase<candidates.size();++phase) {
    const int candidate=candidates[phase];const auto& round=trace.rounds[phase];
    std::size_t answers=0;for (std::size_t id=static_cast<std::size_t>(candidate)+1;id<n;++id) answers+=alive[id]!=0;
    EXPECT_EQ(round.process,candidate);EXPECT_EQ(round.highest_responder,candidate==leader?-1:leader);
    EXPECT_EQ(round.election_messages,n-1-candidate);EXPECT_EQ(round.answer_messages,answers);
    EXPECT_EQ(round.candidates_started,fanout?(phase==0?candidates.size()-1:0):(phase==0 && candidates.size()==2?1:0));
    expected_elections+=n-1-candidate;expected_answers+=answers;
  }
  const auto coordinator=leader<0?std::size_t{0}:static_cast<std::size_t>(leader);
  EXPECT_EQ(trace.result.election_messages,expected_elections);EXPECT_EQ(trace.result.answer_messages,expected_answers);
  EXPECT_EQ(trace.result.coordinator_messages,coordinator);
  EXPECT_EQ(trace.total_messages,expected_elections+expected_answers+coordinator);
  ASSERT_EQ(trace.messages.size(),trace.total_messages);
  std::size_t cursor=0;
  auto message=[&](BullyMessageKind kind,int from,int to,std::size_t phase) {
    ASSERT_LT(cursor,trace.messages.size());const auto& actual=trace.messages[cursor++];
    EXPECT_EQ(actual.kind,kind);EXPECT_EQ(actual.sender,from);EXPECT_EQ(actual.receiver,to);
    EXPECT_EQ(actual.phase,phase);EXPECT_EQ(actual.delivered,alive[to]!=0);EXPECT_TRUE(alive[from]);
    delivered+=alive[to]!=0;
  };
  for (std::size_t phase=0;phase<candidates.size();++phase) {
    const int from=candidates[phase];
    for (std::size_t target=static_cast<std::size_t>(from)+1;target<n;++target) {
      message(BullyMessageKind::Election,from,static_cast<int>(target),phase);
      if (alive[target]) message(BullyMessageKind::Answer,static_cast<int>(target),from,phase);
    }
  }
  for (int lower=0;lower<leader;++lower) message(BullyMessageKind::Coordinator,leader,lower,candidates.size());
  EXPECT_EQ(cursor,trace.messages.size());EXPECT_EQ(delivered,trace.delivered_messages);
  for (std::size_t id=0;id<n;++id) EXPECT_EQ(trace.learned_leader[id],alive[id]?leader:-1);
  BullyTraceOptions lean;lean.record_messages=false;
  const auto summary=fanout?bully_elect_fanout_trace(n,alive,initiator,lean):bully_elect_trace(n,alive,initiator,lean);
  same_result(summary.result,trace.result);EXPECT_EQ(summary.total_messages,trace.total_messages);
  EXPECT_EQ(summary.delivered_messages,trace.delivered_messages);EXPECT_TRUE(summary.messages.empty());
  EXPECT_EQ(summary.learned_leader,trace.learned_leader);EXPECT_EQ(summary.rounds.size(),trace.rounds.size());
  if (fanout) same_result(trace.result,event_reference(alive,initiator));
}
}

TEST(BullyTrace, PreservesLegacyAggregateSignatureAndOriginalNumbers) {
  const std::function<BullyResult(std::size_t,const std::vector<char>&,int)> legacy=bully_elect;
  const std::vector<char> alive{1,1,1,1,0,0};const auto r=legacy(6,alive,1);
  same_result(r,BullyResult{3,6,2,3});
  const auto t=bully_elect_trace(6,alive,1);ASSERT_EQ(t.rounds.size(),2);
  EXPECT_EQ(t.rounds[0].process,1);EXPECT_EQ(t.rounds[1].process,3);EXPECT_EQ(t.total_messages,11);
  check_trace(alive,1,false);
}

TEST(BullyTrace, FullResponderModelHasAnAdditionalCandidateAndActualMessages) {
  const std::vector<char> alive{1,1,1,1,0,0};
  const auto trace=bully_elect_fanout_trace(6,alive,1);same_result(trace.result,BullyResult{3,9,3,3});
  ASSERT_EQ(trace.rounds.size(),3);EXPECT_EQ(trace.rounds[0].process,1);EXPECT_EQ(trace.rounds[1].process,2);EXPECT_EQ(trace.rounds[2].process,3);
  EXPECT_EQ(trace.total_messages,15);EXPECT_EQ(trace.delivered_messages,9);
  EXPECT_EQ(trace.learned_leader,(std::vector<int>{3,3,3,3,-1,-1}));check_trace(alive,1,true);
}

TEST(BullyTrace, ExhaustiveLivenessAndEveryLiveInitiator) {
  std::size_t elections=0;
  for (int n=1;n<=10;++n) for (unsigned mask=0;mask<(1u<<n);++mask) {
    std::vector<char> alive(n);for (int id=0;id<n;++id) alive[id]=static_cast<char>((mask>>id)&1);
    if (!mask) {check_trace(alive,-1,false);check_trace(alive,-1,true);}
    for (int id=0;id<n;++id) if (alive[id]) {check_trace(alive,id,false);check_trace(alive,id,true);++elections;}
  }
  EXPECT_EQ(elections,9217);
}

TEST(BullyTrace, RandomHolesMatchIndependentDelayedDeliverySimulator) {
  std::mt19937 rng(91801);
  for (int trial=0;trial<1000;++trial) {
    const auto n=1+rng()%64;std::vector<char> alive(n);
    for (auto& status:alive) status=rng()%4==0?0:static_cast<char>(rng()%2?2:-1);
    const int initiator=static_cast<int>(rng()%n);alive[initiator]=1;
    check_trace(alive,initiator,false);check_trace(alive,initiator,true);
  }
}

TEST(BullyTrace, AllUpClosedFormsDependOnInitiatorNotOnlyLeaderHealth) {
  for (std::size_t n=1;n<=64;++n) {
    const std::vector<char> alive(n,1);
    for (std::size_t start=0;start<n;++start) {
      const auto fanout=bully_elect_fanout(n,alive,static_cast<int>(start));
      const auto remaining=n-start;
      EXPECT_EQ(fanout.election_messages,remaining*(remaining-1)/2);
      EXPECT_EQ(fanout.answer_messages,fanout.election_messages);EXPECT_EQ(fanout.coordinator_messages,n-1);
      const auto shortcut=bully_elect(n,alive,static_cast<int>(start));
      EXPECT_EQ(shortcut.election_messages,n-1-start);EXPECT_EQ(shortcut.answer_messages,n-1-start);
      if (start==0) EXPECT_EQ(fanout.election_messages+fanout.answer_messages+fanout.coordinator_messages,n*n-1);
      if (start==n-1) EXPECT_EQ(fanout.election_messages+fanout.answer_messages,0);
    }
  }
}

TEST(BullyTrace, DeadTargetsAndLowerNonCandidatesHaveExplicitDeliverySemantics) {
  const std::vector<char> alive{0,1,0,1,0,1,0};
  const auto full=bully_elect_fanout_trace(7,alive,3);same_result(full.result,BullyResult{5,4,1,5});
  EXPECT_EQ(full.learned_leader,(std::vector<int>{-1,5,-1,5,-1,5,-1}));
  EXPECT_EQ(full.delivered_messages,4);EXPECT_EQ(full.total_messages,10);
  EXPECT_EQ(full.rounds.size(),2);check_trace(alive,3,true);check_trace(alive,3,false);
}

TEST(BullyTrace, EmptyAllDownSingletonAndNonzeroFlags) {
  check_trace({},-1,false);check_trace({},-1,true);
  check_trace({0,0,0},-1,false);check_trace({0,0,0},-1,true);
  check_trace({1},0,false);check_trace({1},0,true);
  check_trace({static_cast<char>(-1),0,2},0,false);check_trace({static_cast<char>(-1),0,2},0,true);
}

TEST(BullyTrace, ExtraLivenessEntriesRemainIgnored) {
  const std::vector<char> longer{1,1,1,0,static_cast<char>(-1)};
  same_result(bully_elect(3,longer,0),bully_elect(3,{1,1,1},0));
  same_result(bully_elect_fanout(3,longer,0),bully_elect_fanout(3,{1,1,1},0));
  EXPECT_EQ(bully_elect_trace(3,longer,0).learned_leader.size(),3);
  EXPECT_EQ(bully_elect_fanout_trace(0,longer,-1).result.leader,-1);
  EXPECT_EQ(bully_elect_fanout_trace(1,{0,1},-1).result.leader,-1);
}

TEST(BullyTrace, CompleteCallBudgetsCountUndeliveredAndAnnouncementAttempts) {
  const std::vector<char> alive{1,1,1,1,0,0};
  for (const bool fanout:{false,true}) {
    const auto baseline=fanout?bully_elect_fanout_trace(6,alive,1):bully_elect_trace(6,alive,1);
    for (const bool record:{false,true}) {
      BullyTraceOptions options;options.record_messages=record;
      for (std::size_t cap=1;cap<baseline.total_messages;++cap) {
        options.max_messages=cap;
        if (fanout) {EXPECT_THROW(bully_elect_fanout_trace(6,alive,1,options),std::runtime_error);}
        else {EXPECT_THROW(bully_elect_trace(6,alive,1,options),std::runtime_error);}
      }
      options.max_messages=baseline.total_messages;
      const auto limited=fanout?bully_elect_fanout_trace(6,alive,1,options):bully_elect_trace(6,alive,1,options);
      same_result(limited.result,baseline.result);EXPECT_EQ(limited.total_messages,baseline.total_messages);
    }
  }
  BullyTraceOptions cap;cap.max_messages=1;cap.record_messages=false;
  EXPECT_THROW(bully_elect_fanout_trace(4096,std::vector<char>(4096,1),0,cap),std::runtime_error);
  EXPECT_EQ(bully_elect_fanout_trace(1,{1},0,cap).total_messages,0);
  EXPECT_EQ(bully_elect_fanout_trace(4,{0,0,0,0},-1,cap).total_messages,0);
}

TEST(BullyTrace, ValidationPrecedesWorkAndInvalidOrDeadInitiatorsThrow) {
  for (const bool fanout:{false,true}) {
    auto run=[fanout](std::size_t n,const std::vector<char>& flags,int start) {
      return fanout?bully_elect_fanout_trace(n,flags,start):bully_elect_trace(n,flags,start);
    };
    EXPECT_THROW(run(3,{1,1},0),std::invalid_argument);
    EXPECT_THROW(run(3,{1,1,1},-1),std::invalid_argument);
    EXPECT_THROW(run(3,{1,1,1},-2),std::invalid_argument);
    EXPECT_THROW(run(3,{1,1,1},3),std::invalid_argument);
    EXPECT_THROW(run(3,{1,0,1},1),std::invalid_argument);
    EXPECT_THROW(run(3,{0,0,0},0),std::invalid_argument);
    EXPECT_THROW(run(0,{},0),std::invalid_argument);
    EXPECT_THROW(run(static_cast<std::size_t>(INT_MAX)+1,{},0),std::length_error);
    EXPECT_THROW(run(std::numeric_limits<std::size_t>::max(),{},-1),std::length_error);
  }
  EXPECT_THROW(bully_elect(3,{1,0,1},1),std::invalid_argument);
  EXPECT_THROW(bully_elect_fanout(3,{1},0),std::invalid_argument);
}
