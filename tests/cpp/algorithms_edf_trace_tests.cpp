#include <datamunge/algorithms/realtime_scheduling.hpp>
#include <gtest/gtest.h>
#include <algorithm>
#include <limits>
#include <numeric>
#include <random>
#include <tuple>
using namespace datamunge::algorithms;
namespace {
constexpr auto idle=std::numeric_limits<std::size_t>::max();
struct Oracle {std::vector<RTJob> jobs;std::vector<std::size_t> ticks;std::size_t peak=0;};
Oracle oracle(const std::vector<RTTask>& tasks,int h,RTPolicy policy){
    Oracle o;
    // Build the entire finite arrival set offline, independently of the runtime release loop.
    for(std::size_t i=0;i<tasks.size();++i)for(std::int64_t t=0;t<h;t+=tasks[i].period)o.jobs.push_back({i,int(t),t+tasks[i].period,0,tasks[i].wcet==0?int(t):-1,false});
    std::sort(o.jobs.begin(),o.jobs.end(),[](const auto& a,const auto& b){return std::pair{a.release,a.task_index}<std::pair{b.release,b.task_index};});
    for(int t=0;t<h;++t){std::vector<std::size_t> eligible;
        for(std::size_t id=0;id<o.jobs.size();++id){const auto& j=o.jobs[id];if(j.release<=t&&j.executed<tasks[j.task_index].wcet)eligible.push_back(id);}
        o.peak=std::max(o.peak,eligible.size());
        if(eligible.empty()){o.ticks.push_back(idle);continue;}
        const auto key=[&](auto id){const auto& j=o.jobs[id];const auto priority=policy==RTPolicy::earliest_deadline?j.deadline:std::int64_t(tasks[j.task_index].period);return std::tuple{priority,j.task_index,j.release};};
        const auto best=*std::min_element(eligible.begin(),eligible.end(),[&](auto a,auto b){return key(a)<key(b);});auto& j=o.jobs[best];++j.executed;if(j.executed==tasks[j.task_index].wcet)j.completion=t+1;o.ticks.push_back(best);
    }
    for(auto& j:o.jobs)j.missed=j.deadline<=h&&(j.completion<0||j.completion>j.deadline);
    return o;
}
void certificate(const RTTrace& t){
    const auto o=oracle(t.tasks,t.horizon,t.policy);ASSERT_EQ(t.jobs.size(),o.jobs.size());EXPECT_EQ(t.tick_jobs,o.ticks);
    std::size_t misses=0,completed=0,busy=0,preemptions=0;std::vector<int> executed(t.jobs.size());
    for(std::size_t id=0;id<t.jobs.size();++id){const auto& a=t.jobs[id];const auto& b=o.jobs[id];EXPECT_EQ(a.task_index,b.task_index);EXPECT_EQ(a.release,b.release);EXPECT_EQ(a.deadline,b.deadline);EXPECT_EQ(a.executed,b.executed);EXPECT_EQ(a.completion,b.completion);EXPECT_EQ(a.missed,b.missed);misses+=b.missed;completed+=b.completion>=0;}
    for(std::size_t tick=0;tick<t.tick_jobs.size();++tick){const auto id=t.tick_jobs[tick];
        if(tick&&t.tick_jobs[tick-1]!=idle&&t.tick_jobs[tick-1]!=id){const auto old=t.tick_jobs[tick-1];if(executed[old]<t.tasks[t.jobs[old].task_index].wcet)++preemptions;}
        if(id!=idle){++busy;++executed[id];}}
    int end=0;std::vector<std::size_t> labels;
    for(std::size_t i=0;i<t.schedule.timeline.size();++i){const auto& s=t.schedule.timeline[i];EXPECT_EQ(s.start,end);EXPECT_GT(s.end,s.start);if(i)EXPECT_NE(s.id,t.schedule.timeline[i-1].id);for(int tick=s.start;tick<s.end;++tick)labels.push_back(s.id);end=s.end;}
    ASSERT_EQ(labels.size(),t.tick_jobs.size());for(std::size_t tick=0;tick<labels.size();++tick){auto id=t.tick_jobs[tick];EXPECT_EQ(labels[tick],id==idle?idle:t.tasks[t.jobs[id].task_index].id);}
    EXPECT_EQ(end,t.horizon);EXPECT_EQ(t.schedule.deadline_missed,misses>0);EXPECT_EQ(t.work.ticks,std::size_t(t.horizon));EXPECT_EQ(t.work.release_checks,t.tasks.size()*t.horizon);EXPECT_EQ(t.work.jobs_released,o.jobs.size());EXPECT_EQ(t.work.jobs_completed,completed);EXPECT_EQ(t.work.busy_ticks,busy);EXPECT_EQ(t.work.idle_ticks,t.horizon-busy);EXPECT_EQ(t.work.deadline_misses,misses);EXPECT_EQ(t.work.peak_ready,o.peak);EXPECT_EQ(t.work.preemptions,preemptions);EXPECT_EQ(t.captured_entries,t.tasks.size()+t.horizon+t.jobs.size());
    const auto ordinary=rt_simulate_work(t.tasks,t.horizon,t.policy);EXPECT_EQ(ordinary.schedule.deadline_missed,t.schedule.deadline_missed);ASSERT_EQ(ordinary.schedule.timeline.size(),t.schedule.timeline.size());
    for(std::size_t i=0;i<t.schedule.timeline.size();++i){EXPECT_EQ(ordinary.schedule.timeline[i].id,t.schedule.timeline[i].id);EXPECT_EQ(ordinary.schedule.timeline[i].start,t.schedule.timeline[i].start);EXPECT_EQ(ordinary.schedule.timeline[i].end,t.schedule.timeline[i].end);}
    EXPECT_EQ(ordinary.work.busy_ticks,t.work.busy_ticks);EXPECT_EQ(ordinary.work.deadline_misses,t.work.deadline_misses);
}
}
TEST(EDFTrace, ExhaustiveTwoTasksBothPoliciesAndExactSmallUtilization){
    std::size_t scenarios=0;
    for(int p=1;p<=4;++p)for(int q=1;q<=4;++q)for(int a=0;a<=p+1;++a)for(int b=0;b<=q+1;++b){
        const std::vector<RTTask> tasks{{99,p,a},{2,q,b}};EXPECT_EQ(edf_schedulable(tasks),a*q+b*p<=p*q);
        for(int h:{0,1,std::lcm(p,q),2*std::lcm(p,q)})for(auto policy:{RTPolicy::earliest_deadline,RTPolicy::rate_monotonic}){certificate(rt_trace(tasks,h,policy));++scenarios;}}
    EXPECT_EQ(scenarios,2592u);
}
TEST(EDFTrace, SeededMixedBacklogsZeroWorkAndInputOrderTies){
    std::mt19937 rng(20261107);
    for(int trial=0;trial<600;++trial){std::vector<RTTask> tasks;for(std::size_t i=0,n=rng()%5;i<n;++i){const int p=1+rng()%8;tasks.push_back({100-7*i,p,int(rng()%(p+4))});}
        const int h=rng()%41;certificate(rt_trace(tasks,h));certificate(rt_trace(tasks,h,RTPolicy::rate_monotonic));
        std::reverse(tasks.begin(),tasks.end());certificate(rt_trace(tasks,h));}
    const auto t=rt_trace({{99,2,1},{2,2,1}},2);ASSERT_EQ(t.jobs.size(),2u);EXPECT_EQ(t.tick_jobs,(std::vector<std::size_t>{0,1}));
}
TEST(EDFTrace, InclusiveHorizonAndOverdueJobsAreRetained){
    const auto b=rt_trace({{0,2,3}},2);certificate(b);ASSERT_EQ(b.jobs.size(),1u);EXPECT_TRUE(b.schedule.deadline_missed);EXPECT_TRUE(b.jobs[0].missed);EXPECT_EQ(b.jobs[0].executed,2);EXPECT_EQ(b.jobs[0].completion,-1);
    const auto prior=rt_trace({{0,2,3}},1);EXPECT_FALSE(prior.schedule.deadline_missed);
    const auto ontime=rt_trace({{0,2,2}},2);EXPECT_FALSE(ontime.schedule.deadline_missed);EXPECT_EQ(ontime.jobs[0].completion,2);
    const auto overload=rt_trace({{0,2,3},{1,3,1}},6);certificate(overload);EXPECT_EQ(overload.tick_jobs,(std::vector<std::size_t>{0,0,0,1,2,2}));EXPECT_EQ(overload.jobs[0].completion,3);EXPECT_TRUE(overload.jobs[0].missed);EXPECT_EQ(overload.work.deadline_misses,5u);
    const auto zero=rt_trace({{7,1,0},{9,3,0}},6);certificate(zero);EXPECT_EQ(zero.work.jobs_released,8u);EXPECT_EQ(zero.work.jobs_completed,8u);EXPECT_EQ(zero.work.busy_ticks,0u);EXPECT_FALSE(zero.schedule.deadline_missed);
}
TEST(EDFTrace, ExactRationalUtilizationBeyondFloatingTolerance){
    const int big=std::numeric_limits<int>::max();EXPECT_FALSE(edf_schedulable({{0,big,big},{1,big,1}}));
    EXPECT_TRUE(edf_schedulable({{0,3,1},{1,3,1},{2,3,1}}));EXPECT_TRUE(edf_schedulable({}));
    std::vector<RTTask> small;for(int i=0;i<97;++i)small.push_back({std::size_t(i),97,1});EXPECT_TRUE(edf_schedulable(small));small.push_back({99,big,1});EXPECT_FALSE(edf_schedulable(small));
    // Denominators occupy many native limbs; the final extra demand crosses exact equality.
    std::vector<RTTask> zeros;for(int i=0;i<200;++i)zeros.push_back({std::size_t(i),big-i,0});zeros.push_back({500,2,1});zeros.push_back({501,2,1});EXPECT_TRUE(edf_schedulable(zeros));zeros.push_back({502,big,1});EXPECT_FALSE(edf_schedulable(zeros));
    EXPECT_NEAR(rate_monotonic_bound(std::numeric_limits<std::size_t>::max()),std::log(2.),1e-15);
}
TEST(EDFTrace, FullValidationQuotaAndOwnedHistory){
    const auto policy=RTPolicy::earliest_deadline;
    for(const auto& tasks:std::vector<std::vector<RTTask>>{{{0,0,1}},{{0,-1,0}},{{0,2,-1}},{{0,2,0},{0,3,0}},{{idle,2,0}},{{0,1,2},{3,0,1}}}){
        EXPECT_THROW((void)rt_utilization(tasks),std::invalid_argument);
        EXPECT_THROW((void)edf_schedulable(tasks),std::invalid_argument);
        EXPECT_THROW((void)rate_monotonic_priorities(tasks),std::invalid_argument);
        EXPECT_THROW((void)rate_monotonic_schedulable(tasks),std::invalid_argument);
        EXPECT_THROW((void)rt_trace(tasks,0),std::invalid_argument);
        EXPECT_THROW((void)edf_simulate(tasks,0),std::invalid_argument);
        EXPECT_THROW((void)rate_monotonic_simulate(tasks,0),std::invalid_argument);
        EXPECT_THROW((void)rt_trace_entries(tasks,0),std::invalid_argument);
        EXPECT_THROW((void)plot_rt_schedule(tasks,RTScheduleResult{}),std::invalid_argument);
    }
    EXPECT_THROW((void)rt_trace({},-1),std::invalid_argument);
    EXPECT_THROW((void)edf_simulate({},-1),std::invalid_argument);
    EXPECT_THROW((void)rt_trace({},0,static_cast<RTPolicy>(99)),std::invalid_argument);
    EXPECT_THROW((void)rt_simulate_work({},0,static_cast<RTPolicy>(99)),std::invalid_argument);
    std::vector<RTTask> tasks{{7,2,1}};EXPECT_EQ(rt_trace_entries(tasks,2),4u);const auto t=rt_trace(tasks,2,policy,4);EXPECT_EQ(t.captured_entries,4u);
    EXPECT_THROW((void)rt_trace(tasks,2,policy,3),std::length_error);
    EXPECT_EQ(rt_trace(tasks,2,policy,0).captured_entries,4u);EXPECT_EQ(rt_trace({},0,policy,1).captured_entries,0u);
    EXPECT_THROW((void)rt_trace({},std::numeric_limits<int>::max(),policy,1),std::length_error);
    auto copy=t;tasks[0].id=9;copy.tasks[0].id=2;copy.jobs[0].executed=0;copy.tick_jobs[0]=idle;EXPECT_EQ(t.tasks[0].id,7u);EXPECT_EQ(t.jobs[0].executed,1);EXPECT_EQ(t.tick_jobs[0],0u);
}
TEST(EDFTrace, NativeTimelineHandlesSparseIdsAndRejectsMalformedSlices){
    const std::vector<RTTask> tasks{{1000000,2,1},{7,4,1}};const auto r=edf_simulate(tasks,8);const auto p=plot_rt_schedule(tasks,r);EXPECT_TRUE(p.has_x_limits());EXPECT_TRUE(p.has_y_limits());
    EXPECT_EQ(p.y_tick_label_list(),(std::vector<std::string>{"T1000000","T7"}));EXPECT_EQ(p.y_tick_position_list(),(std::vector<double>{0,1}));
    for(const auto& slices:std::vector<std::vector<RTSlice>>{{{7,-1,1}},{{7,0,0}},{{7,0,2},{7,3,4}},{{9,0,1}}}){
        EXPECT_THROW((void)plot_rt_schedule(tasks,RTScheduleResult{slices,false}),std::invalid_argument);}
    EXPECT_NO_THROW((void)plot_rt_schedule({},edf_simulate({},3)));
    EXPECT_NO_THROW((void)plot_rt_schedule({},RTScheduleResult{}));
}
