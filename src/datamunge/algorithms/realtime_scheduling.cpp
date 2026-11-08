/// @file realtime_scheduling.cpp
/// @brief Exact utilization screening, retained-backlog periodic scheduling and native timelines.
#include <datamunge/algorithms/realtime_scheduling.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <tuple>
#include <unordered_set>

namespace datamunge::algorithms {
/// @brief Private validated geometry, exact integer limbs and owned ready jobs.
namespace {
/// @brief Reserve SIZE_MAX exclusively for idle selections.
constexpr auto rt_idle=std::numeric_limits<std::size_t>::max();
/// @brief Validate every task before computation, including tasks following overloads.
/// @param tasks Raw task records.
/// @throws std::invalid_argument On invalid periods, work, duplicate or idle ids.
void rt_validate(const std::vector<RTTask>& tasks){
    std::unordered_set<std::size_t> ids;
    for(const auto& t:tasks){
        if(t.period<=0||t.wcet<0||t.id==rt_idle||!ids.insert(t.id).second)
            throw std::invalid_argument("real-time tasks need positive periods, nonnegative work and unique non-idle ids");
    }
}
/// @brief Check a policy before any scheduling work.
/// @param policy Caller policy.
/// @throws std::invalid_argument On a value outside the declared enum.
void rt_validate_policy(RTPolicy policy){
    if(policy!=RTPolicy::earliest_deadline && policy!=RTPolicy::rate_monotonic)
        throw std::invalid_argument("invalid real-time scheduling policy");
}
/// @brief Add logical counts without wrapping.
/// @param a Existing count.
/// @param b Additional count.
/// @return Checked sum.
/// @throws std::length_error On size_t overflow.
std::size_t rt_add(std::size_t a,std::size_t b){
    if(b>rt_idle-a)throw std::length_error("real-time count overflow");return a+b;
}
/// @brief Calculate validated release counts and history/container geometry.
/// @param tasks Already validated tasks.
/// @param horizon Nonnegative horizon; checked here before conversion.
/// @param jobs Receives total releases before horizon.
/// @return Complete retained entries N+H+J.
/// @throws std::invalid_argument On negative horizon.
/// @throws std::length_error On count or vector limits, including N*H work.
std::size_t rt_geometry(const std::vector<RTTask>& tasks,int horizon,std::size_t& jobs){
    if(horizon<0)throw std::invalid_argument("negative real-time horizon");
    const auto h=static_cast<std::size_t>(horizon);jobs=0;
    if(h && tasks.size()>rt_idle/h)throw std::length_error("real-time release-check count overflow");
    for(const auto& t:tasks)if(h)jobs=rt_add(jobs,1+(h-1)/static_cast<std::size_t>(t.period));
    if(h>std::vector<std::size_t>{}.max_size()||jobs>std::vector<RTJob>{}.max_size())
        throw std::length_error("real-time history geometry exceeds container limits");
    return rt_add(rt_add(tasks.size(),h),jobs);
}
/// @brief Little-endian base-2^32 nonnegative limbs, normalized to one zero limb.
using RTNatural=std::vector<std::uint32_t>;
/// @brief Multiply normalized limbs by a nonnegative int-sized factor.
/// @param a Nonnegative integer limbs, updated in place.
/// @param factor Nonnegative factor no larger than INT_MAX.
void rt_multiply(RTNatural& a,std::uint32_t factor){
    std::uint64_t carry=0;
    for(auto& word:a){const auto product=std::uint64_t(word)*factor+carry;word=static_cast<std::uint32_t>(product);carry=product>>32;}
    if(carry)a.push_back(static_cast<std::uint32_t>(carry));
    while(a.size()>1&&a.back()==0)a.pop_back();
}
/// @brief Add normalized nonnegative integer limbs without fixed-width truncation.
/// @param a First operand, updated to the sum.
/// @param b Independent second operand.
void rt_natural_add(RTNatural& a,const RTNatural& b){
    a.resize(std::max(a.size(),b.size()),0);std::uint64_t carry=0;
    for(std::size_t i=0;i<a.size();++i){const auto sum=std::uint64_t(a[i])+(i<b.size()?b[i]:0)+carry;a[i]=static_cast<std::uint32_t>(sum);carry=sum>>32;}
    if(carry)a.push_back(static_cast<std::uint32_t>(carry));
}
/// @brief Compare normalized nonnegative integers by most significant words.
/// @param a Left operand.
/// @param b Right operand.
/// @return Whether a is strictly larger than b.
bool rt_greater(const RTNatural& a,const RTNatural& b){
    if(a.size()!=b.size())return a.size()>b.size();
    return std::lexicographical_compare(b.rbegin(),b.rend(),a.rbegin(),a.rend());
}
/// @brief One unfinished job owned by the ready queue; no history references.
struct RTReadyJob {
    std::size_t task; ///< Input task index.
    int release; ///< Release time before H.
    std::int64_t deadline; ///< Absolute deadline without int overflow.
    int remaining; ///< Strictly positive work while queued.
    std::size_t id; ///< Global release-order id, including gaps for zero-work releases.
};
/// @brief Min-priority comparator retaining only unchanged caller task metadata.
struct RTLater {
    const std::vector<RTTask>* tasks; ///< Task metadata alive throughout the kernel call.
    RTPolicy policy; ///< Dynamic deadline or fixed period priority.

    /// @brief Order jobs by priority, input task and same-task release order.
    /// @param a First ready job.
    /// @param b Second ready job.
    /// @return Whether a has lower priority than b.
    bool operator()(const RTReadyJob& a,const RTReadyJob& b)const{
        const auto ka=policy==RTPolicy::earliest_deadline?a.deadline:std::int64_t((*tasks)[a.task].period);
        const auto kb=policy==RTPolicy::earliest_deadline?b.deadline:std::int64_t((*tasks)[b.task].period);
        return std::tuple{ka,a.task,a.release}>std::tuple{kb,b.task,b.release};
    }
};
/// @brief Append a unit and merge adjacent equal task labels, including idle.
/// @param slices Result intervals.
/// @param id Selected task label or idle sentinel.
/// @param t Current unit start, less than nonnegative int horizon.
void rt_append(std::vector<RTSlice>& slices,std::size_t id,int t){
    if(!slices.empty()&&slices.back().id==id&&slices.back().end==t)slices.back().end=t+1;
    else slices.push_back({id,t,t+1});
}
/// @brief Count a task's immediately preceding unfinished job at its implicit deadline.
/// @param task Input task index.
/// @param pending Per-task unfinished counts, with same-task FIFO execution.
/// @param last Last release id for each task.
/// @param result Outcome updated on a newly due miss.
/// @param trace Optional owned evidence in progress.
void rt_due(std::size_t task,const std::vector<std::size_t>& pending,
            const std::vector<std::size_t>& last,RTSimulation& result,RTTrace* trace){
    if(!pending[task])return;
    // Same-task FIFO implies its latest released job is unfinished whenever any is.
    result.work.deadline_misses=rt_add(result.work.deadline_misses,1);
    result.schedule.deadline_missed=true;
    if(trace)trace->jobs[last[task]].missed=true;
}
/// @brief Shared actual EDF/RMS run after complete domain/geometry validation.
/// @param tasks Valid task records, unchanged for this call.
/// @param horizon Valid nonnegative horizon.
/// @param policy Valid selection policy.
/// @param trace Optional allocated owned capture; null means no retained history.
/// @return Merged schedule and checked work.
/// @throws std::length_error On counter/container limits; failures expose no partial result.
RTSimulation rt_core(const std::vector<RTTask>& tasks,int horizon,RTPolicy policy,RTTrace* trace){
    RTSimulation out;out.schedule.deadline_missed=false;
    std::vector<std::size_t> pending(tasks.size(),0),last(tasks.size(),rt_idle);
    std::priority_queue<RTReadyJob,std::vector<RTReadyJob>,RTLater> ready(RTLater{&tasks,policy});
    std::size_t previous=rt_idle;bool previous_unfinished=false;
    for(int t=0;t<horizon;++t){
        out.work.ticks=rt_add(out.work.ticks,1);
        out.work.release_checks=rt_add(out.work.release_checks,tasks.size());
        for(std::size_t i=0;i<tasks.size();++i)if(t%tasks[i].period==0){
            if(t)rt_due(i,pending,last,out,trace);
            const auto id=out.work.jobs_released;out.work.jobs_released=rt_add(id,1);last[i]=id;
            const auto deadline=std::int64_t(t)+tasks[i].period;
            if(trace)trace->jobs.push_back({i,t,deadline,0,tasks[i].wcet==0?t:-1,false});
            if(tasks[i].wcet==0)out.work.jobs_completed=rt_add(out.work.jobs_completed,1);
            else{pending[i]=rt_add(pending[i],1);ready.push({i,t,deadline,tasks[i].wcet,id});}
        }
        out.work.peak_ready=std::max(out.work.peak_ready,ready.size());
        if(ready.empty()){
            rt_append(out.schedule.timeline,rt_idle,t);out.work.idle_ticks=rt_add(out.work.idle_ticks,1);
            previous=rt_idle;previous_unfinished=false;continue;
        }
        auto job=ready.top();ready.pop(); // Owned copy survives heap or trace reallocations.
        if(previous_unfinished&&job.id!=previous)out.work.preemptions=rt_add(out.work.preemptions,1);
        rt_append(out.schedule.timeline,tasks[job.task].id,t);out.work.busy_ticks=rt_add(out.work.busy_ticks,1);
        --job.remaining;
        if(trace){trace->tick_jobs[static_cast<std::size_t>(t)]=job.id;auto& record=trace->jobs[job.id];++record.executed;if(!job.remaining)record.completion=t+1;}
        if(!job.remaining){--pending[job.task];out.work.jobs_completed=rt_add(out.work.jobs_completed,1);}
        else ready.push(job);
        previous=job.id;previous_unfinished=job.remaining>0;
    }
    // No release at H, but its deadlines still belong to the observed prefix.
    if(horizon)for(std::size_t i=0;i<tasks.size();++i)if(horizon%tasks[i].period==0)rt_due(i,pending,last,out,trace);
    return out;
}
} // namespace

double rt_utilization(const std::vector<RTTask>& tasks){
    rt_validate(tasks);double total=0;for(const auto& t:tasks)total+=double(t.wcet)/t.period;return total;
}
bool edf_schedulable(const std::vector<RTTask>& tasks){
    rt_validate(tasks);RTNatural numerator{0},denominator{1};
    for(const auto& t:tasks){
        auto contribution=denominator;rt_multiply(contribution,static_cast<std::uint32_t>(t.wcet));
        rt_multiply(numerator,static_cast<std::uint32_t>(t.period));rt_natural_add(numerator,contribution);
        rt_multiply(denominator,static_cast<std::uint32_t>(t.period));
        if(rt_greater(numerator,denominator))return false;
    }
    return true;
}
double rate_monotonic_bound(std::size_t n){
    if(!n)return 1;const auto count=static_cast<double>(n);return count*std::expm1(std::log(2.)/count);
}
bool rate_monotonic_schedulable(const std::vector<RTTask>& tasks){
    return rt_utilization(tasks)<=rate_monotonic_bound(tasks.size());
}
std::vector<std::size_t> rate_monotonic_priorities(const std::vector<RTTask>& tasks){
    rt_validate(tasks);std::vector<std::size_t> order(tasks.size());std::iota(order.begin(),order.end(),0);
    std::stable_sort(order.begin(),order.end(),[&tasks](auto a,auto b){return tasks[a].period<tasks[b].period;});
    for(auto& i:order)i=tasks[i].id;return order;
}
std::size_t rt_trace_entries(const std::vector<RTTask>& tasks,int horizon){
    rt_validate(tasks);std::size_t jobs=0;return rt_geometry(tasks,horizon,jobs);
}
RTSimulation rt_simulate_work(const std::vector<RTTask>& tasks,int horizon,RTPolicy policy){
    rt_validate(tasks);rt_validate_policy(policy);std::size_t jobs=0;(void)rt_geometry(tasks,horizon,jobs);
    return rt_core(tasks,horizon,policy,nullptr);
}
RTTrace rt_trace(const std::vector<RTTask>& tasks,int horizon,RTPolicy policy,std::size_t max_entries){
    rt_validate(tasks);rt_validate_policy(policy);std::size_t jobs=0;const auto entries=rt_geometry(tasks,horizon,jobs);
    if(max_entries&&entries>max_entries)throw std::length_error("real-time capture limit exceeded");
    RTTrace trace;trace.tasks=tasks;trace.horizon=horizon;trace.policy=policy;trace.captured_entries=entries;
    trace.jobs.reserve(jobs);trace.tick_jobs.assign(static_cast<std::size_t>(horizon),rt_idle);
    auto result=rt_core(tasks,horizon,policy,&trace);trace.schedule=std::move(result.schedule);trace.work=result.work;return trace;
}
graphics::RPlot plot_rt_schedule(const std::vector<RTTask>& tasks,const RTScheduleResult& schedule){
    rt_validate(tasks);std::map<std::size_t,std::size_t> row;
    for(std::size_t i=0;i<tasks.size();++i)row.emplace(tasks[i].id,i);
    int end=0;
    for(const auto& s:schedule.timeline){
        if(s.start!=end||s.end<=s.start||(s.id!=rt_idle&&!row.contains(s.id)))throw std::invalid_argument("invalid real-time timeline interval or task id");end=s.end;
    }
    const std::array<graphics::RGB,5> palette{{{37,99,235},{220,38,38},{22,163,74},{202,138,4},{147,51,234}}};
    auto plot=graphics::RPlot::create();
    for(const auto& s:schedule.timeline)if(s.id!=rt_idle){const auto index=row.at(s.id);const auto y=static_cast<double>(index);
        plot.polygon({double(s.start),double(s.end),double(s.end),double(s.start)},{y-.35,y-.35,y+.35,y+.35},palette[index%palette.size()],true);}
    std::vector<double> ticks;std::vector<std::string> labels;
    for(std::size_t i=0;i<tasks.size();++i){ticks.push_back(static_cast<double>(i));labels.push_back("T"+std::to_string(tasks[i].id));}
    plot.x_limits(0,std::max(end,1)).y_limits(-.7,std::max(.7,double(tasks.size())-.3)).y_ticks(std::move(ticks),std::move(labels))
        .title("Periodic task execution (idle intervals blank)").x_label("Time boundary").y_label("Task id (input order)");
    return plot;
}
} // namespace datamunge::algorithms
