#pragma once

/// @file profiler.hpp
/// @brief A per-op trace of nn work: timed regions, automatic backward-op events, summaries,
///        Chrome trace export and a timeline plot.
///
/// The counterpart of TensorFlow's `RunOptions::FULL_TRACE` / `RunMetadata::step_stats` and of
/// `torch.profiler`. While a Profiler is started, every ProfileScope alive on any thread records
/// one event (name, category, thread, nesting depth, start and duration), and the autograd
/// engine records one "backward" event per node it executes. Forward ops run eagerly as plain
/// function calls, so they are traced by wrapping them in a ProfileScope.
///
/// When no profiler is active a ProfileScope costs one atomic load. At most one profiler is
/// active at a time; starting a second one stops the first.
///
/// @par Example
/// @code
/// nn::Profiler prof;
/// prof.start();
/// {
///     nn::ProfileScope s("forward");
///     y = nn::matmul(x, w);
/// }
/// prof.stop();
/// std::cout << prof.to_markdown();
/// @endcode

#include "datamunge/plot/plot.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

namespace datamunge::nn {

/// @brief One timed interval recorded by a Profiler.
struct ProfileEvent {
    /// Region or op name ("forward", "MatMul", ...).
    std::string name;
    /// Category: "region" for ProfileScope by default, "backward" for autograd nodes.
    std::string category;
    /// Small thread index, 0 for the first thread seen by the profiler, then 1, 2, ...
    int thread{0};
    /// Nesting depth on its thread: 0 for an outermost scope, 1 for a scope inside it, ...
    int depth{0};
    /// Start, in microseconds since Profiler::start().
    double start_us{0};
    /// Duration in microseconds.
    double duration_us{0};
};

/// @brief Aggregate timings of all events with one (category, name) pair.
struct ProfileRow {
    /// Event name.
    std::string name;
    /// Event category.
    std::string category;
    /// Number of events.
    std::size_t calls{0};
    /// Sum of durations, in microseconds.
    double total_us{0};
    /// Sum of durations minus the durations of directly nested events (time not spent in children).
    double self_us{0};
    /// total_us / calls.
    double mean_us{0};
    /// Shortest event, in microseconds.
    double min_us{0};
    /// Longest event, in microseconds.
    double max_us{0};
};

/// @brief Collects ProfileEvents between start() and stop().
///
/// Events are appended under a mutex, so scopes on several threads may record concurrently.
/// The profiler must outlive the scopes that record into it, or be stopped first; a scope that
/// ends after stop() is dropped.
class Profiler {
public:
    /// @brief Creates a stopped profiler with no events.
    Profiler() = default;
    /// @brief Stops the profiler if it is active.
    ~Profiler();
    /// @brief Disallowed: the active-profiler registry refers to this object by address.
    Profiler(const Profiler&) = delete;
    /// @brief Disallowed: the active-profiler registry refers to this object by address.
    Profiler& operator=(const Profiler&) = delete;

    /// @brief Clears the events, resets the clock origin and makes this the active profiler
    ///        (stopping any other active profiler).
    void start();
    /// @brief Stops recording; the events are kept. Does nothing if this profiler is not active.
    void stop();
    /// @brief Whether this profiler is the active one.
    /// @return True between start() and stop().
    [[nodiscard]] bool active() const;
    /// @brief Removes all events (the profiler stays active or stopped).
    void clear();

    /// @brief A copy of the recorded events, in the order they ended.
    /// @return The events.
    [[nodiscard]] std::vector<ProfileEvent> events() const;
    /// @brief Microseconds between start() and stop() (or until now, while active).
    /// @return The wall time.
    [[nodiscard]] double wall_us() const;
    /// @brief Aggregates the events by (category, name), sorted by total time, largest first.
    /// @return One row per distinct (category, name).
    [[nodiscard]] std::vector<ProfileRow> summary() const;
    /// @brief The summary as a Markdown table (name, category, calls, total, self, mean, share of wall time).
    /// @param max_rows  Rows to show; the rest are folded into one "(other)" row. 0 shows all.
    /// @return The table, ending in a newline.
    [[nodiscard]] std::string to_markdown(std::size_t max_rows = 20) const;
    /// @brief The events in the Chrome trace-event JSON format ("X" complete events), viewable in
    ///        chrome://tracing or https://ui.perfetto.dev.
    /// @return The JSON text.
    [[nodiscard]] std::string to_chrome_trace() const;

    /// @brief Records a finished interval; used by ProfileScope. Ignored unless this profiler is active
    ///        and @p generation matches the current recording session.
    /// @param name  Event name.
    /// @param category  Event category.
    /// @param depth  Nesting depth on the calling thread.
    /// @param begin  Start time.
    /// @param end  End time.
    /// @param generation  The session number read when the interval began.
    void record(std::string name, std::string category, int depth, std::chrono::steady_clock::time_point begin,
                std::chrono::steady_clock::time_point end, std::uint64_t generation);

private:
    /// ProfileScope appends its event under the registry lock through append_locked().
    friend class ProfileScope;
    /// @brief Appends an event; the caller holds the profiler mutex and has checked the session.
    /// @param name  Event name.
    /// @param category  Event category.
    /// @param depth  Nesting depth.
    /// @param begin  Start time.
    /// @param end  End time.
    void append_locked(std::string name, std::string category, int depth, std::chrono::steady_clock::time_point begin,
                       std::chrono::steady_clock::time_point end);
    /// Clock origin set by start().
    std::chrono::steady_clock::time_point origin_{};
    /// Time stop() was called (meaningful once stopped).
    std::chrono::steady_clock::time_point stopped_{};
    /// Recorded events.
    std::vector<ProfileEvent> events_;
    /// Native thread ids in order of first appearance; the index is ProfileEvent::thread.
    std::vector<std::size_t> threads_;
};

/// @brief Whether some Profiler is currently active.
/// @return True while a profiler is between start() and stop().
[[nodiscard]] bool profiling();

/// @brief Times its own lifetime as one event of the active Profiler (if any).
///
/// Scopes nest: a scope created while another is alive on the same thread gets depth one
/// greater, and its time is subtracted from the outer scope's self time.
class ProfileScope {
public:
    /// @brief Starts timing when a profiler is active; otherwise does nothing.
    /// @param name  Event name (copied only when a profiler is active).
    /// @param category  Event category.
    explicit ProfileScope(std::string_view name, std::string_view category = "region");
    /// @brief Records the event (if timing started and the same profiler is still active).
    ~ProfileScope();
    /// @brief Disallowed: a scope times exactly one interval.
    ProfileScope(const ProfileScope&) = delete;
    /// @brief Disallowed: a scope times exactly one interval.
    ProfileScope& operator=(const ProfileScope&) = delete;

private:
    /// The profiler active at construction, or null.
    Profiler* profiler_{nullptr};
    /// Recording session number at construction.
    std::uint64_t generation_{0};
    /// Event name.
    std::string name_;
    /// Event category.
    std::string category_;
    /// Nesting depth on this thread.
    int depth_{0};
    /// Start time.
    std::chrono::steady_clock::time_point begin_{};
};

/// @brief Draws the events as a timeline: one lane per (thread, depth), one bar per event, coloured by
///        name, with up to twelve names, most time-consuming first, in the legend.
/// @param profiler  A profiler with at least one event.
/// @param title  Plot title.
/// @param from_us  Start of the time window shown (microseconds since start()).
/// @param to_us  End of the window; infinity (the default) means the profiler's wall time.
/// @return The plot (x axis in microseconds); bars are clipped to the window.
/// @throws std::invalid_argument  If there are no events in the window or the window is empty.
[[nodiscard]] graphics::RPlot plot_timeline(const Profiler& profiler, const std::string& title = "Profile timeline",
                                            double from_us = 0.0,
                                            double to_us   = std::numeric_limits<double>::infinity());

}  // namespace datamunge::nn
