classdef TaskBatch < datamunge.NativeObject
  % TaskBatch  Batched native InvertedPendulum or Hopper rollouts without a graphics dependency. CPU worlds use persistent parallel workers. Actions and all output arrays are finite double precision, with no NN dependency. Single-writer, including reset. Invalid actions fail before forces are applied. A native solver error propagates; reset selected worlds before retrying after such an error. Reset uses small independent uniform joint perturbations and never touches another row's RNG.
  %
  %   Example:
  %     dm::sim::TaskBatch task(dm::sim::Task::InvertedPendulum,4,dm::compute::Context({},2),42);
  %     const auto& result=task.step(std::vector<double>(4,0));
  %
  %   Methods:
  %     size_  Return the number of independent environments.
  %     observation_size  Read the task's observation width.
  %     action_size  Read the continuous action width.
  %     output  Borrow current rollout outputs without advancing the simulation.
  %     step  Advance selected task rows and retain final observations on episode end. Selected completed rows must be reset first. An unselected row remains unchanged.
  %     reset  Restore and perturb selected rows; retain final observations for rollout collectors. Clears rewards/end flags for selected rows. Random streams continue between episodes.
  %     seed  Reproduce fresh reset draws after explicit reseeding; physics is unchanged until reset.
  %
  %   Static methods:
  %     create  Create an owned native CPU task batch. task is InvertedPendulum or Hopper; count and workers must be positive. seed defaults to zero and time_limit to 1000 successful episode steps.
  %
  %   C++: class datamunge::sim::TaskBatch
  methods
    function obj = TaskBatch (varargin)
      % TaskBatch  Construct a C++ datamunge::sim::TaskBatch.
      %
      %   It has no public constructor: C++ functions return its objects.
      obj@datamunge.NativeObject(struct('rosetta_skip__', true));
      if nargin == 1 && isstruct (varargin{1}) && isfield (varargin{1}, 'rosetta_skip__')
        return;
      end
      error ('datamunge:binding', '%s', 'datamunge.TaskBatch has no public constructor');
    end
    function r = size_ (obj)
      % size_  Return the number of independent environments.
      %
      %   r = obj.size_ ()
      %
      %   Output:
      %     r  integer: Positive environment count.
      %
      %   C++: std::size_t datamunge::sim::TaskBatch::size() const
      r = __datamunge__(3175, obj);
    end
    function r = observation_size (obj)
      % observation_size  Read the task's observation width.
      %
      %   r = obj.observation_size ()
      %
      %   Output:
      %     r  integer: Four for InvertedPendulum or eleven for Hopper.
      %
      %   C++: std::size_t datamunge::sim::TaskBatch::observation_size() const
      r = __datamunge__(3176, obj);
    end
    function r = action_size (obj)
      % action_size  Read the continuous action width.
      %
      %   r = obj.action_size ()
      %
      %   Output:
      %     r  integer: One for InvertedPendulum or three for Hopper.
      %
      %   C++: std::size_t datamunge::sim::TaskBatch::action_size() const
      r = __datamunge__(3177, obj);
    end
    function r = output (obj)
      % output  Borrow current rollout outputs without advancing the simulation.
      %
      %   r = obj.output ()
      %
      %   Output:
      %     r  datamunge.TaskOutput: Internal reusable arrays, valid until destruction and modified by step/reset.
      %
      %   C++: const TaskOutput & datamunge::sim::TaskBatch::output() const
      r = __datamunge__(3178, obj);
    end
    function r = step (obj, varargin)
      % step  Advance selected task rows and retain final observations on episode end. Selected completed rows must be reset first. An unselected row remains unchanged.
      %
      %   r = obj.step (actions)
      %   r = obj.step (actions, mask)
      %
      %   Example:
      %     dm::sim::TaskBatch task(dm::sim::Task::Hopper,1);
      %     const auto& result=task.step({0,0,0});
      %
      %   Inputs:
      %     actions  double vector: Row-major [size(),action_size()] values in the documented action box.
      %     mask  integer vector, default the C++ default: Empty selects all; otherwise size() binary bytes.
      %
      %   Output:
      %     r  datamunge.TaskOutput: Borrowed reusable rollout output; no automatic reset.
      %
      %   Errors:
      %     datamunge:invalid_argument  Invalid action shape/value or mask, before mutation. (C++ std::invalid_argument)
      %     datamunge:native  A selected row already finished. (C++ std::logic_error)
      %     datamunge:native  Physics failure; explicit reset is required before retrying. (C++ std::exception)
      %
      %   C++: const TaskOutput & datamunge::sim::TaskBatch::step(const std::vector<double> & actions, const std::vector<std::uint8_t> & mask = {})
      r = __datamunge__(3179, obj, varargin{:});
    end
    function r = reset (obj, varargin)
      % reset  Restore and perturb selected rows; retain final observations for rollout collectors. Clears rewards/end flags for selected rows. Random streams continue between episodes.
      %
      %   r = obj.reset ()
      %   r = obj.reset (mask)
      %
      %   Example:
      %     dm::sim::TaskBatch task(dm::sim::Task::InvertedPendulum,2);task.reset({0,1});
      %
      %   Inputs:
      %     mask  integer vector, default the C++ default: Empty selects all; otherwise size() binary bytes.
      %
      %   Output:
      %     r  datamunge.TaskOutput: Borrowed reset observations and current unselected rows.
      %
      %   Errors:
      %     datamunge:invalid_argument  Invalid mask. (C++ std::invalid_argument)
      %
      %   C++: const TaskOutput & datamunge::sim::TaskBatch::reset(const std::vector<std::uint8_t> & mask = {})
      r = __datamunge__(3180, obj, varargin{:});
    end
    function seed (obj, seed)
      % seed  Reproduce fresh reset draws after explicit reseeding; physics is unchanged until reset.
      %
      %   obj.seed (seed)
      %
      %   Example:
      %     dm::sim::TaskBatch task(dm::sim::Task::Hopper,1);task.seed(7);task.reset();
      %
      %   Inputs:
      %     seed  integer: New batch random seed.
      %
      %   C++: void datamunge::sim::TaskBatch::seed(std::uint64_t seed)
      __datamunge__(3181, obj, seed);
    end
  end
  methods (Static)
    function r = create (varargin)
      % create  Create an owned native CPU task batch. task is InvertedPendulum or Hopper; count and workers must be positive. seed defaults to zero and time_limit to 1000 successful episode steps.
      %
      %   r = datamunge.TaskBatch.create (task, count)
      %   r = datamunge.TaskBatch.create (task, count, workers)
      %   r = datamunge.TaskBatch.create (task, count, workers, seed)
      %   r = datamunge.TaskBatch.create (task, count, workers, seed, time_limit)
      %
      %   Inputs:
      %     task  datamunge.Task value
      %     count  integer
      %     workers  integer, default 1
      %     seed  integer, default 0
      %     time_limit  integer, default 1000
      %
      %   Output:
      %     r  datamunge.TaskBatch or []
      %
      %   C++: datamunge::sim::TaskBatch * datamunge::octave_bindings::task_batch_create(datamunge::sim::Task task, std::size_t count, std::size_t workers = 1, std::uint64_t seed = 0, std::uint64_t time_limit = 1000)
      r = __datamunge__(3182, varargin{:});
    end
  end
end
% Generated by Rosetta from datamungeoctave/rosetta/bindings.hpp: class datamunge::sim::TaskBatch. Do not edit.
