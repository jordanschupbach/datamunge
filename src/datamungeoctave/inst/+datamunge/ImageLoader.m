classdef ImageLoader < datamunge.NativeObject
  % ImageLoader  A configurable, disk-backed batch iterator over a standard image dataset.
  %
  %   obj = datamunge.ImageLoader (dataset)
  %   obj = datamunge.ImageLoader (dataset, split)
  %   obj = datamunge.ImageLoader (dataset, split, directory)
  %
  %   Datasets: "mnist" and "fashion_mnist" (28x28 grayscale, 10 classes; 60000 train / 10000
  %   test), "cifar10" (32x32 RGB, 10 classes; 50000 / 10000) and "cifar100" (32x32 RGB, 100 fine
  %   or 20 coarse classes; 50000 / 10000).
  %
  %   Iteration works in epochs. has_next()/next() walk the current epoch's order; reset() starts
  %   the next epoch (with a fresh shuffle when shuffling is on). Every setter returns the loader
  %   for chaining and rewinds the current epoch, so configure before iterating.
  %
  %   Reading is const and opens files per call, so batch()/take() may be called from several
  %   threads at once; next()/reset() and the setters mutate and must not race.
  %
  %   Example:
  %     dm::datasets::ImageLoader test("cifar10", "test");
  %     test.set_batch_size(500).set_layout("nchw").set_normalize(false);
  %     const auto first = test.next();          // shape {500, 3, 32, 32}, values 0..255
  %     std::print("{}\n", test.class_names()[first.labels[0]]);   // "cat"
  %
  %   Methods:
  %     set_batch_size  Images per batch (default 32).
  %     set_shuffle  Turns shuffling on or off (default off). Epoch e's order is a permutation drawn from (seed(), e), so the same seed always gives the same sequence of epochs.
  %     set_seed  Sets the shuffle seed (default 0); see set_shuffle().
  %     set_drop_last  When true, a final batch smaller than batch_size() is skipped (default false).
  %     set_normalize  true scales pixels to [0, 1] (default); false keeps the raw 0..255 values.
  %     set_layout  Memory layout of ImageBatch::pixels: "nhwc" (default), "nchw" or "flat".
  %     set_limit  Uses only the first `count` examples of the split (0, the default, means all). Handy for quick experiments. Shuffling then permutes just those examples.
  %     set_coarse_labels  CIFAR-100 only: true labels images with the 20 superclasses, false (default) with the 100 fine classes. Ignored by the other datasets.
  %     dataset  The dataset name given to the constructor.
  %     split  "train" or "test".
  %     directory  The directory the files are read from.
  %     batch_size  Images per batch.
  %     shuffle  Whether each epoch is shuffled.
  %     seed  The shuffle seed.
  %     drop_last  Whether a short final batch is skipped.
  %     normalize  Whether pixels are scaled to [0, 1].
  %     layout  "nhwc", "nchw" or "flat".
  %     coarse_labels  Whether CIFAR-100 uses coarse labels.
  %     size_  Examples per epoch: the split size, or the limit if one is set and smaller.
  %     split_size  Examples in the split on disk, ignoring any limit.
  %     num_batches  Batches per epoch, honoring drop_last().
  %     height  Image height in pixels.
  %     width  Image width in pixels.
  %     channels  Channels per pixel: 1 for MNIST-style, 3 (RGB) for CIFAR.
  %     num_classes  Number of label classes (CIFAR-100: 100, or 20 with coarse labels).
  %     class_names  Label names, indexed by label (e.g. "airplane" for CIFAR-10 label 0).
  %     epoch  The current epoch number, starting at 0 and advanced by reset().
  %     position  Index of the batch next() will return in this epoch.
  %     has_next  true while the current epoch has batches left.
  %     next  Reads the next batch of the current epoch.
  %     reset  Starts the next epoch: the position returns to 0 and, if shuffling, a new order is drawn.
  %     batch  Reads batch `index` of the current epoch without moving the position.
  %     take  Reads the listed examples (split positions), in the order given.
  %     load_all  Reads every example of the epoch (all of size()) into one batch, in split order. Mind memory: all 60000 MNIST images take ~376 MB as doubles; see set_limit().
  %
  %   C++: class datamunge::datasets::ImageLoader
  methods
    function obj = ImageLoader (varargin)
      % ImageLoader  Construct a C++ datamunge::datasets::ImageLoader.
      %
      %   obj = datamunge.ImageLoader (dataset)
      %   obj = datamunge.ImageLoader (dataset, split)
      %   obj = datamunge.ImageLoader (dataset, split, directory)
      %
      %   Opens a dataset split, downloading the files first if they are missing.
      %
      %   Inputs:
      %     dataset  char row: "mnist", "fashion_mnist", "cifar10" or "cifar100".
      %     split  char row, default 'train': "train" or "test".
      %     directory  char row, default '': Where the dataset's files are (or should be downloaded); empty means `data_home() + "/" + dataset`.
      %
      %   Errors:
      %     datamunge:invalid_argument  for an unknown dataset or split. (C++ std::invalid_argument)
      %     datamunge:native  if the files are missing and cannot be downloaded, or are not valid files of that dataset. (C++ std::runtime_error)
      %
      %   C++: datamunge::datasets::ImageLoader::ImageLoader(const std::string & dataset, const std::string & split = "train", const std::string & directory = "")
      obj@datamunge.NativeObject(struct('rosetta_skip__', true));
      if nargin == 1 && isstruct (varargin{1}) && isfield (varargin{1}, 'rosetta_skip__')
        return;
      end
      __datamunge__(-2, obj, 638, false, varargin{:});
    end
    function r = set_batch_size (obj, size)
      % set_batch_size  Images per batch (default 32).
      %
      %   r = obj.set_batch_size (size)
      %
      %   Inputs:
      %     size  integer
      %
      %   Output:
      %     r  datamunge.ImageLoader
      %
      %   Errors:
      %     datamunge:invalid_argument  if `size` is not positive. (C++ std::invalid_argument)
      %
      %   C++: ImageLoader & datamunge::datasets::ImageLoader::set_batch_size(int size)
      r = __datamunge__(604, obj, size);
    end
    function r = set_shuffle (obj, shuffle)
      % set_shuffle  Turns shuffling on or off (default off). Epoch e's order is a permutation drawn from (seed(), e), so the same seed always gives the same sequence of epochs.
      %
      %   r = obj.set_shuffle (shuffle)
      %
      %   Inputs:
      %     shuffle  logical
      %
      %   Output:
      %     r  datamunge.ImageLoader
      %
      %   C++: ImageLoader & datamunge::datasets::ImageLoader::set_shuffle(bool shuffle)
      r = __datamunge__(605, obj, shuffle);
    end
    function r = set_seed (obj, seed)
      % set_seed  Sets the shuffle seed (default 0); see set_shuffle().
      %
      %   r = obj.set_seed (seed)
      %
      %   Inputs:
      %     seed  integer
      %
      %   Output:
      %     r  datamunge.ImageLoader
      %
      %   C++: ImageLoader & datamunge::datasets::ImageLoader::set_seed(int seed)
      r = __datamunge__(606, obj, seed);
    end
    function r = set_drop_last (obj, drop_last)
      % set_drop_last  When true, a final batch smaller than batch_size() is skipped (default false).
      %
      %   r = obj.set_drop_last (drop_last)
      %
      %   Inputs:
      %     drop_last  logical
      %
      %   Output:
      %     r  datamunge.ImageLoader
      %
      %   C++: ImageLoader & datamunge::datasets::ImageLoader::set_drop_last(bool drop_last)
      r = __datamunge__(607, obj, drop_last);
    end
    function r = set_normalize (obj, normalize)
      % set_normalize  true scales pixels to [0, 1] (default); false keeps the raw 0..255 values.
      %
      %   r = obj.set_normalize (normalize)
      %
      %   Inputs:
      %     normalize  logical
      %
      %   Output:
      %     r  datamunge.ImageLoader
      %
      %   C++: ImageLoader & datamunge::datasets::ImageLoader::set_normalize(bool normalize)
      r = __datamunge__(608, obj, normalize);
    end
    function r = set_layout (obj, layout)
      % set_layout  Memory layout of ImageBatch::pixels: "nhwc" (default), "nchw" or "flat".
      %
      %   r = obj.set_layout (layout)
      %
      %   Inputs:
      %     layout  char row
      %
      %   Output:
      %     r  datamunge.ImageLoader
      %
      %   Errors:
      %     datamunge:invalid_argument  for any other value. (C++ std::invalid_argument)
      %
      %   C++: ImageLoader & datamunge::datasets::ImageLoader::set_layout(const std::string & layout)
      r = __datamunge__(609, obj, layout);
    end
    function r = set_limit (obj, count)
      % set_limit  Uses only the first `count` examples of the split (0, the default, means all). Handy for quick experiments. Shuffling then permutes just those examples.
      %
      %   r = obj.set_limit (count)
      %
      %   Inputs:
      %     count  integer
      %
      %   Output:
      %     r  datamunge.ImageLoader
      %
      %   Errors:
      %     datamunge:invalid_argument  if `count` is negative. (C++ std::invalid_argument)
      %
      %   C++: ImageLoader & datamunge::datasets::ImageLoader::set_limit(int count)
      r = __datamunge__(610, obj, count);
    end
    function r = set_coarse_labels (obj, coarse)
      % set_coarse_labels  CIFAR-100 only: true labels images with the 20 superclasses, false (default) with the 100 fine classes. Ignored by the other datasets.
      %
      %   r = obj.set_coarse_labels (coarse)
      %
      %   Inputs:
      %     coarse  logical
      %
      %   Output:
      %     r  datamunge.ImageLoader
      %
      %   C++: ImageLoader & datamunge::datasets::ImageLoader::set_coarse_labels(bool coarse)
      r = __datamunge__(611, obj, coarse);
    end
    function r = dataset (obj)
      % dataset  The dataset name given to the constructor.
      %
      %   r = obj.dataset ()
      %
      %   Output:
      %     r  char row
      %
      %   C++: const std::string & datamunge::datasets::ImageLoader::dataset() const
      r = __datamunge__(612, obj);
    end
    function r = split (obj)
      % split  "train" or "test".
      %
      %   r = obj.split ()
      %
      %   Output:
      %     r  char row
      %
      %   C++: const std::string & datamunge::datasets::ImageLoader::split() const
      r = __datamunge__(613, obj);
    end
    function r = directory (obj)
      % directory  The directory the files are read from.
      %
      %   r = obj.directory ()
      %
      %   Output:
      %     r  char row
      %
      %   C++: const std::string & datamunge::datasets::ImageLoader::directory() const
      r = __datamunge__(614, obj);
    end
    function r = batch_size (obj)
      % batch_size  Images per batch.
      %
      %   r = obj.batch_size ()
      %
      %   Output:
      %     r  integer
      %
      %   C++: int datamunge::datasets::ImageLoader::batch_size() const
      r = __datamunge__(615, obj);
    end
    function r = shuffle (obj)
      % shuffle  Whether each epoch is shuffled.
      %
      %   r = obj.shuffle ()
      %
      %   Output:
      %     r  logical
      %
      %   C++: bool datamunge::datasets::ImageLoader::shuffle() const
      r = __datamunge__(616, obj);
    end
    function r = seed (obj)
      % seed  The shuffle seed.
      %
      %   r = obj.seed ()
      %
      %   Output:
      %     r  integer
      %
      %   C++: int datamunge::datasets::ImageLoader::seed() const
      r = __datamunge__(617, obj);
    end
    function r = drop_last (obj)
      % drop_last  Whether a short final batch is skipped.
      %
      %   r = obj.drop_last ()
      %
      %   Output:
      %     r  logical
      %
      %   C++: bool datamunge::datasets::ImageLoader::drop_last() const
      r = __datamunge__(618, obj);
    end
    function r = normalize (obj)
      % normalize  Whether pixels are scaled to [0, 1].
      %
      %   r = obj.normalize ()
      %
      %   Output:
      %     r  logical
      %
      %   C++: bool datamunge::datasets::ImageLoader::normalize() const
      r = __datamunge__(619, obj);
    end
    function r = layout (obj)
      % layout  "nhwc", "nchw" or "flat".
      %
      %   r = obj.layout ()
      %
      %   Output:
      %     r  char row
      %
      %   C++: const std::string & datamunge::datasets::ImageLoader::layout() const
      r = __datamunge__(620, obj);
    end
    function r = coarse_labels (obj)
      % coarse_labels  Whether CIFAR-100 uses coarse labels.
      %
      %   r = obj.coarse_labels ()
      %
      %   Output:
      %     r  logical
      %
      %   C++: bool datamunge::datasets::ImageLoader::coarse_labels() const
      r = __datamunge__(621, obj);
    end
    function r = size_ (obj)
      % size_  Examples per epoch: the split size, or the limit if one is set and smaller.
      %
      %   r = obj.size_ ()
      %
      %   Output:
      %     r  integer
      %
      %   C++: int datamunge::datasets::ImageLoader::size() const
      r = __datamunge__(622, obj);
    end
    function r = split_size (obj)
      % split_size  Examples in the split on disk, ignoring any limit.
      %
      %   r = obj.split_size ()
      %
      %   Output:
      %     r  integer
      %
      %   C++: int datamunge::datasets::ImageLoader::split_size() const
      r = __datamunge__(623, obj);
    end
    function r = num_batches (obj)
      % num_batches  Batches per epoch, honoring drop_last().
      %
      %   r = obj.num_batches ()
      %
      %   Output:
      %     r  integer
      %
      %   C++: int datamunge::datasets::ImageLoader::num_batches() const
      r = __datamunge__(624, obj);
    end
    function r = height (obj)
      % height  Image height in pixels.
      %
      %   r = obj.height ()
      %
      %   Output:
      %     r  integer
      %
      %   C++: int datamunge::datasets::ImageLoader::height() const
      r = __datamunge__(625, obj);
    end
    function r = width (obj)
      % width  Image width in pixels.
      %
      %   r = obj.width ()
      %
      %   Output:
      %     r  integer
      %
      %   C++: int datamunge::datasets::ImageLoader::width() const
      r = __datamunge__(626, obj);
    end
    function r = channels (obj)
      % channels  Channels per pixel: 1 for MNIST-style, 3 (RGB) for CIFAR.
      %
      %   r = obj.channels ()
      %
      %   Output:
      %     r  integer
      %
      %   C++: int datamunge::datasets::ImageLoader::channels() const
      r = __datamunge__(627, obj);
    end
    function r = num_classes (obj)
      % num_classes  Number of label classes (CIFAR-100: 100, or 20 with coarse labels).
      %
      %   r = obj.num_classes ()
      %
      %   Output:
      %     r  integer
      %
      %   C++: int datamunge::datasets::ImageLoader::num_classes() const
      r = __datamunge__(628, obj);
    end
    function r = class_names (obj)
      % class_names  Label names, indexed by label (e.g. "airplane" for CIFAR-10 label 0).
      %
      %   r = obj.class_names ()
      %
      %   Output:
      %     r  cell array of char
      %
      %   C++: std::vector<std::string> datamunge::datasets::ImageLoader::class_names() const
      r = __datamunge__(629, obj);
    end
    function r = epoch (obj)
      % epoch  The current epoch number, starting at 0 and advanced by reset().
      %
      %   r = obj.epoch ()
      %
      %   Output:
      %     r  integer
      %
      %   C++: int datamunge::datasets::ImageLoader::epoch() const
      r = __datamunge__(630, obj);
    end
    function r = position (obj)
      % position  Index of the batch next() will return in this epoch.
      %
      %   r = obj.position ()
      %
      %   Output:
      %     r  integer
      %
      %   C++: int datamunge::datasets::ImageLoader::position() const
      r = __datamunge__(631, obj);
    end
    function r = has_next (obj)
      % has_next  true while the current epoch has batches left.
      %
      %   r = obj.has_next ()
      %
      %   Output:
      %     r  logical
      %
      %   C++: bool datamunge::datasets::ImageLoader::has_next() const
      r = __datamunge__(632, obj);
    end
    function r = next (obj)
      % next  Reads the next batch of the current epoch.
      %
      %   r = obj.next ()
      %
      %   Output:
      %     r  datamunge.ImageBatch
      %
      %   Errors:
      %     datamunge:out_of_range  when the epoch is finished (call reset() for the next one). (C++ std::out_of_range)
      %
      %   C++: ImageBatch datamunge::datasets::ImageLoader::next()
      r = __datamunge__(633, obj);
    end
    function reset (obj)
      % reset  Starts the next epoch: the position returns to 0 and, if shuffling, a new order is drawn.
      %
      %   obj.reset ()
      %
      %   C++: void datamunge::datasets::ImageLoader::reset()
      __datamunge__(634, obj);
    end
    function r = batch (obj, index)
      % batch  Reads batch `index` of the current epoch without moving the position.
      %
      %   r = obj.batch (index)
      %
      %   Inputs:
      %     index  integer
      %
      %   Output:
      %     r  datamunge.ImageBatch
      %
      %   Errors:
      %     datamunge:out_of_range  if `index` is not in [0, num_batches()). (C++ std::out_of_range)
      %
      %   C++: ImageBatch datamunge::datasets::ImageLoader::batch(int index) const
      r = __datamunge__(635, obj, index);
    end
    function r = take (obj, indices)
      % take  Reads the listed examples (split positions), in the order given.
      %
      %   r = obj.take (indices)
      %
      %   Inputs:
      %     indices  integer vector
      %
      %   Output:
      %     r  datamunge.ImageBatch
      %
      %   Errors:
      %     datamunge:out_of_range  if an index is not in [0, size()). (C++ std::out_of_range)
      %
      %   C++: ImageBatch datamunge::datasets::ImageLoader::take(const std::vector<int> & indices) const
      r = __datamunge__(636, obj, indices);
    end
    function r = load_all (obj)
      % load_all  Reads every example of the epoch (all of size()) into one batch, in split order. Mind memory: all 60000 MNIST images take ~376 MB as doubles; see set_limit().
      %
      %   r = obj.load_all ()
      %
      %   Output:
      %     r  datamunge.ImageBatch
      %
      %   C++: ImageBatch datamunge::datasets::ImageLoader::load_all() const
      r = __datamunge__(637, obj);
    end
  end
end
% Generated by Rosetta from datamungeoctave/rosetta/bindings.hpp: class datamunge::datasets::ImageLoader. Do not edit.
