classdef Image < datamunge.NativeObject
  % Image  A 2D raster image: row-major, top-left origin, interleaved channel bytes (matching the layout every common image codec uses on disk, so I/O needs no reshuffling). Pixel data is always stored as the mode's exact channel count (1/2/3/4 bytes per pixel) -- get_pixel()/set_pixel() are the only place mode-specific expansion happens.
  %
  %   obj = datamunge.Image ()
  %   obj = datamunge.Image (width, height)
  %   obj = datamunge.Image (width, height, mode)
  %   obj = datamunge.Image (width, height, mode, fill_color)
  %
  %   Methods:
  %     width  C++ datamunge::image::Image::width.
  %     height  C++ datamunge::image::Image::height.
  %     mode  C++ datamunge::image::Image::mode.
  %     channels  C++ datamunge::image::Image::channels.
  %     data  C++ datamunge::image::Image::data.
  %     get_pixel  C++ datamunge::image::Image::get_pixel.
  %     set_pixel  C++ datamunge::image::Image::set_pixel.
  %     fill  C++ datamunge::image::Image::fill.
  %
  %   Static methods:
  %     channel_count  C++ datamunge::image::Image::channel_count.
  %
  %   C++: class datamunge::image::Image
  methods
    function obj = Image (varargin)
      % Image  Construct a C++ datamunge::image::Image.
      %
      %   obj = datamunge.Image ()
      %   obj = datamunge.Image (width, height)
      %   obj = datamunge.Image (width, height, mode)
      %   obj = datamunge.Image (width, height, mode, fill_color)
      %
      %   C++: datamunge::image::Image::Image()
      %   C++: datamunge::image::Image::Image(int width, int height, ImageMode mode = RGB)
      %   C++: datamunge::image::Image::Image(int width, int height, ImageMode mode, const Pixel & fill_color)
      obj@datamunge.NativeObject(struct('rosetta_skip__', true));
      if nargin == 1 && isstruct (varargin{1}) && isfield (varargin{1}, 'rosetta_skip__')
        return;
      end
      __datamunge__(-2, obj, 529, false, varargin{:});
    end
    function r = width (obj)
      % width  C++ datamunge::image::Image::width.
      %
      %   r = obj.width ()
      %
      %   Output:
      %     r  integer
      %
      %   C++: int datamunge::image::Image::width() const
      r = __datamunge__(520, obj);
    end
    function r = height (obj)
      % height  C++ datamunge::image::Image::height.
      %
      %   r = obj.height ()
      %
      %   Output:
      %     r  integer
      %
      %   C++: int datamunge::image::Image::height() const
      r = __datamunge__(521, obj);
    end
    function r = mode (obj)
      % mode  C++ datamunge::image::Image::mode.
      %
      %   r = obj.mode ()
      %
      %   Output:
      %     r  datamunge.ImageMode value
      %
      %   C++: ImageMode datamunge::image::Image::mode() const
      r = __datamunge__(522, obj);
    end
    function r = channels (obj)
      % channels  C++ datamunge::image::Image::channels.
      %
      %   r = obj.channels ()
      %
      %   Output:
      %     r  integer
      %
      %   C++: int datamunge::image::Image::channels() const
      r = __datamunge__(523, obj);
    end
    function r = data (obj, varargin)
      % data  C++ datamunge::image::Image::data.
      %
      %   r = obj.data ()
      %
      %   Overloads are chosen at run time from the argument types (C++):
      %   C++: const std::vector<std::uint8_t> & datamunge::image::Image::data() const
      %   C++: std::vector<std::uint8_t> & datamunge::image::Image::data()
      r = __datamunge__(524, obj, varargin{:});
    end
    function r = get_pixel (obj, x, y)
      % get_pixel  C++ datamunge::image::Image::get_pixel.
      %
      %   r = obj.get_pixel (x, y)
      %
      %   Inputs:
      %     x  integer
      %     y  integer
      %
      %   Output:
      %     r  datamunge.Pixel
      %
      %   C++: Pixel datamunge::image::Image::get_pixel(int x, int y) const
      r = __datamunge__(525, obj, x, y);
    end
    function set_pixel (obj, x, y, p)
      % set_pixel  C++ datamunge::image::Image::set_pixel.
      %
      %   obj.set_pixel (x, y, p)
      %
      %   Inputs:
      %     x  integer
      %     y  integer
      %     p  datamunge.Pixel
      %
      %   C++: void datamunge::image::Image::set_pixel(int x, int y, const Pixel & p)
      __datamunge__(526, obj, x, y, p);
    end
    function fill (obj, p)
      % fill  C++ datamunge::image::Image::fill.
      %
      %   obj.fill (p)
      %
      %   Inputs:
      %     p  datamunge.Pixel
      %
      %   C++: void datamunge::image::Image::fill(const Pixel & p)
      __datamunge__(527, obj, p);
    end
  end
  methods (Static)
    function r = channel_count (mode)
      % channel_count  C++ datamunge::image::Image::channel_count.
      %
      %   r = datamunge.Image.channel_count (mode)
      %
      %   Inputs:
      %     mode  datamunge.ImageMode value
      %
      %   Output:
      %     r  integer
      %
      %   C++: int datamunge::image::Image::channel_count(ImageMode mode)
      r = __datamunge__(528, mode);
    end
  end
end
% Generated by Rosetta from datamungeoctave/rosetta/bindings.hpp: class datamunge::image::Image. Do not edit.
