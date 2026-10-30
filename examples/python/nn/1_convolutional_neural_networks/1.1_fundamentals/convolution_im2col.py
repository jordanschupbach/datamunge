# Native Python NN example. Patch extraction and checks execute in Python.
# CMD: nix develop .#python --command env -u PYTHONPATH build/venv/pydatamunge-run/bin/python examples/python/nn/1_convolutional_neural_networks/1.1_fundamentals/convolution_im2col.py
"""Convolution as explicit NHWC im2col, core GEMM, and additive col2im."""
import math
from pydatamunge import nn


class PatchGeometry:
    def __init__(self, shape, kh, kw, sh=1, sw=1, dh=1, dw=1, padding='VALID'):
        if len(shape) != 4 or min(*shape,kh,kw,sh,sw,dh,dw) <= 0 or padding not in ('VALID','SAME'):
            raise ValueError('invalid patch geometry')
        self.shape = tuple(shape)
        n,h,w,c = shape
        eh,ew = (kh-1)*dh+1,(kw-1)*dw+1
        oh = (h+sh-1)//sh if padding=='SAME' else max(0,(h-eh)//sh+1)
        ow = (w+sw-1)//sw if padding=='SAME' else max(0,(w-ew)//sw+1)
        if not oh or not ow:
            raise ValueError('patch geometry has no output positions')
        top = max(0,(oh-1)*sh+eh-h)//2 if padding=='SAME' else 0
        left = max(0,(ow-1)*sw+ew-w)//2 if padding=='SAME' else 0
        self.rows,self.columns = n*oh*ow,kh*kw*c
        self.output = [n,oh,ow]
        self.indices = []
        for b in range(n):
            for y in range(oh):
                for x in range(ow):
                    for ky in range(kh):
                        for kx in range(kw):
                            iy,ix = y*sh+ky*dh-top,x*sw+kx*dw-left
                            if 0 <= iy < h and 0 <= ix < w:
                                for channel in range(c):
                                    self.indices.append(((((b*oh+y)*ow+x)*kh+ky)*kw*c+kx*c+channel,
                                                         ((b*h+iy)*w+ix)*c+channel))

    def im2col(self, images):
        if tuple(images.shape()) != self.shape or images.dtype()!=nn.DType.Float32:
            raise ValueError('matching float32 NHWC input required')
        source = images.values()
        output = [0.]*(self.rows*self.columns)
        for column,pixel in self.indices:
            output[column] = source[pixel]
        return nn.from_vector(output,[self.rows,self.columns])

    def col2im(self, columns):
        if tuple(columns.shape())!=(self.rows,self.columns) or columns.dtype()!=nn.DType.Float32:
            raise ValueError('matching float32 patch matrix required')
        source = columns.values()
        output = [0.]*math.prod(self.shape)
        for column,pixel in self.indices:
            output[pixel] += source[column]
        return nn.from_vector(output,self.shape)


def difference(a,b):
    assert tuple(a.shape())==tuple(b.shape())
    return max(abs(x-y) for x,y in zip(a.values(),b.values()))


def compare(shape,kh,kw,outputs,sh,sw,dh,dw,padding,generator):
    geometry = PatchGeometry(shape,kh,kw,sh,sw,dh,dw,padding)
    filter_shape = [kh,kw,shape[-1],outputs]
    output_shape = geometry.output+[outputs]
    # Use the same bound random stream in Python and R (the C++ reference uses mt19937).
    x = nn.normal(shape,0.,.3,generator)
    kernel = nn.normal(filter_shape,0.,.3,generator)
    upstream = nn.normal(output_shape,0.,.3,generator)
    columns = geometry.im2col(x)
    matrix = nn.reshape(kernel,[geometry.columns,outputs])
    manual_of = lambda patches: nn.reshape(nn.matmul(patches,matrix),output_shape)
    manual = manual_of(columns)
    options = nn.Conv2DOptions()
    options.strides,options.dilations = [sh,sw],[dh,dw]
    options.padding = nn.Padding.Same if padding=='SAME' else nn.Padding.Valid
    native = nn.conv2d(x,kernel,options)
    dy = nn.reshape(upstream,[geometry.rows,outputs])
    dcolumns = nn.matmul(dy,matrix,False,True)
    dx = geometry.col2im(dcolumns)
    dw_manual = nn.reshape(nn.matmul(columns,dy,True,False),filter_shape)
    errors = [difference(manual,native),
              difference(dx,nn.conv2d_backprop_input(shape,kernel,upstream,options)),
              difference(dw_manual,nn.conv2d_backprop_filter(x,filter_shape,upstream,options))]
    assert max(errors)<2e-6
    expanded_shape = [(kh-1)*dh+1,(kw-1)*dw+1,shape[-1],outputs]
    expanded = [0.]*math.prod(expanded_shape)
    original = kernel.values()
    indices = []
    for a in range(kh):
        for b in range(kw):
            for c in range(shape[-1]):
                for o in range(outputs):
                    source = ((a*kw+b)*shape[-1]+c)*outputs+o
                    target = ((a*dh*expanded_shape[1]+b*dw)*shape[-1]+c)*outputs+o
                    expanded[target]=original[source]
                    indices.append(target)
    expanded_kernel = nn.from_vector(expanded,expanded_shape)
    plain = nn.Conv2DOptions()
    plain.strides,plain.padding = [sh,sw],options.padding
    assert difference(dx,nn.conv2d_backprop_input(shape,expanded_kernel,upstream,plain))<2e-6
    expanded_dw = nn.conv2d_backprop_filter(x,expanded_shape,upstream,plain).values()
    assert difference(dw_manual,nn.from_vector([expanded_dw[i] for i in indices],filter_shape))<2e-6
    lhs = sum(a*b for a,b in zip(columns.values(),dcolumns.values()))
    rhs = sum(a*b for a,b in zip(x.values(),dx.values()))
    assert abs(lhs-rhs)<2e-6
    finite_error = 0.
    for index in [0,x.numel()//2,x.numel()-1]:
        values = []
        for delta in [-.002,.002]:
            changed = list(x.values());changed[index] += delta
            output = manual_of(geometry.im2col(nn.from_vector(changed,shape)))
            values.append(sum(a*b for a,b in zip(output.values(),upstream.values())))
        finite_error=max(finite_error,abs((values[1]-values[0])/.004-dx.values()[index]))
    assert finite_error<3e-5
    print(f'{padding} {shape}: errors(y,dx,dw,finite)={errors+[finite_error]}')


def main(arguments=None):
    generator = nn.Generator(149)
    for case in [([2,5,6,2],3,2,3,1,1,1,1,'VALID'),([2,5,6,2],3,2,3,2,2,1,1,'SAME'),
                 ([1,6,7,3],2,3,2,2,1,1,1,'SAME'),([1,7,8,2],3,2,3,1,1,2,2,'VALID'),
                 ([1,7,8,2],3,2,3,1,1,2,2,'SAME'),([2,4,5,3],1,1,4,1,1,1,1,'VALID')]:
        compare(*case,generator)
    geometry=PatchGeometry([1,5,5,1],3,3)
    coverage=geometry.col2im(geometry.im2col(nn.ones([1,5,5,1])))
    assert coverage.at([0,0,0,0])==1 and coverage.at([0,2,2,0])==9
    try:
        PatchGeometry([1,5,5,1],0,3)
    except ValueError:
        pass
    else:
        raise AssertionError('invalid patch kernel must be rejected')
    print('All im2col convolution checks passed.')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
