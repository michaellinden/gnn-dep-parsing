import dynet as dy
import math
from . import dy_model
from ..init import init_wrap


@dy_model
class Linear:
    "Construct a Affine Transformation."

    def __init__(
            self,
            model: dy.ParameterCollection,
            in_dim: int,
            out_dim: int,
            bias: bool = True,
            init: dy.PyInitializer = dy.GlorotInitializer(),
            is_original=False):

        self.is_original = is_original
        pc = model.add_subcollection()
        init = init_wrap(init, (out_dim, in_dim))
        if is_original:
            self.W = pc.add_parameters((out_dim, in_dim), init=init)
        else:
            self.W = pc.add_parameters((in_dim, out_dim), init=init)
        #self.W = pc.add_parameters((out_dim, in_dim), init=init)
        if bias:
            self.b = pc.add_parameters((out_dim,), init=0)
        self.pc = pc
        self.bias = bias
        self.spec = (in_dim, out_dim, bias, init)

    def __call__(self, x):
        b = self.b if self.bias else 0
        #print('----------------')
        #print(x.dim())
        #print(self.W.dim())


        if self.is_original:
            return self.W * x
        else:
            return x * self.W
        
        #return self.W * x + b 
