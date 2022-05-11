
from derivatives import *

class layer:
    def __init__(self,dim_in,dim_out):
        self.W=np.random.rand(dim_in,dim_out)
        self.b=np.random.rand(dim_in,1)
        self.g_x = None
        self.g_w = None
        self.g_b = None


    def activate(self,x):
        return np.tanh((self.W @ x) + self.b)
    def backward(self,x,v):
        self.g_x = jacTMV_x(self.W,self.b,x,v)
        self.g_w = jacTMV_w(self.W,self.b,x,v)
        self.g_b =jacTMV_b(self.W,self.b,x,v)