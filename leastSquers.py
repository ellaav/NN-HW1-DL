import numpy as np


def create_example():
    x=np.linspace(0.01, 1, num=198)
    a=0.8
    b=0.4
    epsilon=0.3 *np.random.randn(len(x)) / np.sqrt(x)
    y=a*x+b +epsilon
    c=np.array(list(map(lambda x : [1,0] if x>0  else [0,1],y-a*x+b)))
    xx=np.array([x,y])
    return xx ,c







