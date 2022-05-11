import numpy as np


def tanh_backward(dA):

    dZ =1.0 - np.tanh(dA)**2
    return dZ

def jacMV_x(W , b, x, v):
    wx_b = W @ x + b
    act_deriv = tanh_backward(wx_b)
    diag_act_deriv = np.diag(act_deriv.reshape(act_deriv.shape[0],))
    diag_w = np.matmul(diag_act_deriv,W)
    return np.matmul(diag_w, v)




def jacMV_w(W,b, x, v):
    wx_b = W @ x + b
    act_deriv = tanh_backward(wx_b)
    diag_act = np.multiply(act_deriv, (v @ x))
    return diag_act

def jacMV_b(W,b, x, v):
    act_deriv = tanh_backward(np.add(np.matmul(W, x), b))
    diag_act_deriv = np.diag(act_deriv.reshape(act_deriv.shape[0], ))
    return diag_act_deriv @ v


def jacTMV_b(W,b, x, v):
    wx_b = W @ x + b
    grad_batch = np.multiply(tanh_backward(wx_b), v)
    return np.mean(grad_batch, axis=1).reshape(b.shape[0], b.shape[-1])



def jacTMV_x(W,b, x, v):
    wx_b = (W @ x) + b
    act_deriv = tanh_backward(wx_b)
    act_hadamard = np.multiply(act_deriv, v)
    return W.T @ act_hadamard

def jacTMV_w(W,b, x, v):
    wx_b = W @ x + b
    tan=tanh_backward(wx_b)
    return np.multiply(tan, v) @ x.T
