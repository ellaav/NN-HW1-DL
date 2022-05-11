
import numpy as np


def tanh_backward(dA):

    dZ =1.0 - np.tanh(dA)**2
    return dZ




def jacTMV_b(W1,W2,b, x, v):
    wx_b = (W1 @ x) + b
    act_deriv = tanh_backward(wx_b)
    m = W2.T @ v
    return  act_deriv*m


def jacTMV_x(W1,W2,b, x, v):
    wx_b = (W1 @ x) + b
    act_deriv = tanh_backward(wx_b)
    m=W2.T@v
    act_hadamard = act_deriv*m
    act=W1.T @ act_hadamard
    return v+act

def jacTMV_w1(W1,W2,b, x, v):
    wx_b = (W1 @ x) + b
    act_deriv = tanh_backward(wx_b)
    m = W2.T @ v
    act_hadamard = act_deriv * m
    return act_hadamard@ x.T


def jacTMV_w2(W1,W2,b, x, v):
    wx_b = W1 @ x + b
    tan=tanh_backward(wx_b)
    return v@tan.T

def jacMV_x(W1,W2 , b, x, v):
    wx_b = W1 @ x + b
    act_deriv = tanh_backward(wx_b)
    diag_act_deriv = np.diag(act_deriv.reshape(act_deriv.shape[0],))
    diag_w = np.matmul(diag_act_deriv,W1)
    d=np.eye(v.shape[0])+W2@diag_w
    return np.matmul(d, v)




def jacMV_w1(W1,W2,b, x, v):
    wx_b = W1 @ x + b
    act_deriv = tanh_backward(wx_b)
    diag_act = np.multiply(act_deriv, (v @ x))
    return W2@diag_act

def jacMV_w2(W1,W2,b, x, v):
    wx_b = W1 @ x + b
    act=np.tanh(wx_b)
    return v @ act


def jacMV_b(W1,W2,b, x, v):
    act_deriv = tanh_backward(np.add(np.matmul(W1, x), b))
    diag_act_deriv = np.diag(act_deriv.reshape(act_deriv.shape[0], ))
    return (W2@diag_act_deriv )@ v


def jacTMV_b(W1,W2,b, x, v):
    wx_b = W1 @ x + b
    grad_batch = np.multiply(tanh_backward(wx_b), v)
    return np.mean(grad_batch, axis=1).reshape(b.shape[0], b.shape[-1])


