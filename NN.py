import numpy as np
from classifications_Tools import *
from derivatives import *
from layer_class import layer
class NN:
    def __init__(self,l_dim):
        self.linear_layer=layer(l_dim[1],l_dim[0])
        self.W_loss=np.random.rand(l_dim[-2],l_dim[-1])* np.sqrt(2 / l_dim[-1])
        self.grad_w=None
        self.deep_layes=self.initialize_layers(l_dim[1],len(l_dim)-2)
        # self.params=self.initialize_parameters_deep(l_dim[1],len(l_dim)-2)
        self.L=len(l_dim)-2
        self.activation_cache=[]

    def initialize_layers(self,layer_dim,L):
        layers=[]
        for l in range(1, L):
            layers.append(layer(layer_dim, layer_dim))
        return layers




    def update_parameters(self ,lr):
        self.linear_layer.W=self.linear_layer.W - lr * self.linear_layer.g_w
        for l in self.deep_layes:
            l.W = l.W - lr * l.g_w
        self.W_loss=self.W_loss-lr*self.grad_w



    def L_model_forward(self,X):
        A = X

        self.activation_cache = [A]
        # save labels for backward pass



        self.activation_cache.append(self.linear_layer.activate(A))

        for layer in self.deep_layes:
            self.activation_cache.append(layer.activate(self.activation_cache[-1]))

        return self.activation_cache[-1]





    def L_model_backward(self, Y):
        hidden_units = self.activation_cache
        hidden_units.reverse()

        # cross entropy grad
        self.grad_w=gradient_w(self.W_loss,hidden_units[0],Y)
        g_inp=grad_inp(self.W_loss,hidden_units[0],Y)


        # linear layers grads in reverse order
        for i, layer in enumerate(reversed(self.deep_layes), 1):
            layer.backward(hidden_units[i], g_inp)
            inp_grads = layer.g_x

        self.linear_layer.backward(hidden_units[-1], inp_grads)

        # the number of layers


