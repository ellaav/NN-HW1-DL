import random

import numpy as np
import matplotlib.pyplot as plt
from mat4py import loadmat
from classifications_Tools import *


def sigmoid(Z):
    """
    Numpy sigmoid activation implementation
    Arguments:
    Z - numpy array of any shape
    Returns:
    A - output of sigmoid(z), same shape as Z
    cache -- returns Z as well, useful during backpropagation
    """
    A = 1 / (1 + np.exp(-Z))
    cache = Z
    return A, cache


def relu(Z):
    """
    Numpy Relu activation implementation
    Arguments:
    Z - Output of the linear layer, of any shape
    Returns:
    A - Post-activation parameter, of the same shape as Z
    cache - a python dictionary containing "A"; stored for computing the backward pass efficiently
    """
    A = np.maximum(0, Z)
    cache = Z
    return A, cache


def sigmoid_backward(dA, cache):
    """
    The backward propagation for a single SIGMOID unit.
    Arguments:
    dA - post-activation gradient, of any shape
    cache - 'Z' where we store for computing backward propagation efficiently
    Returns:
    dZ - Gradient of the cost with respect to Z
    """
    Z = cache
    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)
    return dZ


def relu_backward(dA, cache):
    """
    The backward propagation for a single RELU unit.
    Arguments:
    dA - post-activation gradient, of any shape
    cache - 'Z' where we store for computing backward propagation efficiently
    Returns:
    dZ - Gradient of the cost with respect to Z
    """
    Z = cache
    dZ = np.array(dA, copy=True)  # just converting dz to a correct object.
    # When z <= 0, you should set dz to 0 as well.
    dZ[Z <= 0] = 0
    return dZ


def initialize_parameters(input_layer, hidden_layer, output_layer):
    # initialize 1st layer output and input with random values
    W1 = np.random.randn(hidden_layer, input_layer) * 0.01
    # initialize 1st layer output bias
    b1 = np.zeros((hidden_layer, 1))
    # initialize 2nd layer output and input with random values
    W2 = np.random.randn(output_layer, hidden_layer) * 0.01
    # initialize 2nd layer output bias
    b2 = np.zeros((output_layer, 1))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters


def initialize_parameters_deep(layer_dimension):
    parameters = {}

    L = len(layer_dimension)

    for l in range(1, L):
        parameters["W" + str(l)] = np.random.randn(layer_dimension[l], layer_dimension[l - 1]) * 0.01
        parameters["b" + str(l)] = np.zeros((layer_dimension[l], 1))

    return parameters


def linear_forward(A, W, b):
    Z = np.dot(W, A) + b

    cache = (A, W, b)

    return Z, cache


def linear_activation_forward(A_prev, W, b, activation):
    if activation == "sigmoid":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)

    elif activation == "relu":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)

    cache = (linear_cache, activation_cache)

    return A, cache


def L_model_forward(X, parameters):
    caches = []
    A = X

    # number of layers in the neural network
    L = len(parameters) // 2

    # Using a for loop to replicate [LINEAR->RELU] (L-1) times
    for l in range(1, L):
        A_prev = A

        # Implementation of LINEAR -> RELU.
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)],
                                             activation="relu")

        # Adding "cache" to the "caches" list.
        caches.append(cache)

    # Implementation of LINEAR -> SIGMOID.
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation="sigmoid")

    # Adding "cache" to the "caches" list.
    caches.append(cache)

    return AL, caches


def compute_cost(AL, Y):
    # number of examples
    m = Y.shape[1]

    # Compute loss from AL and y.
    cost = -np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL)) / m

    # To make sure our cost's shape is what we expect (e.g. this turns [[23]] into 23).
    cost = np.squeeze(cost)

    return cost


def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = 1. / m * np.dot(dZ, A_prev.T)
    db = 1. / m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache

    if activation == "relu":
        dZ = relu_backward(dA, cache[1])
        dA_prev, dW, db = linear_backward(dZ, cache[0])

    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, cache[1])
        dA_prev, dW, db = linear_backward(dZ, cache[0])

    return dA_prev, dW, db


def L_model_backward(AL, Y, caches):
    grads = {}

    # the number of layers
    L = len(caches)
    m = AL.shape[1]

    # after this line, Y is the same shape as AL
    Y = Y.reshape(AL.shape)

    # Initializing the backpropagation
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "dAL, current_cache". Outputs: "grads["dAL-1"], grads["dWL"], grads["dbL"]
    current_cache = caches[L - 1]
    grads["dA" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL,
                                                                                                      current_cache,
                                                                                                      "sigmoid")

    # Loop from l=L-2 to l=0
    for l in reversed(range(L - 1)):
        # lth layer: (RELU -> LINEAR) gradients.
        # Inputs: "grads["dA" + str(l + 1)], current_cache".
        # Outputs: "grads["dA" + str(l)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)]

        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache, "relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads


def update_parameters(parameters, grads, learning_rate):
	# number of layers in the neural network
    L = len(parameters) // 2

    # Update rule for each parameter
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate*grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate*grads["db" + str(l+1)]

    return parameters


def L_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False):  # lr was 0.009
    # keep track of cost
    costs = []

    # Parameters initialization.
    parameters = initialize_parameters_deep(layers_dims)

    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        AL, caches = L_model_forward(X, parameters)

        # Compute cost.
        cost = compute_cost(AL, Y)
        pred= predict(X,parameters)
        lab = np.argmax(Y, axis=0)
        acc = lab == pred
        print(np.mean(acc) * 100)



        # Backward propagation.
        grads = L_model_backward(AL, Y, caches)


        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)

        # Print the cost every 100 training example
        if print_cost :#and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))
            # print("Loss after iteration %i: %f" % (i, loss))
        if print_cost :#and i % 100 == 0:
            costs.append(cost)

    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters


def predict(X, parameters):
    m = X.shape[1]

    # number of layers in the neural network
    n = len(parameters) // 2
    p = np.zeros((1, m))

    # Forward propagation
    probas, caches = L_model_forward(X, parameters)

    # convert probas to 0/1 predictions
    for i in range(0, probas.shape[1]):
        if probas[0, i] > 0.5:
            p[0, i] = 1
        else:
            p[0, i] = 0

    return p
def run(file):
    data = loadmat(file)

    Ctrain = np.array(data['Ct'])
    Ctrain = Ctrain.T
    Xtrain = np.array(data['Yt'])#[:,1000:2000]
    c_v = np.array(data['Cv'])
    c_v = c_v.T
    x_v = np.array(data['Yv'])


    layer_dims = [Xtrain.shape[0],15,20, Ctrain.shape[-1]]
    parameters = initialize_parameters_deep(layer_dims)
    X = np.random.rand(4, 4)
    Y = np.array([[1, 1, 0, 0]])
    AL, caches = L_model_forward(Xtrain, parameters)

    print("X.shape =", Xtrain.shape)
    print("AL =", AL)
    print("Lenght of caches list = ", len(caches))
    print("parameters:", parameters)
    # print("cost = ", compute_cost(AL, Ctrain.T))

    parameter=L_layer_model(Xtrain,Ctrain.T,layer_dims,learning_rate=0.075,num_iterations=40,print_cost=True)
    # pred=predict(Xtrain,parameters)
    # lab = np.argmax(Ctrain, axis=1)
    # acc = lab == pred
    # print(np.mean(acc) * 100)

    # print(loss_function(AL,Xtrain,Ctrain))
    train_acc=[]
    test_acc=[]
    loss=[]
    # for i in range(len(parameter)):
    #     (linear_cache, activation_cache)= caches[i]
    #     A,w,b=linear_cache
    #     # wi=parameter['W'+str(i+1)]
    #     # prediction = softmax(w, A)
    #     # prediction = np.argmax(A.T, axis=1)
    #     lab = np.argmax(Ctrain, axis=1)
    #     # acc = lab == prediction
    #     # train_acc.append(np.mean(acc) * 100)
    #     # loss.append(loss_function(w,A,Ctrain))
    # print(train_acc)
    # print(loss)


run('HW1_Data\GMMData.mat')