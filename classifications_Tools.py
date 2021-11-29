import random

import numpy as np
import matplotlib.pyplot as plt
from mat4py import loadmat
from leastSquers import *
from random import shuffle



def get_rand_w(x, c):
    return np.random.randn(x.shape[0], c.shape[-1])


def softmax(w, x):
    xt_w = x.T @ w
    eta = np.max(xt_w)
    exp = np.exp(xt_w - eta)
    return exp / np.sum(exp, axis=1).reshape(-1, 1)


def loss_function(w, x, c):
    m = x.shape[-1]
    soft_max = softmax(w, x)
    logs = np.log(soft_max)
    z = c.T @ logs
    # res = np.sum(z)
    # TODO: understand if need sum or trace
    res=np.trace(z)

    return -1 / m * res


def gradient_w(w, x, c):
    m = x.shape[-1]
    exps = np.exp(x.T @ w)
    sums = np.array(np.sum(exps, axis=1))
    diag = np.diag(sums)
    diag_inv = np.linalg.inv(diag)
    z = diag_inv @ exps
    z = np.subtract(z, c)
    return 1 / m * x @ z


def gradient_test(file):
    data = loadmat(file)

    c = np.array(data['Ct'])
    c = c.T
    x = np.array(data['Yt'])
    w = get_rand_w(x,c)
    d = get_rand_w(x,c)
    d_n = d / np.linalg.norm(d)
    loss_factor_w = loss_function(w, x, c)
    grad_w = gradient_w(w, x, c)
    epsilon_vals = np.geomspace(0.5, 0.5 ** 20, 20)
    without_g, with_g = [], []
    for eps in epsilon_vals:
        eps_d = eps * d_n
        w_tag = w + eps_d
        loss_factor_w_tag = loss_function(w_tag, x, c)

        o_eps = abs(loss_factor_w_tag - loss_factor_w)
        o_eps_sq = abs(loss_factor_w_tag - loss_factor_w - eps_d.ravel().T @ grad_w.ravel())
        print(o_eps)
        print(o_eps_sq)
        without_g.append(o_eps)
        with_g.append(o_eps_sq)

    l = range(20)
    plt.title('gradient test')
    plt.plot(l, without_g, label='First Order')
    plt.plot(l, with_g, label='Second Order')
    plt.yscale('log')
    plt.legend()
    plt.show()


def gradient_test2():
    x = np.random.rand(5, 10)  # 5 features, 10 samples

    d = np.random.rand(5, 3)  # 5 features, 3 labels
    d = d / np.linalg.norm(d)

    w = np.random.rand(5, 3)  # 5 features, 3 labels

    labels = np.random.randint(3, size=10)  # random draw of labels for 10 samples
    c = np.zeros((labels.size, 3))  # 10 samples, 3 labels
    c[np.arange(labels.size), labels] = 1  # columns in c are one-hot encoded


    loss_factor_w = loss_function(w, x, c)
    grad_w = gradient_w(w, x, c)
    epsilon_vals = np.geomspace(0.5, 0.5 ** 20, 20)
    without_g, with_g = [], []
    for eps in epsilon_vals:
        eps_d = eps * d
        w_tag = w + eps_d
        loss_factor_w_tag = loss_function(w_tag, x, c)

        o_eps = abs(loss_factor_w_tag - loss_factor_w)
        o_eps_sq = abs(loss_factor_w_tag - loss_factor_w - eps_d.ravel().T @ grad_w.ravel())
        print(o_eps)
        print(o_eps_sq)
        without_g.append(o_eps)
        with_g.append(o_eps_sq)

    l = range(20)
    plt.title('gradient test')
    plt.plot(l, without_g, label='First Order')
    plt.plot(l, with_g, label='Second Order')
    plt.yscale('log')
    plt.legend()
    plt.show()



def SGD_one_step(x, c,w,lr):
    g = gradient_w(w, x, c)
    return w - lr * g


def sgd_test(file,k,lr):
    data = loadmat(file)

    c = np.array(data['Ct'])
    c = c.T
    x = np.array(data['Yt'])
    c_v = np.array(data['Cv'])
    c_v = c_v.T
    x_v = np.array(data['Yv'])
    m = x.shape[-1]
    if k>m:
        k=m

    j = int(m / k)

    X,C=creat_batces(x,c,k)

    w = get_rand_w(x, c)
    train_acc=[]
    test_acc = []
    k = int(m / j)  # if m%k!=0
    pred = softmax(w, x)
    prediction = np.argmax(pred, axis=1)
    labels = np.argmax(c, axis=1)
    acc = labels == prediction
    train_acc.append(np.mean(acc) * 100)
    pred = softmax(w, x_v)
    prediction = np.argmax(pred, axis=1)
    labels = np.argmax(c_v, axis=1)
    acc = labels == prediction
    test_acc.append(np.mean(acc) * 100)
    loss = []
    loss.append(loss_function(w,x,c))

    for i in range(k):
        gradient_w(w, X[i], C[i])
        w = SGD_one_step(X[i], C[i], w, lr)
        pred = softmax(w, x)

        # calculate train error

        prediction = np.argmax(pred,axis=1)
        labels=np.argmax(c,axis=1)
        acc = labels == prediction
        train_acc.append(np.mean(acc) * 100)
        pred = softmax(w, x_v)
        prediction = np.argmax(pred, axis=1)
        labels = np.argmax(c_v, axis=1)
        acc = labels == prediction
        test_acc.append(np.mean(acc) * 100)
        loss.append(loss_function(w, x, c))



    return train_acc,test_acc,loss

def shuffle(x):
    return np.array(sorted(x, key=lambda k: random.random()))

def creat_batces(x,c,k):
    n,m=x.shape
    j=int(m/k)
    x=(shuffle(x.T)).T
    c=(shuffle(c)).T

    if j==0:
        X=[x[:,i] for i in range(m)]
        C = [c[:, i] for i in range(m)]
        C = [ci.T for ci in C]
        return X,C


    X=[x[:,i:i+j] for i in range(0,m,j)]
    C=[c[:,i:i+j] for i in range(0,m,j)]
    C=[ci.T for ci in C]

    return X,C
    # C=[ci.T for ci in C





def SGD_test_least_squers():
    lr=0.5
    x, c = create_example()
    x = np.array(x)
    c = np.array(c)
    Xtrain = x[:, :150]
    Ctrain = c[:150]
    Xtest = x[:, 150:]
    Ctest = c[150:]
    batches=np.array_split(Xtrain,10,axis=1)
    labels=np.array_split(Ctrain.T,10,axis=1)
    labels=[l.T for l in labels]
    w=get_rand_w(Xtrain,Ctrain)
    train_acc=[]
    test_acc=[]
    prediction = softmax(w, Xtrain)
    prediction = np.argmax(prediction, axis=1)
    lab = np.argmax(Ctrain, axis=1)
    acc = lab == prediction
    train_acc.append(np.mean(acc))
    prediction_test = softmax(w, Xtest)
    prediction_test = np.argmax(prediction_test, axis=1)
    lab_test = np.argmax(Ctest, axis=1)
    acc = lab_test == prediction_test
    test_acc.append(np.mean(acc))
    loss=[]
    loss.append(loss_function(w,Xtrain,Ctrain))
    for i in range(len(batches)):
        b=batches[i]
        l=labels[i]
        # loss=gradient_w(w,Xtrain,Ctrain)
        w=SGD_one_step(b,l,w,lr)
        prediction=softmax(w,Xtrain)
        prediction=np.argmax(prediction,axis=1)
        lab=np.argmax(Ctrain,axis=1)
        acc=lab==prediction
        train_acc.append(np.mean(acc)*100)
        prediction_test = softmax(w, Xtest)
        prediction_test = np.argmax(prediction_test, axis=1)
        lab_test = np.argmax(Ctest, axis=1)
        acc = lab_test == prediction_test
        test_acc.append(np.mean(acc)*100)
        loss.append(loss_function(w, Xtrain, Ctrain))
    print(train_acc)
    print(test_acc)

    plt.figure()
    plt.plot(range(len(train_acc)), train_acc, label='train')
    plt.plot(range(len(test_acc)), test_acc, label='test')
    plt.legend()

    # plt.show()

    plt.figure()
    plt.plot(range(len(loss)), loss, label='loss')
    # plt.show()


def choose_param(file):
    loss_arr=[]
    lrs=[0.001,0.005,0.01,0.05,0.1,0.5]
    for lr in lrs:
        print("checking lr-",lr," in file-",file)
        train_acc,test_acc,loss= sgd_test(file,10000,lr)
        loss_arr.append(loss)
    # print(loss_arr)
    # print(np.argmin(loss_arr, axis=0))
    min_loss=np.amin(loss_arr,axis=1)


    bestLr=lrs[np.argmin(min_loss)]
    print("best learning rate=",bestLr)
    return bestLr
    # loss_arr=[]
    # ks=[10,20,30,40,50,60,70]
    # for k in ks:
    #     print("checking k=",k," for file=",file)
    #     train_acc, test_acc, loss = sgd_test(file, k, bestLr)
    #     loss_arr.append(loss)
    # sizes=[len(l) for l in loss_arr]
    # max_size=np.max(sizes)
    # loss_arr=[l+list(10*np.ones(max_size-len(l))) for l in loss_arr]
    # min_loss = np.amin(loss_arr, axis=1)
    # # print(arg_min_loss)
    # # print(loss_arr)
    # # loss_arr = [loss_arr[i][arg_min_loss[i]] for i in range(len(loss_arr))]
    # bestk = ks[np.argmin(min_loss)]
    # print(file)
    # print("best learning rate=",bestLr,"\nbest k=",bestk)
    # return bestk,bestLr

def SGD_test_plots():
    dir='HW1_Data\\'
    files=['GMMData.mat','PeaksData.mat','SwissRollData.mat']
    ks=[30,40,10]
    lrs=[0.5,0.05,0.1]
    i=0
    for f in files:
        # k,lr=ks[i],lrs[i] #choose_param(dir+f)#
        lr=choose_param(dir+f)
        k=ks[i]
        train_acc,test_acc,loss= sgd_test(dir+f,k,lr)
        plt.figure()
        plt.plot(range(len(train_acc)),train_acc,label='train')
        plt.plot(range(len(test_acc)), test_acc,label='test')
        plt.title('SGD test: {} Set, Acc of lr={} and number of batches={}'.format(f, lr,k))
        plt.legend()
        name='plots\SGD_test_'+str(lr)+'_'+str(k)+'_'+f+'.png'
        plt.savefig(name)
        # plt.show()
        plt.figure()
        plt.plot(range(len(loss)),loss,label='loss')
        plt.title('loss of train sample rate')
        name="plots\SGD_test_loss_"+" "+f+'.png'
        plt.savefig(name)
        i=i+1

        # plt.show()




# SGD_test_plots()
# SGD_test_least_squers()
# sgd_test('HW1_Data\GMMData.mat',32,0.05)
# gradient_test('HW1_Data\SwissRollData.mat')
# gradient_test2()
