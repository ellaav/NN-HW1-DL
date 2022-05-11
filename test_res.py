from matplotlib import pyplot as plt

from classifications_Tools import load_data_and_Suffle_train
from derivatives_residual import *


def jacMV_x_test():
    d = np.random.rand(3, 1)
    x = np.random.rand(3, 1)
    normalized_d = d / np.linalg.norm(d)

    eps_num = 20
    eps_vals = np.geomspace(0.5, 0.5 ** eps_num, eps_num)
    W1 =np.random.rand(3,3)
    W2 =np.random.rand(3,3)
    b = np.random.rand(3, 1)
    fx =x+ W2@np.tanh(W1 @x +b)

    no_grad, x_grad = [], []

    for eps in eps_vals:
        e_normalized_d = eps * normalized_d
        x_perturbatzia = np.add(x, e_normalized_d)
        fx_d = x_perturbatzia+ W2@np.tanh(W1 @x_perturbatzia +b)
        jackMV_x = jacMV_x(W1,W2, b , x , e_normalized_d)
        print(jackMV_x)
        print('epsilon: ', eps)
        print(np.linalg.norm(np.subtract(fx_d, fx)))
        print(np.linalg.norm(np.subtract(np.subtract(fx_d, fx), jackMV_x)))
        no_grad.append(np.linalg.norm(np.subtract(fx_d, fx)))
        x_grad.append(np.linalg.norm(np.subtract(np.subtract(fx_d, fx), jackMV_x)))

    l = range(eps_num)
    plt.plot(l, no_grad, label='First Order')
    plt.plot(l, x_grad, label='Second Order')
    plt.yscale('log')
    plt.legend()
    plt.title('jacMV_x_res test ')
    plt.show()







def jacMV_b_test():
    d = np.random.rand(3, 1)
    x = np.random.rand(3, 1)
    normalized_d = d / np.linalg.norm(d)

    eps_num = 20
    eps_vals = np.geomspace(0.5, 0.5 ** eps_num, eps_num)

    W1 =np.random.randn(3,3)
    W2 =np.random.randn(3,3)

    b = np.random.rand(3, 1)
    fx =x+ W2@np.tanh(W1 @x +b)

    no_grad, b_grad = [], []
    B = b

    for eps in eps_vals:
        eps_d = eps * normalized_d
        b = np.add(B, eps_d)

        fx_d = x+ W2@np.tanh(W1 @x +b)

        jackMV_b = jacMV_b(W1,W2,b,x, eps_d)

        no_grad.append(np.linalg.norm(np.subtract(fx_d, fx)))
        b_grad.append(np.linalg.norm(np.subtract(np.subtract(fx_d, fx), jackMV_b)))

    l = range(eps_num)
    plt.plot(l, no_grad, label='First Order')
    plt.plot(l, b_grad, label='Second Order')
    plt.yscale('log')
    plt.legend()
    plt.title('jacMV_b_res test')

    plt.show()



def jacMV_w1_test():
    d = np.random.rand(3, 3)
    x = np.random.rand(3, 1)
    # normalized_d = d / np.linalg.norm(d)

    eps_num = 20
    eps_vals = np.geomspace(0.5, 0.5 ** eps_num, eps_num)

    W1 =np.random.randn(3,3)
    W2 = np.random.randn(3, 3)
    b = np.random.rand(3, 1)
    fx =x+ W2@np.tanh(W1 @x +b)


    no_grad, w_grad = [], []
    w = W1

    for eps in eps_vals:

        eps_d = eps * d

        W1 = np.add(w, eps_d)
        fx_d = x+ W2@np.tanh(W1 @x +b)

        # eps_d = eps_d.reshape(-1, 1)
        jacMV_W1 = jacMV_w1(W1,W2,b,x, eps_d)

        first_order = fx_d - fx
        second_order = first_order - jacMV_W1

        no_grad.append(np.linalg.norm(first_order))
        w_grad.append(np.linalg.norm(second_order))

    l = range(eps_num)
    plt.plot(l, no_grad, label='First Order')
    plt.plot(l, w_grad, label='Second Order')
    plt.yscale('log')
    plt.legend()
    plt.title('jacMV_W1_res test')

    plt.show()

def jacMV_w2_test():
    d = np.random.rand(3, 3)
    x = np.random.rand(3, 1)
    # normalized_d = d / np.linalg.norm(d)

    eps_num = 20
    eps_vals = np.geomspace(0.5, 0.5 ** eps_num, eps_num)

    W1 =np.random.randn(3,3)
    W2 = np.random.randn(3, 3)
    b = np.random.rand(3, 1)
    fx =x+ W2@np.tanh(W1 @x +b)


    no_grad, w_grad = [], []
    w = W1
    w2=W2
    for eps in eps_vals:

        eps_d = eps * d

        W1 = np.add(w, eps_d)
        W2=np.add(w2, eps_d)
        fx_d = x+ W2@np.tanh(W1 @x +b)

        # eps_d = eps_d.reshape(-1, 1)
        jacMV_W2 = jacMV_w2(W1,W2,b,x, eps_d)

        first_order = fx_d - fx
        second_order = first_order - jacMV_W2

        no_grad.append(np.linalg.norm(first_order))
        w_grad.append(np.linalg.norm(second_order))

    l = range(eps_num)
    plt.plot(l, no_grad, label='First Order')
    plt.plot(l, w_grad, label='Second Order')
    plt.yscale('log')
    plt.legend()
    plt.title('jacMV_W2_res test')

    plt.show()





def new_grad_test(file):
    print(file)
    # Load data
    Xtrain,Ctrain,Xtest,Ctest=load_data_and_Suffle_train(file)
    layer_dims = [Xtrain.shape[0], Ctrain.shape[-1]]


def jacTMV_x_test():
    v = np.random.rand(3, 1)
    x = np.random.rand(3, 1)
    u = np.random.rand(3, 1)
    W1=np.random.rand(3, 3)
    W2 = np.random.rand(3, 3)
    b=np.random.rand(3, 1)

    jacMV_X = jacMV_x(W1,W2,b,x, v)


    g_x=jacTMV_x(W1,W2,b,x,u)
    jacTMV_X = g_x


    u_jac = u.T @ jacMV_X
    v_jacT = v.T @ jacTMV_X

    print(abs(np.subtract(u_jac, v_jacT)))

def jacTMV_w1_test():
    v = np.random.rand(3, 3)
    x = np.random.rand(3, 1)
    u = np.random.rand(3, 1)
    W1=np.random.rand(3, 3)
    W2 = np.random.rand(3, 3)
    b=np.random.rand(3, 1)

    jacMV_W = jacMV_w1(W1,W2,b,x, v)

    g_w=jacTMV_w1(W1,W2,b,x,u)
    jacTMV_W = g_w

    u_jac = u.T @ jacMV_W
    v_jacT = v.ravel().T @ jacTMV_W.ravel()

    print(abs(np.subtract(u_jac, v_jacT)))


def jacTMV_w2_test():
    v = np.random.rand(3, 3)
    x = np.random.rand(3, 1)
    u = np.random.rand(3, 1)
    W1=np.random.rand(3, 3)
    W2 = np.random.rand(3, 3)
    b=np.random.rand(3, 1)

    jacMV_W = jacMV_w2(W1,W2,b,x, v)

    g_w=jacTMV_w2(W1,W2,b,x,u)
    jacTMV_W = g_w

    u_jac = u.T @ jacMV_W
    v_jacT = v.ravel().T @ jacTMV_W.ravel()

    print(abs(np.subtract(u_jac, v_jacT)))


def jacTMV_b_test():
    v = np.random.rand(3, 1)
    x = np.random.rand(3, 1)
    u = np.random.rand(3, 1)
    W1=np.random.rand(3, 3)
    W2=np.random.rand(3, 3)
    b=np.random.rand(3, 1)

    jackMV_B = jacMV_b(W1,W2,b,x, v)


    g_b=jacTMV_b(W1,W2,b,x,u)
    jackTMV_B = g_b


    u_jack = u.T @ jackMV_B
    v_jackT = v.T @ jackTMV_B

    print(abs(np.subtract(u_jack, v_jackT)))


# jacTMV_x_test()
# jacTMV_w1_test()
# jacTMV_w2_test()
# jacTMV_b_test()

# jacMV_x_test()
# jacMV_b_test()
# jacMV_w1_test()
# jacMV_w2_test()
# test_acc_res_net()
