
from classifications_Tools import *
from derivatives import *



def jacMV_x_test():
    d = np.random.rand(3, 1)
    x = np.random.rand(3, 1)
    normalized_d = d / np.linalg.norm(d)

    eps_num = 20
    eps_vals = np.geomspace(0.5, 0.5 ** eps_num, eps_num)
    W =np.random.rand(3,3)
    b = np.random.rand(3, 1)
    fx =np.tanh(W @x+b)

    no_grad, x_grad = [], []

    for eps in eps_vals:
        e_normalized_d = eps * normalized_d
        x_perturbatzia = np.add(x, e_normalized_d)
        fx_d = np.tanh(W@x_perturbatzia +b)
        jackMV_x = jacMV_x(W, b , x , e_normalized_d)
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
    plt.title('jacMV test ')
    plt.show()







def jacMV_b_test():
    d = np.random.rand(3, 1)
    x = np.random.rand(3, 1)
    normalized_d = d / np.linalg.norm(d)

    eps_num = 20
    eps_vals = np.geomspace(0.5, 0.5 ** eps_num, eps_num)

    W =np.random.randn(3,3)
    b = np.random.rand(3, 1)
    fx =np.tanh(W @x +b)

    no_grad, b_grad = [], []
    B = b

    for eps in eps_vals:
        eps_d = eps * normalized_d
        b = np.add(B, eps_d)

        fx_d = np.tanh(W @x +b)

        jackMV_b = jacMV_b(W,b,x, eps_d)

        no_grad.append(np.linalg.norm(np.subtract(fx_d, fx)))
        b_grad.append(np.linalg.norm(np.subtract(np.subtract(fx_d, fx), jackMV_b)))

    l = range(eps_num)
    plt.plot(l, no_grad, label='First Order')
    plt.plot(l, b_grad, label='Second Order')
    plt.yscale('log')
    plt.legend()
    plt.title('jacMV test')

    plt.show()



def jacMV_w_test():
    d = np.random.rand(4, 3)
    x = np.random.rand(3, 1)

    eps_num = 20
    eps_vals = np.geomspace(0.5, 0.5 ** eps_num, eps_num)

    W =np.random.randn(4,3)
    b = np.random.rand(4, 1)
    fx =np.tanh(W @x +b)


    no_grad, w_grad = [], []
    w = W

    for eps in eps_vals:

        eps_d = eps * d

        W = np.add(w, eps_d)
        fx_d = np.tanh(W @x+b)

        # eps_d = eps_d.reshape(-1, 1)
        jacMV_W = jacMV_w(W,b,x, eps_d)

        first_order = fx_d - fx
        second_order = first_order - jacMV_W

        no_grad.append(np.linalg.norm(first_order))
        w_grad.append(np.linalg.norm(second_order))

    l = range(eps_num)
    plt.plot(l, no_grad, label='First Order')
    plt.plot(l, w_grad, label='Second Order')
    plt.yscale('log')
    plt.legend()
    plt.title('jacMV test')

    plt.show()



def jacTMV_w_test():
    v = np.random.rand(4, 3)
    x = np.random.rand(3, 1)
    u = np.random.rand(4, 1)
    W =np.random.randn(4,3)
    b = np.random.rand(4, 1)
    jacMV_W =jacMV_w(W,b,x, v)


    g_x = jacTMV_x(W,b,x,  u)
    g_w = jacTMV_w(W,b,x, u)
    g_b = jacTMV_b(W,b,x,  u)
    jacTMV_W = g_w

    u_jac = u.T @ jacMV_W
    v_jacT = v.ravel().T @ jacTMV_W.ravel()

    print(abs(np.subtract(u_jac, v_jacT)))



# jacMV_x_test()
# jacMV_b_test()
# jacMV_w_test()
# jacTMV_w_test()
# SGD_test_plots()
