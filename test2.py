import sys

from NN_new import *
from derivatives import *
from tqdm import tqdm

def network_grad_test(file):


    Xtrain, Ytrain, test_sets, train_sets = load_data_and_Suffle_train(file)

    # hyper params
    n_dim=100
    layer_dims = [Xtrain.shape[0],n_dim,n_dim, Ytrain.shape[-1]]

    # init model
    model =NN(layer_dims)

    batch = Xtrain
    labels = Ytrain

    outputs = model.L_model_forward(batch)
    f_x=loss_function(model.W_loss,outputs,labels)

    # get the d vectors to perturb the weights
    d_vecs = get_per(model)

    model.L_model_backward(labels)  # compute grad(x) per layer

    # concatenate grad per layer
    grad_x = get_raveled_grads_per_layer(model)

    # save weights of each layer, for testing:
    weights_list = get_weights_from_layers(model)

    # save the initial weights
    w_ce = model.W_loss
    w_li = model.linear_layer.W

    first_order_l, second_order_l = [], []
    eps_vals = np.geomspace(0.5, 0.5 ** 20, 20)

    for eps in eps_vals:

        eps_d = eps * d_vecs[0]
        eps_ds = [eps_d.ravel()]
        model.linear_layer.W = np.add(w_li, eps_d)
        for d, ll, w in zip(d_vecs[1:-1], model.deep_layes, weights_list[1:-1]):
            eps_d = eps * d
            ll.W = np.add(w, eps_d)
            eps_ds.append(eps_d.ravel())
        eps_d = eps * d_vecs[-1]
        model.W_loss = np.add(w_ce, eps_d)
        eps_ds.append(eps_d.ravel())
        eps_ds = np.concatenate(eps_ds, axis=0)

        output_d = model.L_model_forward(batch)
        fx_d = loss_function(model.W_loss,output_d, labels)

        first_order = abs(fx_d - f_x)
        r=eps_ds.ravel().T
        rr=grad_x.ravel()
        second_order = abs(fx_d - f_x - eps_ds.ravel().T @ grad_x.ravel())

        print(first_order)
        print(second_order)

        first_order_l.append(first_order)
        second_order_l.append(second_order)

    l = range(20)
    plt.title('Network gradient test')
    plt.plot(l, first_order_l, label='First Order')
    plt.plot(l, second_order_l, label='Second Order')
    plt.yscale('log')
    plt.legend()
    plt.show()

def get_raveled_grads_per_layer(model):
    grad_x = [model.linear_layer.g_w.ravel()]
    for ll in model.deep_layes:
        grad_x.append(ll.g_w.ravel())
    grad_x.append(model.grad_w.ravel())
    grad_x = np.concatenate(grad_x, axis=0)
    return grad_x


def get_per(model):
    d_vecs = [np.random.rand(model.linear_layer.W.shape[0], model.linear_layer.W.shape[1])]
    for ll in model.deep_layes:
        d_vecs.append(np.random.rand(ll.W.shape[0], ll.W.shape[1]))
    d_vecs.append(np.random.rand(model.W_loss.shape[0], model.W_loss.shape[1]))
    return [d / np.linalg.norm(d) for d in d_vecs]

def get_weights_from_layers(model):
    weights_list = [model.linear_layer.W]
    for ll in model.deep_layes:
        weights_list.append(ll.W)
    weights_list.append(model.W_loss)
    return weights_list


def test(file):
    Xtrain, Ytrain, test_sets, train_sets = load_data_and_Suffle_train(file)
    iter_num=50
    k=45
    lr=0.15
    n_dim=70
    layer_dims = [Xtrain.shape[0],n_dim,n_dim,n_dim,Ytrain.shape[-1]]
    model=NN(layer_dims)


    all_batches, all_labels = creat_batches(Xtrain,Ytrain,k)

    accs_hyper_params_train = []
    accs_hyper_params_test = []

    for e in range(1, iter_num):
        acc_train = []
        loss_l = []
        for i in tqdm(range(len(all_batches)), total=len(all_batches),
                                  file=sys.stdout):
            labels=all_labels[i]
            batch=all_batches[i]

            outputs = model.L_model_forward(batch)

            loss = loss_function(model.W_loss,outputs, labels)
            loss_l.append(loss)

            model.L_model_backward(labels)
            model.update_parameters(lr)

            outputs = softmax(model.W_loss,outputs)

            labels = np.asarray([np.where(l == 1)[0][0] for l in labels])
            pred=np.argmax(outputs, axis=1).reshape(-1, 1)
            prediction = np.asarray([p[0] for p in pred])

            acc_train = np.append(acc_train, prediction == labels, axis=0)

        print('Epoch {} train acc: {}  train loss: {}'.format(e, np.mean(acc_train), np.mean(loss_l)))
        if e%5==0:
            lr=lr+0.003

        if e==45:
            lr=0.01


        accs_hyper_params_train.append(np.mean(acc_train) * 100)
        accs_hyper_params_test.append(np.mean(test_accuracy(model,(all_batches, all_labels) )) * 100)

    plt.plot(range(1, iter_num), accs_hyper_params_train, label='Train Accuracy')
    plt.plot(range(1, iter_num), accs_hyper_params_test, label='Validation Accuracy')
    plt.title('{} Data set, lr={:.5f} and batch size={}'.format(file, lr, k))
    plt.legend()
    plt.show()


def test_accuracy(model, test_sets):

    acc_test = []
    loss_test = []
    all_batches, all_labels = test_sets

    for i in tqdm(range(len(all_batches)), total=len(all_batches),
                  file=sys.stdout):
        labels = all_labels[i]
        batch = all_batches[i]

        outputs = model.L_model_forward(batch)

        loss = loss_function(model.W_loss,outputs,labels)
        loss_test.append(loss)

        outputs = softmax(model.W_loss,outputs)
        labels = np.asarray([np.where(l == 1)[0][0] for l in labels])
        pred = np.argmax(outputs, axis=1).reshape(-1, 1)
        prediction = np.asarray([p[0] for p in pred])

        acc_test = np.append(acc_test, prediction == labels, axis=0)

    print('Test acc: {}'.format(np.mean(acc_test)))
    return acc_test


# network_grad_test('HW1_Data\SwissRollData.mat')
test('HW1_Data\PeaksData.mat')
test('HW1_Data\SwissRollData.mat')
test('HW1_Data\GMMData.mat')