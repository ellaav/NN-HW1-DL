
from NN_new import *


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.





# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    dir='HW1_Data\\'
    files=  ['GMMData.mat','PeaksData.mat','SwissRollData.mat']
    # Load data
    for file in files:
        print(file)

        Xtrain, Ctrain, Xtest, Ctest = load_data_and_Suffle_train(dir+file)

        ks = [ 64, 85]
        lrs = [ 0.0075, 0.0125,0.125]
        plt.figure()
        bestplots={}
        iter=20
        for neurons in [200]:

            for k in ks:
                for lr in lrs:
                    for j in range(iter):

                        print_cost = True
                        train_acc = []
                        test_acc = []
                        loss = []
                        # hyper params
                        layer_dims = [Xtrain.shape[0], neurons, neurons,neurons, Ctrain.shape[-1]]
                        model=NN(layer_dims)
                        batches, lables = creat_batches(Xtrain, Ctrain, k)
                        for i in range(len(batches)):
                            X = batches[i]
                            Y = lables[i]
                            AL=model.L_model_forward(X)
                            if i%10==0:
                                pred = softmax(model.W_loss, AL)

                                prediction = np.argmax(pred,axis=1)
                                labels=np.argmax(Y,axis=1)
                                acc = labels == prediction
                                train_acc.append(np.mean(acc) * 100)

                            model.L_model_backward(Y)


                            # Update parameters.
                            parameters = model.update_parameters(lr)

                        batches, lables = creat_batches(Xtest, Ctest, k)
                        for i in range(len(batches)):
                            X = batches[i]
                            Y = lables[i]
                            AL=model.L_model_forward(X)
                            pred = softmax(model.W_loss, AL)

                            prediction = np.argmax(pred,axis=1)
                            labels=np.argmax(Y,axis=1)
                            acc = labels == prediction
                            test_acc.append(np.mean(acc) * 100)

                        if len(bestplots)<=4:
                            bestplots[(k,lr,neurons)]=(train_acc,test_acc)
                        else:
                            for key,(_,test) in zip(bestplots.keys(),bestplots.values()):
                                if np.mean(test_acc)>np.mean(test):
                                    bestplots[(k,lr,neurons)]=(train_acc,test_acc)
                                    bestplots.pop(key)
                                    break
    for (k,lr,neurons),(train_acc,test_acc) in zip(bestplots.keys(),bestplots.values()):
        print("k=",k," lr=",lr," n= ",neurons)
        print(train_acc)
        print(np.mean(test_acc))

        l1='train k='+str(k)+"lr="+str(lr)+"n="+str(neurons)
        plt.plot(range(len(train_acc)), train_acc, label=l1)
        # plt.plot(range(len(test_acc)), test_acc, label=l2)
        plt.legend()

    name = 'plots\Acctest'+ file + '.png'
    plt.savefig(name)


# # See PyCharm help at https://www.jetbrains.com/help/pycharm/
