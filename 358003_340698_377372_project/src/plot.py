from src.methods.deep_network import Trainer, MLP
from src.utils import accuracy_fn
import matplotlib.pyplot as plt

def train_test_acc_param_mlp(xtrain, ytrain, xtest, ytest, input_size, n_classes, hidden_layers, params,model,args):
    AccTrain = []
    AccTest = []
    N = 3
    for param in params:
        tempTrain = 0
        tempTest = 0
        for _ in range(N):
            if args.nn_type == "mlp":
                model = MLP(input_size, n_classes, hidden_layers,False)
            else:
                raise ValueError("Not implemented yet")
            method_obj = Trainer(model, lr=args.lr, epochs=param, batch_size=args.nn_batch_size)
            # Fit (:=train) the method on the training data
            trainPreds = method_obj.fit(xtrain, ytrain)
            testPreds = method_obj.predict(xtest)
            tempTrain += accuracy_fn(trainPreds, ytrain)
            tempTest += accuracy_fn(testPreds, ytest)
        AccTrain.append(tempTrain / N)
        AccTest.append(tempTest / N)
        

    AccTrainDrop = []
    AccTestDrop = []
    for param in params:
        tempTrain = 0
        tempTest = 0
        for _ in range(N):
            if args.nn_type == "mlp":
                model = MLP(input_size, n_classes, hidden_layers,True,0.2)
            else:
                raise ValueError("Not implemented yet")
            method_obj = Trainer(model, lr=args.lr, epochs=param, batch_size=args.nn_batch_size)
            # Fit (:=train) the method on the training data
            trainPreds = method_obj.fit(xtrain, ytrain)
            testPreds = method_obj.predict(xtest)
            tempTrain += accuracy_fn(trainPreds, ytrain)
            tempTest += accuracy_fn(testPreds, ytest)
        AccTrainDrop.append(tempTrain / N)
        AccTestDrop.append(tempTest / N)

    # plt.semilogx(params, AccTrain, 'r') # plotting t, a separately 
    # plt.semilogx(params, AccTest, 'b') # plotting t, b separately 
    # plt.title("Train and test accuracy vs different parameters")
    # plt.show()

    plt.plot(params, AccTrain, 'r', label = "Train data") # plotting t, a separately 
    plt.plot(params, AccTest, 'b',label = "Test data") # plotting t, b separately 
    plt.plot(params, AccTrainDrop, 'g', label = "Train data with dropout") # plotting t, a separately 
    plt.plot(params, AccTestDrop, 'y',label = "Test data with dropout") # plotting t, b separately 
    plt.title("Train and test accuracy vs different number of iterations with dropout")
    plt.legend()
    plt.show()
    return