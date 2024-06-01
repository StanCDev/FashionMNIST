from src.methods.deep_network import Trainer, MLP, MyViT
from src.utils import accuracy_fn, macrof1_fn
import matplotlib.pyplot as plt

def train_test_acc_param_mlp(xtrain, ytrain, xtest, ytest, input_size, n_classes, hidden_layers, params,model,args):
    print(f"params={params}")
    AccTrain = []
    AccTest = []
    N = 1
    for param in params:
        print(f"Param = {param}")
        tempTrain = 0
        tempTest = 0
        #for _ in range(N):
        if args.nn_type == "mlp":
            model = MLP(input_size, n_classes, hidden_layers,False)
        elif args.nn_type == "transformer":
            N , Ch, D, D = xtrain.shape
            model = MyViT((Ch,D,D),n_patches=7,n_blocks=param,hidden_d=64,n_heads=8,out_d=n_classes)
        else:
            raise ValueError("Not implemented yet")
        method_obj = Trainer(model, lr=args.lr, epochs=args.max_iters, batch_size=args.nn_batch_size)
        ##############################################################################################################################


        ## 4. Train and evaluate the method

        # Fit (:=train) the method on the training data
        preds_train = method_obj.fit(xtrain, ytrain)

        # Predict on unseen data
        preds = method_obj.predict(xtest)

        ## Report results: performance on train and valid/test sets
        acc = accuracy_fn(preds_train, ytrain)
        macrof1 = macrof1_fn(preds_train, ytrain)
        print(f"\nTrain set: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")


        ## As there are no test dataset labels, check your model accuracy on validation dataset.
        # You can check your model performance on test set by submitting your test set predictions on the AIcrowd competition.
        #print(f"Shape of preds = {preds.shape} vs shape of xtest = {xtest.shape}")
        if not args.test:
            acc = accuracy_fn(preds, ytest)
            macrof1 = macrof1_fn(preds, ytest)
            print(f"Validation set:  accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")
        ##############################################################################################################################
        # Fit (:=train) the method on the training data
        tempTrain += accuracy_fn(preds_train, ytrain)
        tempTest += accuracy_fn(preds, ytest)
        AccTrain.append(tempTrain / N)
        AccTest.append(tempTest / N)

    AccTrainDrop = []
    AccTestDrop = []
    if args.nn_type == "mlp":
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

    print(f"acctrain={AccTrain}")
    print(f"acctest={AccTest}")
    if args.nn_type == "mlp":
        plt.plot(params, AccTrain, 'r', label = "Train data") # plotting t, a separately 
        plt.plot(params, AccTest, 'b',label = "Test data") # plotting t, b separately 
        plt.plot(params, AccTrainDrop, 'g', label = "Train data with dropout") # plotting t, a separately 
        plt.plot(params, AccTestDrop, 'y',label = "Test data with dropout") # plotting t, b separately 
        plt.xlabel("Number of Epochs")
        plt.ylabel("Accuracy %")
        plt.title("Train and test accuracy vs different number of iterations with dropout")
    if args.use_pca and args.nn_type == "mlp":
        ...
    if args.nn_type == "transformer":
        plt.plot(params, AccTrain, 'r', label = "Train data") # plotting t, a separately 
        plt.plot(params, AccTest, 'b',label = "Test data") # plotting t, b separately 
        plt.xlabel("Number of attention blocks")
        plt.ylabel("Accuracy %")
        plt.title("Train and test accuracy vs different number of blocks")
    plt.legend()
    plt.show()
    return