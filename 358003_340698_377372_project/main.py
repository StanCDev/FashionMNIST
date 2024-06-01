import argparse

import numpy as np
import torch
from torchinfo import summary

from src.data import load_data
from src.methods.pca import PCA
from src.methods.deep_network import MLP, CNN, Trainer, MyViT
from src.utils import normalize_fn, append_bias_term, accuracy_fn, macrof1_fn, get_n_classes
from src.plot import train_test_acc_param_mlp
np.random.seed(100)


def main(args):
    """
    The main function of the script. Do not hesitate to play with it
    and add your own code, visualization, prints, etc!

    Arguments:
        args (Namespace): arguments that were parsed from the command line (see at the end 
                          of this file). Their value can be accessed as "args.argument".
    """
    ## 1. First, we load our data and flatten the images into vectors
    xtrain, xtest, ytrain = load_data(args.data)
    xtrain = xtrain.reshape(xtrain.shape[0], -1)
    xtest = xtest.reshape(xtest.shape[0], -1)

    ## 2. Then we must prepare it. This is were you can create a validation set,
    #  normalize, add bias, etc.
    xtrain = normalize_fn(xtrain, np.mean(xtrain), np.std(xtrain))
    xtest = normalize_fn(xtest,np.mean(xtest), np.std(xtest))

    # Make a validation set
    if not args.test:
        N = xtrain.shape[0]
        all_ind = np.arange(N)
        split_ratio = 0.2
        split_size = int(split_ratio * N)

        ################### RANDOM SHUFFLING ################
        all_ind = np.random.permutation(all_ind)
        #####################################################

        ########### TRAINING AND VALIDATION INDICES #########
        val_ind = all_ind[: split_size]
        train_ind = np.setdiff1d(all_ind, val_ind, assume_unique=True)
        #####################################################

        xtrain_original = xtrain
        ytrain_original = ytrain

        xtrain = xtrain_original[train_ind]
        xtest = xtrain_original[val_ind]

        ytest = ytrain_original[val_ind]
        ytrain = ytrain_original[train_ind]

    ### Here we transform matrices back to vectors in the cases of CNN and transformer
    if args.nn_type == "cnn" or args.nn_type == "transformer":
        ##print(f"Shape of xtrain before {xtrain.shape}")
        N , D = xtrain.shape
        sqrtD = int(np.sqrt(D))
        if sqrtD * sqrtD != D:
            raise ValueError("Images are not square")
        xtrain = xtrain.reshape(N,1,sqrtD,sqrtD)
        ##print(f"Shape of xtrain after {xtrain.shape}")

        N2 , D2 = xtest.shape
        sqrtD2 = int(np.sqrt(D2))
        if sqrtD2 * sqrtD2 != D2:
            raise ValueError("Images are not square")
        xtest = xtest.reshape(N2,1,sqrtD2,sqrtD2)


    # Dimensionality reduction (MS2)
    if args.use_pca:
        print("Using PCA")
        pca_obj = PCA(d=args.pca_d)
        pca_obj.find_principal_components(xtrain)

        xtrain = pca_obj.reduce_dimension(xtrain)
        xtest = pca_obj.reduce_dimension(xtest)


    ## 3. Initialize the method you want to use.
    model = None

    # Neural Networks (MS2)
    # Prepare the model (and data) for Pytorch
    # Note: you might need to reshape the data depending on the network you use!
    hidden_layers : list[int] = [256,128,64]
    n_classes = get_n_classes(ytrain)
    if args.nn_type == "mlp":
        print(f"Shape of xtrain {xtrain.shape}")
        input_size = xtrain.shape[1]
        model = MLP(input_size, n_classes, hidden_layers,True) ### WRITE YOUR CODE HERE
    elif args.nn_type == "cnn":
        N , Ch, D, D = xtrain.shape
        model = CNN(Ch, n_classes, D) ### change 1
    elif args.nn_type == "transformer":
        N , Ch, D, D = xtrain.shape
        ## change nbr blocks to 8
        model = MyViT((Ch,D,D),n_patches=7,n_blocks=1,hidden_d=64,n_heads=8,out_d=n_classes)
    else:
        raise ValueError("Inputted model is not a good model")

    summary(model)


    ######################## SELECTING DEVICE ########################
    device = torch.device("cpu")
    if args.device == "cuda":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            print("cuda not available on this device")
    elif args.device == "mps":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            print("mps not available on this device")

    model.to(device)

    ######################## COMMENTING STARTS HERE ########################
    # # Trainer object
    # method_obj = Trainer(model, lr=args.lr, epochs=args.max_iters, batch_size=args.nn_batch_size)


    # ## 4. Train and evaluate the method

    # # Fit (:=train) the method on the training data
    # preds_train = method_obj.fit(xtrain, ytrain)

    # # Predict on unseen data
    # preds = method_obj.predict(xtest)

    # ## Report results: performance on train and valid/test sets
    # acc = accuracy_fn(preds_train, ytrain)
    # macrof1 = macrof1_fn(preds_train, ytrain)
    # print(f"\nTrain set: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")


    # ## As there are no test dataset labels, check your model accuracy on validation dataset.
    # # You can check your model performance on test set by submitting your test set predictions on the AIcrowd competition.
    # #print(f"Shape of preds = {preds.shape} vs shape of xtest = {xtest.shape}")
    # if not args.test:
    #     acc = accuracy_fn(preds, ytest)
    #     macrof1 = macrof1_fn(preds, ytest)
    #     print(f"Validation set:  accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")

    ######################## COMMENTING ENDS HERE ########################
    ### WRITE YOUR CODE HERE if you want to add other outputs, visualization, etc.
    ########### Experimental stuff ###########
    if True:
        params = []
        if args.nn_type == "mlp":
            params = [5 * i for i in range(1,11)]
        elif args.nn_type == "transformer":
            params = list(range(1,10+1))
        input_size = xtrain.shape[1]
        train_test_acc_param_mlp(xtrain,ytrain,xtest,ytest, input_size, n_classes , hidden_layers, params , model, args)



if __name__ == '__main__':
    # Definition of the arguments that can be given through the command line (terminal).
    # If an argument is not given, it will take its default value as defined below.
    parser = argparse.ArgumentParser()
    # Feel free to add more arguments here if you need!

    # MS2 arguments
    parser.add_argument('--data', default="dataset", type=str, help="path to your dataset")
    parser.add_argument('--nn_type', default="mlp",
                        help="which network architecture to use, it can be 'mlp' | 'transformer' | 'cnn'")
    parser.add_argument('--nn_batch_size', type=int, default=64, help="batch size for NN training")
    parser.add_argument('--device', type=str, default="cpu",
                        help="Device to use for the training, it can be 'cpu' | 'cuda' | 'mps'")
    parser.add_argument('--use_pca', action="store_true", help="use PCA for feature reduction")
    parser.add_argument('--pca_d', type=int, default=100, help="the number of principal components")


    parser.add_argument('--lr', type=float, default=1e-5, help="learning rate for methods with learning rate")
    parser.add_argument('--max_iters', type=int, default=100, help="max iters for methods which are iterative")
    parser.add_argument('--test', action="store_true",
                        help="train on whole training data and evaluate on the test data, otherwise use a validation set")


    # "args" will keep in memory the arguments and their values,
    # which can be accessed as "args.data", for example.
    args = parser.parse_args()
    main(args)