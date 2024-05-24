import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

## MS2
import utils

class MLP(nn.Module):
    """
    An MLP network which does classification.

    It should not use any convolutional layers.
    """

    def __init__(self, input_size, n_classes, layers : list[int] = [32]):
        """
        Initialize the network.
        
        You can add arguments if you want, but WITH a default value, e.g.:
            __init__(self, input_size, n_classes, my_arg=32)
        
        Arguments:
            input_size (int): size of the input
            n_classes (int): number of classes to predict
            layers (list): list that specifies how many hidden layers and how many neurons per hidden layer
        """
        #super().__init__()
        assert len(layers) != 0 , "MLP has no hidden layers!"
        super(MLP, self).__init__()

        self.fc = []
        layers = [input_size] + layers + [n_classes]
        for i in range(len(layers) - 1):
            self.fc.append(nn.Linear(layers[i], layers[i+1]))

    

    def forward(self, x):
        """
        Predict the class of a batch of samples with the model.

        Arguments:
            x (tensor): input batch of shape (N, D)
        Returns:
            preds (tensor): logits of predictions of shape (N, C)
                Reminder: logits are value pre-softmax.
        """
        preds = x
        for fc in self.fc:
            preds = F.relu(fc(preds)) #not sure about the shape, what about softmax??
        return preds


class CNN(nn.Module):
    """
    A CNN which does classification.

    It should use at least one convolutional layer.
    """

    def __init__(self, input_channels, n_classes):
        """
        Initialize the network.
        
        You can add arguments if you want, but WITH a default value, e.g.:
            __init__(self, input_channels, n_classes, my_arg=32)
        
        Arguments:
            input_channels (int): number of channels in the input
            n_classes (int): number of classes to predict
        """
        super().__init__()
        ##
        ###
        #### WRITE YOUR CODE HERE!
        ###
        ##

    def forward(self, x):
        """
        Predict the class of a batch of samples with the model.

        Arguments:
            x (tensor): input batch of shape (N, Ch, H, W)
        Returns:
            preds (tensor): logits of predictions of shape (N, C)
                Reminder: logits are value pre-softmax.
        """
        ##
        ###
        #### WRITE YOUR CODE HERE!
        ###
        ##
        return preds



class MyMSA(nn.Module):
    def __init__(self, d, n_heads=2):
        super(MyMSA, self).__init__()
        self.d = d
        self.n_heads = n_heads

        assert d % n_heads == 0, f"Can't divide dimension {d} into {n_heads} heads"

        d_head = int(d / n_heads)
        self.q_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.k_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.v_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.d_head = d_head
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, sequences):
        # Sequences has shape (N, seq_length, token_dim)
        # We go into shape    (N, seq_length, n_heads, token_dim / n_heads)
        # And come back to    (N, seq_length, item_dim)  (through concatenation)
        result = []
        for sequence in sequences:
            seq_result = []
            for head in range(self.n_heads):
                q_mapping = self.q_mappings[head]
                k_mapping = self.k_mappings[head]
                v_mapping = self.v_mappings[head]

                seq = sequence[:, head * self.d_head: (head + 1) * self.d_head]
                q, k, v = q_mapping(seq), k_mapping(seq), v_mapping(seq)

                attention = self.softmax(q @ k.T / (self.d_head ** 0.5))
                seq_result.append(attention @ v)
            result.append(torch.hstack(seq_result))
        return torch.cat([torch.unsqueeze(r, dim=0) for r in result])




class MyViT(nn.Module):
    """
    A Transformer-based neural network
    """

    def __init__(self, chw, n_patches, n_blocks, hidden_d, n_heads, out_d):
        """
        Initialize the network.
        
        """
        super().__init__()

        self.chw = chw 
        self.n_patches = n_patches
        self.hidden_d = hidden_d
        

        assert chw[1] % n_patches == 0, "Number of patches do not divide the image evenly"
        assert chw[2] % n_patches == 0, "Number of patches do not divide the image evenly"

        self.patch_size = (chw[1] / n_patches, chw[2] / n_patches)
        
        self.input_d = int(chw[0] * self.patch_size[0] * self.patch_size[1])
        self.linear_mapper = nn.Linear(self.input_d, self.hidden_d)

        self.class_token = nn.Parameter(torch.rand(1, self.hidden_d))

        self.pos_embed = nn.Parameter(torch.tensor(self.get_positional_embeddings(self.n_patches ** 2 + 1, self.hidden_d)))
        self.pos_embed.requires_grad = False


        ##
        ###
        #### WRITE YOUR CODE HERE!
        ###
        ##
        
    def patchify(images, n_patches):
        n, c, h, w = images.shape

        assert h == w, "Assert image is square"

        patches = torch.zeros(n, n_patches ** 2, h * w * c // n_patches ** 2)
        patch_size = h // n_patches

        for idx, image in enumerate(images):
            for i in range(n_patches):
                for j in range(n_patches):
                    patch = image[:, i * patch_size: (i + 1) * patch_size, j * patch_size: (j + 1) * patch_size]
                    patches[idx, i * n_patches + j] = patch.flatten()
        return patches

    def get_positional_embeddings(sequence_length, d):
        result = torch.ones(sequence_length, d)
        for i in range(sequence_length):
            for j in range(d):
                result[i][j] = np.sin(i / (10000 ** (j / d))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / d)))
        return result
    
    def forward(self, x):
        """
        Predict the class of a batch of samples with the model.

        Arguments:
            x (tensor): input batch of shape (N, Ch, H, W), N is the number of images, Ch the number of channels and H and W height and width
        Returns:
            preds (tensor): logits of predictions of shape (N, C)
                Reminder: logits are value pre-softmax.
        """
        patches = self.patchify(x, self.n_patches)
        tokens = self.linear_mapper(patches)
        tokens = torch.stack([torch.vstack((self.class_token, tokens[i])) for i in range(len(tokens))])

        pos_embed = self.pos_embed.repeat(x.shape[0], 1, 1) #IDK IM NOT SURE ABOUT THIS, instead of x0 say n
        out = tokens + pos_embed
        nn.M
        return out
        
        


class Trainer(object):
    """
    Trainer class for the deep networks.

    It will also serve as an interface between numpy and pytorch.
    """

    def __init__(self, model, lr, epochs, batch_size):
        """
        Initialize the trainer object for a given model.

        Arguments:
            model (nn.Module): the model to train
            lr (float): learning rate for the optimizer
            epochs (int): number of epochs of training
            batch_size (int): number of data points in each batch
        """
        self.lr = lr
        self.epochs = epochs
        self.model = model
        self.batch_size = batch_size

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(model.parameters(), lr)  ### WRITE YOUR CODE HERE

    def train_all(self, dataloader):
        """
        Fully train the model over the epochs. 
        
        In each epoch, it calls the functions "train_one_epoch". If you want to
        add something else at each epoch, you can do it here.

        Arguments:
            dataloader (DataLoader): dataloader for training data
        """
        epochs = range(self.epochs)
        for ep in epochs:
            self.train_one_epoch(dataloader,ep, len(epochs))
            ### WRITE YOUR CODE HERE if you want to do add something else at each epoch

    def train_one_epoch(self, dataloader, ep, epochs):
        """
        Train the model for ONE epoch.

        Should loop over the batches in the dataloader. (Recall the exercise session!)
        Don't forget to set your model to training mode, i.e., self.model.train()!

        Arguments:
            dataloader (DataLoader): dataloader for training data
        """
        for it, batch in enumerate(dataloader):
            # 5.1 Load a batch, break it down in images and targets.
            x, y = batch

            # 5.2 Run forward pass.
            logits = self.model.forward(x)  ### WRITE YOUR CODE HERE
            
            # 5.3 Compute loss (using 'criterion').
            loss = self.criterion(logits,y)  ### WRITE YOUR CODE HERE
            
            # 5.4 Run backward pass.
            loss.backward()  ### WRITE YOUR CODE HERE^
            
            # 5.5 Update the weights using 'optimizer'.
            self.optimizer.step()  ### WRITE YOUR CODE HERE
            
            # 5.6 Zero-out the accumulated gradients.
            self.model.zero_grad()  ### WRITE YOUR CODE HERE^

            print('\rEp {}/{}, it {}/{}: loss train: {:.2f}, accuracy train: {:.2f}'.
                format(ep + 1, epochs, it + 1, len(dataloader), loss,
                        utils.accuracy(logits, y)), end='')

    def predict_torch(self, dataloader):
        """
        Predict the validation/test dataloader labels using the model.

        Hints:
            1. Don't forget to set your model to eval mode, i.e., self.model.eval()!
            2. You can use torch.no_grad() to turn off gradient computation, 
            which can save memory and speed up computation. Simply write:
                with torch.no_grad():
                    # Write your code here.

        Arguments:
            dataloader (DataLoader): dataloader for validation/test data
        Returns:
            pred_labels (torch.tensor): predicted labels of shape (N,),
                with N the number of data points in the validation/test data.
        """
        self.model.eval()
        with torch.no_grad():
            acc_run = 0
            for it, batch in enumerate(dataloader):
                # Get batch of data.
                x, y = batch
                curr_bs = x.shape[0]
                acc_run += utils.accuracy(self.model(x), y) * curr_bs
            acc = acc_run / len(dataloader.dataset)

            print(', accuracy test: {:.2f}'.format(acc))
        return pred_labels
    
    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        This serves as an interface between numpy and pytorch.

        Arguments:
            training_data (array): training data of shape (N,D)
            training_labels (array): regression target of shape (N,)
        Returns:
            pred_labels (array): target of shape (N,)
        """

        # First, prepare data for pytorch
        train_dataset = TensorDataset(torch.from_numpy(training_data).float(), 
                                      torch.from_numpy(training_labels))
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        self.train_all(train_dataloader)

        return self.predict(training_data)

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        This serves as an interface between numpy and pytorch.
        
        Arguments:
            test_data (array): test data of shape (N,D)
        Returns:
            pred_labels (array): labels of shape (N,)
        """
        # First, prepare data for pytorch
        test_dataset = TensorDataset(torch.from_numpy(test_data).float())
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        pred_labels = self.predict_torch(test_dataloader)

        # We return the labels after transforming them into numpy array.
        return pred_labels.cpu().numpy()