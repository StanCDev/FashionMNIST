import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

## MS2
from .. import utils

class MLP(nn.Module):
    """
    An MLP network which does classification.

    It should not use any convolutional layers.
    """

    def __init__(self, input_size, n_classes, layers : list[int] = [256,128,64], dropout : bool = True,p : float =0.2):
        """
        Initialize the network.
        
        You can add arguments if you want, but WITH a default value, e.g.:
            __init__(self, input_size, n_classes, my_arg=32)
        
        Arguments:
            input_size (int): size of the input
            n_classes (int): number of classes to predict
            layers (list): list that specifies how many hidden layers and how many neurons per hidden layer
            p (float): ratio of dropped out weights
        """
        #super().__init__()
        assert len(layers) != 0 , "MLP has no hidden layers!"
        super(MLP, self).__init__()

        fc = []
        layers = [input_size] + layers + [n_classes]
        for i in range(len(layers) - 1):
            fc.append(nn.Linear(layers[i], layers[i+1]))
            fc.append(nn.ReLU())
            if dropout:
                fc.append(nn.Dropout(p))
        self.network = nn.Sequential(*fc)

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
        # for i, fc in enumerate(self.network):
        #     if i == len(self.network)-1:
        #         preds = fc(preds)
        #     else:
        #         preds = F.relu(fc(preds)) #not sure about the shape, what about softmax??
        preds = self.network(x)
        return preds


class CNN(nn.Module):
    """
    A CNN which does classification.

    It should use at least one convolutional layer.
    """

    def __init__(self, input_channels, n_classes, D = 28, conv_layers=[(6, 3, 1),(16, 3, 1)], fc_layers=[256, 128, 64]):
        """
        Initialize the network.
        
        You can add arguments if you want, but WITH a default value, e.g.:
            __init__(self, input_channels, n_classes, my_arg=32)
        
        Arguments:
            input_channels (int): number of channels in the input
            n_classes (int): number of classes to predict
            D (int): width / height of square image
            conv_layers (list[(int,int,int)]): list of triples (out_channels, kernel_size, and padding) for every 
                convolutional layer. For every conv. there is one max pooling
                Note that padding should be equal to (kernel size - 1) / 2
            fc_layers (list[int]): list of neurons for every hidden layer
        """
        super(CNN, self).__init__()

        ### ADD AN ASSERT WITH THE CONDITIONS
        nbr_pools : int = len(conv_layers)
        assert D % (2 ** nbr_pools) == 0 , "For every convolutional layer we divide the size of the image by two. \
            hence must be width/height of image must be dividible by 2 number of convolutional layer times"

        self.conv_layers = nn.ModuleList()

        in_channels : int = input_channels
        

        for out_channels, kernel_size, padding in conv_layers:
            self.conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding))
            in_channels = out_channels

        self.fc_layers = nn.ModuleList()

        ### note that output dimension is nbr of out put channels / 2 ** nbr_pools
        input_size = int( D * D * in_channels / 2 ** (2 * nbr_pools) )

        ##nn.Flatten()


        self.fc_layers = nn.ModuleList()
        layers = [input_size] + fc_layers + [n_classes]
        for i in range(len(layers) - 1):
            self.fc_layers.append(nn.Linear(layers[i], layers[i+1]))

    def forward(self, x):
        """
        Predict the class of a batch of samples with the model.

        Arguments:
            x (tensor): input batch of shape (N, Ch, H, W)
        Returns:
            preds (tensor): logits of predictions of shape (N, C)
                Reminder: logits are value pre-softmax.
        """
        for conv_layer in self.conv_layers:
            x = F.relu(conv_layer(x))
            x = F.max_pool2d(input=x, kernel_size=2)


        x = x.flatten(-3)
        preds = x
        for i, fc in enumerate(self.fc_layers):
            if i == len(self.fc_layers)-1:
                preds = fc(preds)
            else:
                preds = F.relu(fc(preds)) #not sure about the shape, what about softmax??
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
        for sequence in sequences: # got to change the for loop !!!!!!!!!!!!!
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

class MyViTBlock(nn.Module):
    def __init__(self, hidden_d, n_heads, mlp_ratio=4):
        super(MyViTBlock, self).__init__()
        self.hidden_d = hidden_d
        self.n_heads = n_heads

        self.norm1 = nn.LayerNorm(hidden_d)
        self.mhsa = MyMSA(hidden_d, n_heads)
        self.norm2 = nn.LayerNorm(hidden_d)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_d, mlp_ratio * hidden_d),
            nn.GELU(),
            nn.Linear(mlp_ratio * hidden_d, hidden_d)
        )

    def forward(self, x):
        out = x + self.mhsa(self.norm1(x))
        out = out + self.mlp(self.norm2(out))
        return out


class MyViT(nn.Module):
    """
    A Transformer-based neural network
    """

    def __init__(self, chw, n_patches, n_blocks, hidden_d, n_heads, out_d):
        """
        Initialize the network.
        
        """
        super().__init__()

        self.chw = chw # ( C , H , W )
        self.n_patches = n_patches
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.hidden_d = hidden_d
        
        # Input and patches sizes
        assert chw[1] % n_patches == 0 # Input shape must be divisible by number of patches
        assert chw[2] % n_patches == 0 # Input shape must be divisible by number of patches
        self.patch_size = (chw[1] / n_patches, chw[2] / n_patches)

        # 1) Linear mapper
        self.input_d = int(chw[0] * self.patch_size[0] * self.patch_size[1])
        self.linear_mapper = nn.Linear(self.input_d, self.hidden_d)
        
        # 2) Learnable classification token
        self.class_token = nn.Parameter(torch.rand(1, self.hidden_d))
        
        # 3) Positional embedding
        self.positional_embeddings = utils.get_positional_embeddings(n_patches ** 2 + 1, hidden_d)
        
        # 4) Transformer encoder blocks
        self.blocks = nn.ModuleList([MyViTBlock(hidden_d, n_heads) for _ in range(n_blocks)])

        # 5) Classification MLPk
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_d, out_d),
            nn.Softmax(dim=-1)
        )


        ##
        ###
        #### WRITE YOUR CODE HERE!
        ###
        ##
        
    
    
    def forward(self, x):
        """
        Predict the class of a batch of samples with the model.

        Arguments:
            x (tensor): input batch of shape (N, Ch, H, W), N is the number of images, Ch the number of channels and H and W height and width
        Returns:
            preds (tensor): logits of predictions of shape (N, C)
                Reminder: logits are value pre-softmax.
        """
        # Dividing images into patches
        n, c, h, w = x.shape
        patches = utils.patchify(x, self.n_patches)
        
        # Running linear layer tokenization
        # Map the vector corresponding to each patch to the hidden size dimension
        tokens = self.linear_mapper(patches)
        
        # Adding classification token to the tokens
        tokens = torch.cat((self.class_token.expand(n, 1, -1), tokens), dim=1)
        
        # Adding positional embedding
        out = tokens + self.positional_embeddings.repeat(n, 1, 1)
        
        # Transformer Blocks
        for block in self.blocks:
            out = block(out)

        # Getting the classification token only
        out = out[:, 0]
        
        return self.mlp(out) # Map to output dimension, output category distribution
            
        
        


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
            print("")

    def train_one_epoch(self, dataloader, ep, epochs):
        """
        Train the model for ONE epoch.

        Should loop over the batches in the dataloader. (Recall the exercise session!)
        Don't forget to set your model to training mode, i.e., self.model.train()!

        Arguments:
            dataloader (DataLoader): dataloader for training data
        """
        self.model.train()
        for it, batch in enumerate(dataloader):
            # 5.1 Load a batch, break it down in images and targets.
            x, y = batch
            y = y.long()
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
        pred_labels = []
        with torch.no_grad():
            for it, x in enumerate(dataloader):
                x = x[0]
                y = self.model(x)
                pred_labels.append(torch.argmax(y, dim=1)) # or just y
        return torch.cat(pred_labels)
    
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