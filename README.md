# FashionMNIST
FashionMNIST is an exploratory library to test Deep Learning methods on the Fashion-MNIST toy data set. Supported architectures are an MLP, Le net CNN, Residual CNN, and a ViT.

## Getting started
Start off by cloning the repo, and run the following to train a simple MLP.

The required packages are [pytorch](https://pytorch.org/), [torchinfo](https://pypi.org/project/torchinfo/) and [numpy](https://numpy.org/).

```bash
cd 358003_340698_377372_project
python --data <Path_to_FMNIST_dataset>
```

## Command-Line Arguments

The script accepts several command-line arguments for configuring the training process:

- **`--data`** (default: `"dataset"`): Specifies the path to your dataset. This should be a string indicating the directory location of the dataset.

- **`--nn_type`** (default: `"mlp"`): Chooses the neural network architecture to use. Options are `'mlp'`, `'transformer'`, `'cnn'`, or `'res'`.

- **`--nn_batch_size`** (default: `64`): Defines the batch size for neural network training. This should be an integer value.

- **`--device`** (default: `"cpu"`): Sets the device to use for training. Available options are `'cpu'`, `'cuda'` (for NVIDIA GPUs), or `'mps'` (for Apple Silicon devices).

- **`--use_pca`**: If this flag is set, Principal Component Analysis (PCA) will be used for feature reduction.

- **`--pca_d`** (default: `100`): Specifies the number of principal components to retain when using PCA for feature reduction. This should be an integer.

- **`--save`** (default: `"NONE"`): Indicates the path where you want to save your model after training. Provide a string with the desired save location.

- **`--load`** (default: `"NONE"`): Specifies the path from where you want to load a pre-trained model. Provide a string with the load location.

- **`--lr`** (default: `1e-5`): Sets the learning rate for training methods that require it. This should be a float value.

- **`--max_iters`** (default: `100`): Defines the maximum number of iterations for iterative training methods. This should be an integer.

- **`--test`**: If this flag is set, the model will train on the entire training dataset and evaluate on the test data. Otherwise, the script will use a validation set for evaluation.

## Report

We wrote a report explaining our [results and experiments](FashionMNIST/report.pdf).

## Contributors

[@StanCDev](https://github.com/StanCDev)
[@alicecnm](https://github.com/alicecnm)
[@CyrillStrassburg](https://github.com/CyrillStrassburg)