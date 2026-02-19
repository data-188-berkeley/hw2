import sys

sys.path.append("../python")
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)
# MY_DEVICE = ndl.backend_selection.cuda()


def ResidualBlock(dim: int, hidden_dim: int, norm: nn.Module = nn.BatchNorm1d, drop_prob: float = 0.1) -> nn.Module:
    """Constructs a ResidualBlock, which consists of the following layers:
        in->Linear(in=d,out=h)->Norm->Relu->Dropout->Linear(in=d,out=h)->Norm->EWiseAdd-> out
        |                                                                         ^
        |                                                                         |
        ---------------------------------------------------------------------------

    Where `d` denotes input (and output) dimensionality `dim`, `h` denotes hidden dimension,
    and the (long) ASCII line represents the residual ("skip") connection

    Args:
        dim (int): Input (and output) dimensionality.
        hidden_dim (int): Hidden dimension of the block.
        norm (nn.Module, optional): Activation normalization function. Defaults to nn.BatchNorm1d.
        drop_prob (float, optional): Dropout probability. Defaults to 0.1.

    Returns:
        nn.Module: output residual block.
    """
    ### BEGIN YOUR SOLUTION
    raise NotImplementedError()
    ### END YOUR SOLUTION


def MLPResNet(
    dim: int,
    hidden_dim: int = 100,
    num_blocks: int = 3,
    num_classes: int = 10,
    norm: nn.Module = nn.BatchNorm1d,
    drop_prob: float = 0.1,
) -> nn.Module:
    ### BEGIN YOUR SOLUTION
    raise NotImplementedError()
    ### END YOUR SOLUTION


def epoch(dataloader, model: nn.Module, opt=None) -> tuple[float, float]:
    """Runs input model on dataloader, for either training or eval.
    If `opt` is given: perform training.
    Else: assume eval ("test") mode.

    Assumes input model is a classifier. Uses SoftmaxLoss (aka cross-entropy loss).

    Args:
        dataloader (_type_): Dataloader to run training/testing on.
        model (nn.Module): Input model.
        opt (_type_, optional): Training optimizer (if given). Defaults to None.

    Returns:
        tuple[float, float]: (float avg_cls_error, float avg_loss), where:
            avg_cls_error: 0/1 classification error, averaged over all samples.
            avg_loss: SoftmaxLoss error, averaged over all samples.
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    raise NotImplementedError()
    ### END YOUR SOLUTION


def train_mnist(
    batch_size: int = 100,
    epochs: int = 10,
    optimizer=ndl.optim.Adam,
    lr: float = 0.001,
    weight_decay: float = 0.001,
    hidden_dim: int = 100,
    data_dir: str = "data",
)-> tuple[tuple[float, float, float, float], nn.Module]:
    """Trains an MNIST digit classifier.

    Args:
        batch_size (int, optional): Training batchsize to use. Defaults to 100.
        epochs (int, optional): Number of epochs to train for. Defaults to 10.
        optimizer (_type_, optional): Optimizer to use. Defaults to ndl.optim.Adam.
        lr (float, optional): Learning rate (aka stepsize, alpha). Defaults to 0.001.
        weight_decay (float, optional): aka L2 regularization param. Defaults to 0.001.
        hidden_dim (int, optional): Hidden dim of model architecture. Defaults to 100.
        data_dir (str, optional): Directory containing train/test data files. Defaults to "data".

    Returns:
        tuple[float, float, float, float]: errors/losses of final epoch of training.
            (train_err, train_loss, test_err, test_loss)
            where "*_err" is 0/1 classification error.
        nn.Module: the trained model.
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    raise NotImplementedError()
    ### END YOUR SOLUTION


if __name__ == "__main__":
    train_mnist(data_dir="../data")
