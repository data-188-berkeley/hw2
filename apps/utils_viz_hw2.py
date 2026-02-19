import os

import needle as ndl
import needle.nn as nn

"""
Some optional visualization utils for HW2.
"""


def visualize_classification_preds(model: nn.Module, test_dataset: ndl.data.Dataset):
    """Given a trained classification model, visualize the predictions on an input
    dataset.

    Args:
        model (nn.Module): Trained digit classification model. The forward method
            should output predicted logits.
        test_dataset (ndl.data.Dataset): Test dataset to inference on.
    """
    # Note: this assumes that the entire test set can fit into memory.
    #   This is OK for small datasets (like MNIST), but for larger datasets
    #   we'll want to use a Dataloader and iterate over batches.
    X_te = ndl.Tensor(test_dataset.X)  # shape=[num_samples, im_height, im_width, num_chans=1]
    y_te = ndl.Tensor(test_dataset.y)  # shape=[num_samples]

    # Visualize model predictions
    from needle.utils.visualize_mnist import visualize_mnist_epoch
    X_te_flat = nn.Flatten()(X_te)  # shape=[batchsize, n]
    final_pred_logits = model(X_te_flat)
    for digit in range(0, 10):
        visualize_mnist_epoch(final_pred_logits.numpy(), X_te.numpy(), 1, 28, 28, y_te.numpy(), digit=digit, top_k=5)


def visualize_mnist_classifier(model: nn.Module, data_dir: str = "data"):
    # Load test set
    test_dataset = ndl.data.MNISTDataset(
        os.path.join(data_dir, "t10k-images-idx3-ubyte.gz"),
        os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz"),
    )
    visualize_classification_preds(model, test_dataset)
