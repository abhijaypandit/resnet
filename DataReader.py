import os
import pickle
import numpy as np

"""This script implements the functions for reading data.
"""

def load_data(data_dir):
    """Load the CIFAR-10 dataset.

    Args:
        data_dir: A string. The directory where data batches
            are stored.

    Returns:
        x_train: An numpy array of shape [50000, 3072].
            (dtype=np.float32)
        y_train: An numpy array of shape [50000,].
            (dtype=np.int32)
        x_test: An numpy array of shape [10000, 3072].
            (dtype=np.float32)
        y_test: An numpy array of shape [10000,].
            (dtype=np.int32)
    """

    ## Training Set
    for batch in range(1, 6):
        file = data_dir + "data_batch_{}".format(batch)

        with open(file, 'rb') as fileopen:
            dataset = pickle.load(fileopen, encoding='bytes')
        
        data = np.array(dataset[b'data']).astype(np.float32)
        labels = np.array(dataset[b'labels']).astype(np.int32)

        if batch == 1:
          x_train = data
          y_train = labels

        else:
          x_train = np.concatenate((x_train, data), axis=0)
          y_train = np.concatenate((y_train, labels), axis=0)

    ## Testing Set
    file = data_dir + "test_batch"

    with open(file, 'rb') as fileopen:
        dataset = pickle.load(fileopen, encoding='bytes')

    data = np.array(dataset[b'data']).astype(np.float32)
    labels = np.array(dataset[b'labels']).astype(np.int32)

    x_test = data
    y_test = labels

    return x_train, y_train, x_test, y_test


def load_test_images(data):
    """Load the images in private testing dataset.

    Args:
        data_dir: A string. The directory where the testing images
        are stored.

    Returns:
        x_test: An numpy array of shape [N, 32, 32, 3].
            (dtype=np.float32)
    """

    x_test = np.load(data).astype(np.float32)

    return x_test


def train_valid_split(x_train, y_train, train_ratio=0.8):
    """Split the original training data into a new training dataset
    and a validation dataset.

    Args:
        x_train: An array of shape [50000, 3072].
        y_train: An array of shape [50000,].
        train_ratio: A float number between 0 and 1.

    Returns:
        x_train_new: An array of shape [split_index, 3072].
        y_train_new: An array of shape [split_index,].
        x_valid: An array of shape [50000-split_index, 3072].
        y_valid: An array of shape [50000-split_index,].
    """

    split_index = int(x_train.shape[0]*train_ratio)

    x_train_new = x_train[:split_index]
    y_train_new = y_train[:split_index]
    x_valid = x_train[split_index:]
    y_valid = y_train[split_index:]

    return x_train_new, y_train_new, x_valid, y_valid
