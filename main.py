import os
import torch
import argparse
import numpy as np

from torchsummary import summary

from DataReader import load_data, train_valid_split, load_test_images
from ImageUtils import show_image
from Model import Model

def configure():
    parser = argparse.ArgumentParser()
    
    # Command line arguments and their default values 
    
    parser.add_argument("--block_size", type=int, default=3, help='size of each residual block')
    parser.add_argument("--batch_size", type=int, default=128, help='training batch size')
    parser.add_argument("--epochs", type=int, default=200, help='number of epochs')
    parser.add_argument("--start_filters", type=int, default=16, help='number of filters for input convolution')
    parser.add_argument("--num_classes", type=int, default=10, help='number of classes for classification')
    parser.add_argument("--save_interval", type=int, default=10, help='model checkpoint save interval')
    parser.add_argument("--training", type=bool, default=False, help='set mode to training')
    parser.add_argument("--testing", type=bool, default=False, help='set mode to testing')
    #parser.add_argument("--log_accuracy", type=bool, default=False, help='enable logging training and validation accuracy')

    return parser.parse_args()

def main(config):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    model = Model(config, device)
    #summary(model.network, (3,32,32))
    #model.to(device)

    data_dir = "../data/cifar-10-batches-py/"

    x_train, y_train, x_test, y_test = load_data(data_dir)

    ### Training mode ### 
    if config.training is True:
        #x_train, y_train, x_valid, y_valid = train_valid_split(x_train, y_train)

        #model.train(x_train, y_train, x_valid, y_valid)
        model.train(x_train, y_train)

    ### Testing mode ###
    elif config.testing is True:
        #model.train(x_train, y_train)
        model.evaluate(x_test, y_test, 190)

    ### Prediction mode ###
    else:
        x_test_private = load_test_images('../private_test_images_v3.npy')
        classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

        if os.path.exists('../predictions.npy') is False:
            model.predict(x_test_private, 190)
        preds = np.load('../predictions.npy')

        idx = np.random.randint(2000)
        print('Probability distribution:', preds[idx])
        print('Prediction:', classes[np.argmax(preds[idx])])
        show_image(x_test_private[idx], 1)

if __name__ == '__main__':
    config = configure()
    main(config)
 