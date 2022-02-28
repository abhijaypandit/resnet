# Fall 2021 Term Project 

Author: Abhijay Pandit

This folder contains all the source codes necessary to train and test model on CIFAR-10 dataset.
Note that the CIFAR-10 dataset should be present in '../data/cifar-10-batches-py/' directory.

The code by default runs in the prediction mode.

To run, open the code directory in a terminal and run
> python main.py

This will load the file '../predictions.npy' if available, else will create one by using a pretrained model.
It will also perform prediction on a random test image.

Note that the file '../private_test_images_v3.npy' should be available before starting prediction.

To train the model, run
> python main.py --training 1

This will train the model and save model checkpoints in '../saved_models' directory.

To test the model, run
> python main.py --testing 1

This will evaluate the model accuracy using a pretrained model on the test set.

Available command line arguments are-
"--block_size", type=int, default=3, help='size of each residual block'
"--batch_size", type=int, default=128, help='training batch size'
"--epochs", type=int, default=200, help='number of epochs'
"--start_filters", type=int, default=16, help='number of filters for input convolution'
"--num_classes", type=int, default=10, help='number of classes for classification'
"--save_interval", type=int, default=10, help='model checkpoint save interval'
"--training", type=bool, default=False, help='set mode to training'
"--testing", type=bool, default=False, help='set mode to testing'

To specify any of the above command line arguments, run
> python main.py --<argument_name> <type_value>
