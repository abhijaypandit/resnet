import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn.modules import module
#from torch.functional import Tensor
from tqdm import tqdm

from torchsummary import summary

from Network import Network
from ImageUtils import parse_record

"""This script defines the training, validation and testing process.
"""

class Model(nn.Module):

    def __init__(self, config, device):
        super(Model, self).__init__()
        self.config = config
        self.device = device
        self.network = Network(config).to(device)

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.network.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
        #self.optimizer = torch.optim.Adam(self.network.parameters(), lr=0.1, weight_decay=0.0001)
        #self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=75, gamma=0.1)

    def train(self, x_train, y_train, x_valid=None, y_valid=None):
        #self.network.train()
        
        num_samples = x_train.shape[0]
        num_batches = num_samples // self.config.batch_size

        train_loss = []
        train_accuracy = []
        valid_accuracy = []

        for epoch in range(1, self.config.epochs+1):
            self.network.train()

            # Adjust learning rate at 100th and 150th epoch
            if epoch > 150:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = 0.001
            elif epoch > 100:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = 0.01

            start_time = time.time()

            shuffle_index = np.random.permutation(num_samples)
            curr_x_train = x_train[shuffle_index]
            curr_y_train = y_train[shuffle_index]

            for batch in range(1, num_batches+1):
                batch_x = curr_x_train[batch*self.config.batch_size : (batch+1)*self.config.batch_size]
                batch_y = curr_y_train[batch*self.config.batch_size : (batch+1)*self.config.batch_size]

                batch_x_tensor = []

                for i in range(batch_x.shape[0]):
                    batch_x_tensor.append(parse_record(batch_x[i], 1))

                batch_x_tensor = np.array(batch_x_tensor)

                batch_x_tensor = torch.from_numpy(batch_x_tensor)
                batch_y_tensor = torch.from_numpy(batch_y)

                # Send tensor to CPU/GPU
                batch_x_tensor = batch_x_tensor.to(self.device)
                batch_y_tensor = batch_y_tensor.to(self.device)
                batch_y_tensor = batch_y_tensor.type(torch.LongTensor).to(self.device)

                outputs = self.network(batch_x_tensor)
                loss = self.loss(outputs, batch_y_tensor)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                print('\rBatch {:d}/{:d} Loss {:.6f} '.format(batch, num_batches, loss), end="")
            
            #self.scheduler.step()
            duration = time.time() - start_time
            print('Epoch {:d} Loss {:.6f} Duration {:.3f} seconds.'.format(epoch, loss, duration))

            train_loss.append(loss.cpu().detach().numpy())

            # Save and evaluate the model
            if epoch % self.config.save_interval == 0:
                torch.save(self.network.state_dict(), "../saved_models/model-{}.ckpt".format(epoch))
                print("Model checkpoint created.")
                #self.evaluate(x_valid, y_valid, epoch)

                train_accuracy.append(self.evaluate(x_train, y_train, epoch))

                if x_valid is not None and y_valid is not None:
                    valid_accuracy.append(self.evaluate(x_valid, y_valid, epoch))

        # Plot the metrics
        self.plot(train_loss, 'epochs', 'Training Loss')
        self.plot(train_accuracy, 'epochs*10', 'Training Accuracy')
        self.plot(valid_accuracy, 'epochs*10', 'Validation Accuracy')

    def evaluate(self, x, y, checkpoint):
        self.load_model(checkpoint)

        preds = []
        for i in tqdm(range(x.shape[0])):
            x_ = parse_record(x[i], 0).reshape(-1,3,32,32)
            x_ = torch.from_numpy(x_).to(self.device)
            prediction = self.network(x_)
            preds.append(torch.argmax(prediction[0]))

        y = torch.tensor(y)
        preds = torch.tensor(preds)
        accuracy = torch.sum(preds==y)/y.shape[0]
        print('Accuracy: {:.4f}'.format(accuracy))

        return accuracy
            
    def predict(self, x, checkpoint):
        self.load_model(checkpoint)
        
        preds = []
        for i in tqdm(range(x.shape[0])):
            x_ = parse_record(x[i], 0, 1).reshape(-1,3,32,32)
            x_ = torch.from_numpy(x_).to(self.device)
            prediction = self.network(x_)
            preds.append(prediction[0].cpu().detach().numpy())

        np.save('../predictions.npy', preds)

    def load_model(self, checkpoint):
        model = torch.load("../saved_models/model-{}.ckpt".format(checkpoint), map_location=self.device)
        self.network.load_state_dict(model, strict=True)
        print("Model checkpoint {} loaded.".format(checkpoint))
        self.network.eval()

    def plot(self, data, xlabel, ylabel):
        plt.figure()
        plt.plot(data)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.savefig('../{}.png'.format(ylabel))