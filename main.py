import os
import sys
bin = os.path.abspath(os.path.dirname(__file__))
sys.path.append(bin + "/lib")
import argparse
import pickle as pk
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import json

from stgcn import TwoStreamSTGCN
from stgcn import LSTM 
from prepare import co_occurence, get_



__doc__ = """
MicroSTNet Help Manual:
    Function: Predict the relative abundance of microorganisms at a future moment in time
    Input file requirements: csv file containing row names and column names, row names are species names and column names are time information.
    Output: 1. Predicted versus true value curve on the validation set.
            2. MAE/MSE curve.
            3. Table of abundance information at future moments
"""
__author__="Gao Shichen"
__mail__= "gaoshichend@163.com"
__date__= "2024/10/09"
__update__ = "2024/12/24"

class Config():
    best_loss = 10000
    best_mae = 10000

config = Config()

def train_epoch(training_input, training_input_Motion, training_target, batch_size):
    permutation = torch.randperm(training_input.shape[0])

    epoch_training_losses = []
    for i in range(0, training_input.shape[0], batch_size):
        net.train()
        optimizer.zero_grad()

        indices = permutation[i:i + batch_size]
        X_batch, X_motion_batch, y_batch = training_input[indices], training_input_Motion[indices], training_target[indices]
        X_batch = X_batch.to(device=args.device)
        X_motion_batch = X_motion_batch.to(device=args.device)
        y_batch = y_batch.to(device=args.device)

        out = net(A_wave, X_batch, X_motion_batch)
        loss = loss_criterion(out, y_batch)
        loss.backward()
        optimizer.step()
        epoch_training_losses.append(loss.detach().cpu().numpy())
    return sum(epoch_training_losses)/len(epoch_training_losses)

class Motion(Dataset):
    def __init__(self, arrays):
        self.arrays = arrays.numpy()

    def __calc__bk(self):
        for i in range(self.arrays.shape[0]):
            for j in range(self.arrays.shape[1]):
                sub_array = self.arrays[i, j, :, 0]
                self.arrays[i, j, :, 0] = sub_array
        return self.arrays
    
    def __calc__(self, target):
        self.target = target.numpy()
        self.target_copy = np.zeros(shape=self.target.shape)
        for i in range(self.arrays.shape[0]):
            for j in range(self.arrays.shape[1]):
                sub_array = self.arrays[i, j, :, 0]
                diff_array = np.diff(sub_array, prepend=sub_array[0])
                for t in range(0, int(self.target.shape[2])):
                    if t == 0:
                        self.target_copy[i,j,t] = self.target[i,j,t] - self.arrays[i, j, -1, 0]
                    else:
                        self.target_copy[i,j,t] = self.target[i,j,t] - self.target[i,j,t-1]
                self.arrays[i, j, :, 0] = diff_array
        return torch.from_numpy(self.arrays), torch.from_numpy(self.target_copy)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="author:\t{0}\nmail:\t{1}\ndate:\t{2}\n".format(__author__,__mail__,__date__))
    parser.add_argument('--enable-cuda', dest="enable_cuda", action='store_true',
                        help='Enable CUDA')
    parser.add_argument('--lstm', dest="lstm", action='store_true',
                        help='Use LSTM model')
    parser.add_argument('--num_timesteps_input', dest="num_timesteps_input", type=int, default=12, help='[ Optional Default(12) ]: Set the length of the time step used for training.')
    parser.add_argument('--num_timesteps_output', dest='num_timesteps_output', type=int, default=1, help='[ Optional Default(1) ]: Set the length of the time step for the prediction output.')
    parser.add_argument('-e', '--epochs', dest='epochs', type=int, default=200, help='[ Optional Default:100 ]: Setting the epochs size.')
    parser.add_argument('-b', '--batch_size', dest='batch_size', type=int, default=8, help='[ Optional Default(30) ]: Setting the batch size size.')
    parser.add_argument('-t', '--threads', dest='threads', type=int, default=120, help='[ Optional Default(40) ]: Setting the number of threads to use.')
    parser.add_argument('-l', '--loss_function', dest='loss_function', type=str, choices=['MSELoss','L1Loss'], default='L1loss', help='[ Optional Default(L1loss) ]: Set the type of loss function used for model training.')
    parser.add_argument('-i', '--input', dest='input', type=str, help='[ Required ]: Setting the input file.')
    parser.add_argument('-r', '--ratio', dest='ratio', type=float, default=0.1)
    parser.add_argument('-s', '--subject', dest='subject', type=str, help='[ Required ]: Setting the subject name.')
    args = parser.parse_args()
    args.device = None
    if args.enable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        print("Note: Will train the model on GPU!")
    else:
        args.device = torch.device('cpu')
        print("Note: Will train the model on CPU!")

    torch.set_num_threads(args.threads)

    torch.manual_seed(7)
    co_occurence(bin, args.input, args.subject)
    get_(bin, args.input, args.subject, args.num_timesteps_input, args.num_timesteps_output, args.lstm, args.ratio)
    A = np.load("data/{subject}-adj_mat.npy".format(subject=args.subject))
    with open("data/{subject}-train.pk".format(subject=args.subject),"rb")as handle:
        data = pk.load(handle)
    training_input, training_target, val_input, val_target, means, stds = data[0]+1, data[1]+1, data[2]+1, data[3]+1, data[4], data[5]

    training_input_Motion, training_target_Motion = Motion(training_input).__calc__(training_target)
    val_input_Motion, val_target_Motion = Motion(val_input).__calc__(val_target)
    
    A_wave = A
    A_wave = torch.from_numpy(A_wave)

    A_wave = A_wave.to(device=args.device)
    lstm_type = ""
    if args.lstm:
        lstm_type = "-lstm"
        net = LSTM(A_wave.shape[0],
                    training_input.shape[3],
                    args.num_timesteps_input,
                    args.num_timesteps_output).to(device=args.device)
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
        loss_criterion = nn.MSELoss()
    else:
        net = TwoStreamSTGCN(A_wave.shape[0],
                    training_input.shape[3],
                    args.num_timesteps_input,
                    args.num_timesteps_output).to(device=args.device)
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
        loss_criterion = nn.L1Loss()
        if args.loss_function == 'MSELoss':
            loss_criterion = nn.MSELoss()
        elif args.loss_function == 'L1Loss':
            loss_criterion = nn.L1Loss()

    training_losses = []
    validation_losses = []
    validation_maes = []
    for epoch in range(args.epochs):
        loss = train_epoch(training_input, training_input_Motion, training_target,
                           batch_size=args.batch_size)
        training_losses.append(loss)

        val_loss = 0
        with torch.no_grad():
            net.eval()
            val_input = val_input.to(device=args.device)
            val_input_Motion = val_input_Motion.to(device=args.device)
            val_target = val_target.to(device=args.device)

            out = net(A_wave, val_input, val_input_Motion)
            val_loss = loss_criterion(out, val_target).to(device="cpu")
            validation_losses.append(np.ndarray.item(val_loss.detach().numpy()))
            out_unnormalized = out.detach().cpu().numpy()
            target_unnormalized = val_target.detach().cpu().numpy()

            mae = np.mean(np.absolute(out_unnormalized - target_unnormalized))
            validation_maes.append(mae)

            out = None
            val_input = val_input.to(device="cpu")
            val_target = val_target.to(device="cpu")

        if val_loss < config.best_loss:
            config.best_loss = val_loss
            torch.save(net.state_dict(), "checkpoints/{subject}{lstm_type}-best_loss.pth".format(subject = args.subject, lstm_type=lstm_type))
        if mae < config.best_mae:
            config.best_mae = mae
            torch.save(net.state_dict(), "checkpoints/{subject}{lstm_type}-best_mae.pth".format(subject = args.subject, lstm_type=lstm_type))

        print("epoch: {epoch}".format(epoch=epoch))
        print("Training loss: {}".format(training_losses[-1]))
        print("Validation loss: {}".format(validation_losses[-1]))
        print("Validation MAE: {}\n".format(validation_maes[-1]))
    plt.plot(training_losses, label="training loss")
    plt.plot(validation_losses, label="validation loss")
    plt.legend()
    plt.title("MAELoss Over Epochs: {subject} Dataset".format(subject = args.subject))
    plt.savefig("{subject}{lstm_type}-loss.jpg".format(subject = args.subject, lstm_type=lstm_type), dpi=300)
    plt.clf()
    checkpoint_path = "checkpoints/"
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    with open("checkpoints/{subject}{lstm_type}-losses.pk".format(subject = args.subject, lstm_type=lstm_type), "wb") as fd:
        pk.dump((training_losses, validation_losses, validation_maes), fd)
    plt.plot(validation_maes, label="validation maes")
    plt.legend()
    plt.title("MAE Over Epochs: {subject}{lstm_type} Dataset".format(subject = args.subject, lstm_type=lstm_type))
    plt.savefig("{subject}{lstm_type}-mae.jpg".format(subject = args.subject, lstm_type=lstm_type), dpi=300)

    with open("forecasting.json", "r") as file:
        data = json.load(file)
    key = "{subject}-{timestep}-mae".format(subject = args.subject,timestep=args.num_timesteps_output)

    if key in data:
        data[key] = str(config.best_mae)
    else:
        data[key] = str(config.best_mae)
    with open("forecasting.json", "w") as file:
        json.dump(data, file, indent=4)