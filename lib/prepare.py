import pandas as pd
import numpy as np
from itertools import combinations
from scipy.stats import spearmanr
from scipy.stats import entropy
from scipy.stats import zscore
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import sys
import random
import pickle as pk
import torch

torch.manual_seed(7)

def co_occurence(bindir, input_file, subject):
    with open(input_file, "r") as handle:
        filecontent = handle.readlines()

    data = []
    for i in range(1, len(filecontent)):
        data.append(list(map(float, filecontent[i].replace("\n", "").split(",")[1:])))

    data = np.array(list(data))

    data += 1e-6  
    geom_mean = np.exp(np.mean(np.log(data), axis=1, keepdims=True))
    data = np.log(data/geom_mean)
    
    spearman_matrix = np.zeros((len(data), len(data)))
    spearman_p = np.zeros((len(data), len(data)))
    for i, j in combinations(range(len(data)), 2):
        spearman_matrix[i, j] = spearmanr(data[i], data[j]).correlation
        spearman_matrix[j, i] = spearman_matrix[i, j]
        spearman_p[i, j] = spearmanr(data[i], data[j]).pvalue
        spearman_p[j, i] = spearman_p[i, j]

    p_values_flat = [value for row in spearman_p for value in row]
    rejected, corrected_p_values, _, _ = multipletests(p_values_flat, method='fdr_bh')
    corrected_p_values_matrix = [[corrected_p_values[i * len(spearman_p) + j] for j in range(len(spearman_p))] for i in range(len(spearman_p))]

    corrected_p_values_matrix = np.array(corrected_p_values_matrix)
    regulate_matrix = np.array(spearman_matrix).astype(np.float32)
    for i in range(0, len(corrected_p_values_matrix)):
        for j in range(0, len(corrected_p_values_matrix)):
            if corrected_p_values_matrix[i,j] > 0.05:
                regulate_matrix[i,j] = 0
    regulate_matrix = regulate_matrix.astype(np.float32)
    plt.imshow(regulate_matrix, cmap='hot', interpolation='nearest')
    plt.colorbar()  
    plt.savefig('{bindir}/{subject}-heatmap.png'.format(bindir=bindir, subject=subject))
    plt.clf()
    np.save("{bindir}/data/{subject}-adj_mat.npy".format(bindir=bindir, subject=subject), regulate_matrix)

def load_metr_la_data(X, means, stds):
    X = X.astype(np.float32)
    X = X - means.reshape(1, -1, 1)
    X = X / stds.reshape(1, -1, 1)

    return X

def generate_dataset(X, num_timesteps_input, num_timesteps_output):
    indices = [(i, i + (num_timesteps_input + num_timesteps_output)) for i
               in range(X.shape[2] - (
                num_timesteps_input + num_timesteps_output) + 1)]
    features, target = [], []
    for i, j in indices:
        features.append(
            X[:, :, i: i + num_timesteps_input].transpose(
                (0, 2, 1)))
        target.append(X[:, 0, i + num_timesteps_input: j])
    return np.array(features), \
           np.array(target)

def cut_datasets(training_input, training_target, val_input, val_target, replace_ratio):
    assert isinstance(training_input, torch.Tensor) and isinstance(training_target, torch.Tensor)
    assert isinstance(val_input, torch.Tensor) and isinstance(val_target, torch.Tensor)

    val_size = val_input.size(0)
    num_samples_to_add = int(replace_ratio * val_size)  
    
    if num_samples_to_add == 0:
        return training_input, training_target  

    step = max(1, val_size // num_samples_to_add)  
    replace_indices = torch.arange(0, val_size, step)[:num_samples_to_add]  
    if replace_indices[-1] != val_size - 1:
        replace_indices = torch.cat((replace_indices, torch.tensor([val_size - 1], dtype=torch.long)), dim=0)

    selected_val_input = val_input[replace_indices]
    selected_val_target = val_target[replace_indices]

    training_input = torch.cat((training_input, selected_val_input), dim=0)
    training_target = torch.cat((training_target, selected_val_target), dim=0)

    return training_input, training_target

def get_(bindir, input_file, subject, num_timesteps_input, num_timesteps_output, ctrl, ratio):
    with open(input_file,"r")as handle:
        contents = handle.readlines()

    node = []
    title_list = contents[0].replace("\n", "").split(",")[1:]
    samples = []
    for i in range(1, len(contents)):
        sample = []
        content_list = contents[i].replace("\n", "").split(",")
        for i in range(1,len(content_list)):
            sample.append([content_list[i]])
        samples.append(sample)

    node = list(np.array(samples).transpose((1,0,2)))
    print(np.array(node).shape)

    int(len(title_list)*0.8)

    if ctrl:
        train_original_data = node[0:int(len(title_list)*0.7)]
        val_original_data = node[int(len(title_list)*0.7):]
    else:
        train_original_data = node[0:int(len(title_list)*0.7)]
        val_original_data = node[int(len(title_list)*0.7):]
    node_list = []
    node_list = list(np.array(node).transpose((1, 2, 0)))

    X = np.array(node_list)
    X = X.astype(np.float32)
    means = np.mean(X, axis=(0, 2))
    stds = np.std(X, axis=(0, 2))
    train_original_data_normalization= list(load_metr_la_data(np.array(train_original_data).transpose((1,2,0)), means, stds))
    val_original_data_normalization = list(load_metr_la_data(np.array(val_original_data).transpose((1,2,0)), means, stds))
    training_input, training_target = [],[]
    features,target = generate_dataset(np.array(train_original_data_normalization), num_timesteps_input, num_timesteps_output)
    training_input = features
    training_target = target
    val_input, val_target = [],[]
    features,target = generate_dataset(np.array(val_original_data_normalization), num_timesteps_input, num_timesteps_output)
    val_input = features
    val_target = target
    if ctrl:
        pass 
    else:
        training_input, training_target = cut_datasets(torch.from_numpy(training_input), torch.from_numpy(training_target), torch.from_numpy(val_input), torch.from_numpy(val_target), ratio)
    with open("{bindir}/data/{subject}-train.pk".format(bindir=bindir, subject=subject), "wb") as fd:
         pk.dump((training_input, training_target, torch.from_numpy(val_input), torch.from_numpy(val_target), means, stds), fd)

if __name__ == "__main__":

    print("exit")
