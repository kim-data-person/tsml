import os
import math
import zipfile
import numpy as np
import torch


def load_metr_la_data():
    '''
    Load METR-LA dataset (https://github.com/liyaguang/DCRNN)
    return:
        - A: weighted adjacency matrix (207, 207)
        - X: node values, normalized (207, 2, 34272)
        - means: mean values of X
        - stds: standard deviation values of X
    '''
    A = np.load("data/adj_mat.npy") # (207, 207)
    X = np.load("data/node_values.npy") # (207, 2, 34272)
    return A, X

def generate_weighted_adj_matrix(A):
    return A
"""
def generate_weighted_adj_matrix(lat_degree, long_degree, sigma2=10, epsilon=0.5):
    '''
    Input:
        - lat_degree: latitude of nodes (degree)
        - long_degree: longitude of nodes (degree)
        - sigma2, epsilon: thresholds to control the distribution and sparsity of matrix W
    Output:
        - W: weighted adjacency matrix
    '''

    def get_distance_km(diff_lat_degree, diff_long_degree):
        '''
        Note: valid only if diff_lat_degree<180 and diff_long_degree<180
        '''
        diff_lat_rad = math.radians(diff_lat_degree)
        diff_long_rad = math.radians(diff_long_degree)

        earth_radius_km = 6371.0 # Globally-averaged earth's radius in kilometers

        distance_km = earth_radius_km * (diff_lat_rad**2 + diff_long_rad**2)**0.5
        return distance_km

    def get_weight(distance_km, sigma2, epsilon):
        weight = np.exp(-distance_km**2 / sigma2)
        if weight < epsilon:
            return 0
        else:
            return weight
        
    num_nodes = len(lat_degree) # number of graph nodes
        
    W = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                distance_km = get_distance_km(lat_degree[i]-lat_degree[j], long_degree[i]-long_degree[j])
                W[i, j] = get_weight(distance_km, sigma2, epsilon)
    return W
"""

def get_normalized_adj_matrix(A):
    """
    Returns the degree normalized adjacency matrix.
    
    Input A contains the spatial information of the graph, graph signals in the position domain.
    Output A_wave represents the graph signal in the frequency domain.
    """
    A = A + np.diag(np.ones(A.shape[0], dtype=np.float32))
    D = np.array(np.sum(A, axis=1)).reshape((-1,)) # sum of each row, then reshape to 1D array
    D[D <= 10e-5] = 10e-5    # Prevent infs
    diag = np.reciprocal(np.sqrt(D)) 
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    return A_wave

def get_normalized_node_signal(X):

    X = X.transpose((1, 2, 0)) # (2, 34272, 207)
    X = X.astype(np.float32)

    # Normalization using Z-score method
    means = np.mean(X, axis=(0, 2)) # mean of each feature
    X = X - means.reshape(1, -1, 1) # (1,2,1), subtract mean of each feature
    stds = np.std(X, axis=(0, 2)) # std of each feature
    X = X / stds.reshape(1, -1, 1) # (1,2,1), devided by std of each feature

    return X, means, stds

def normalize_data(A, X):
    X, means, stds = get_normalized_node_signal(X)
    A = generate_weighted_adj_matrix(A)
    A_wave = get_normalized_adj_matrix(A)
    A_wave = torch.from_numpy(A_wave)

    return A_wave, X, means, stds

def generate_dataset(X, num_timesteps_input, num_timesteps_output):
    """
    Takes node features for the graph and divides them into multiple samples
    along the time-axis by sliding a window of size (num_timesteps_input+
    num_timesteps_output) across it in steps of 1.
    :param X: Node features of shape (num_vertices, num_features,
    num_timesteps)
    return:
        - Node features divided into multiple samples. Shape is
          (num_samples, num_vertices, num_features, num_timesteps_input).
        - Node targets for the samples. Shape is
          (num_samples, num_vertices, num_features, num_timesteps_output).
    """
    # Generate the beginning index and the ending index of a sample, which
    # contains (num_points_for_training + num_points_for_predicting) points
    indices = [(i, i + (num_timesteps_input + num_timesteps_output)) for i
               in range(X.shape[2] - (
                num_timesteps_input + num_timesteps_output) + 1)]

    # Save samples
    features, target = [], []
    for i, j in indices:
        features.append(
            X[:, :, i: i + num_timesteps_input].transpose(
                (0, 2, 1))) 
        target.append(X[:, 0, i + num_timesteps_input: j]) # (num_samples, num_nodes, num_timesteps_output, num_features)

    # features.shape = (num_samples, num_nodes=207, num_timesteps_input=12, num_features=2)
    # target.shape = (num_samples, num_nodes=207, num_timesteps_output=3), only 0th feature

    features_tensor = torch.from_numpy(np.array(features))
    target_tensor = torch.from_numpy(np.array(target))

    #print('features_tensor.shape', features_tensor.shape)
    #print('target_tensor.shape', target_tensor.shape)

    return features_tensor, target_tensor

def split_dataset(X, num_timesteps_input, num_timesteps_output):
    # Split dataset into train, val, test
    split_line1 = int(X.shape[2] * 0.6)
    split_line2 = int(X.shape[2] * 0.8)

    train_original_data = X[:, :, :split_line1]
    val_original_data = X[:, :, split_line1:split_line2]
    test_original_data = X[:, :, split_line2:]

    train_input, train_target = generate_dataset(train_original_data,
                                                       num_timesteps_input=num_timesteps_input,
                                                       num_timesteps_output=num_timesteps_output)
    val_input, val_target = generate_dataset(val_original_data,
                                             num_timesteps_input=num_timesteps_input,
                                             num_timesteps_output=num_timesteps_output)
    test_input, test_target = generate_dataset(test_original_data,
                                               num_timesteps_input=num_timesteps_input,
                                               num_timesteps_output=num_timesteps_output)
    
    split_data = {}
    split_data['train_input'] = train_input
    split_data['train_target'] = train_target
    split_data['val_input'] = val_input
    split_data['val_target'] = val_target
    split_data['test_input'] = test_input
    split_data['test_target'] = test_target
    
    return split_data