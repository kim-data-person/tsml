import numpy as np
import pickle
import torch
from torch.utils.data import Dataset


def get_spatial_conv_multiplier(W: np.array):

    I = np.eye(W.shape[0])
    D_tilde = np.diag(np.sum(W, axis=1)) + I
    W_tilde = W + I

    D_tilde_sqrt_inv = np.linalg.inv(np.sqrt(D_tilde))

    # Calculate the multiplicator of spatial convolution block - 1st order Chebyshev polynomial approximation 
    spatial_conv_multiplier = D_tilde_sqrt_inv @ W_tilde @ D_tilde_sqrt_inv

    return spatial_conv_multiplier

def get_normalized_node_features(features: list, prefecture_names: list):
    """
    Args:
        features: list of features to be used as input
        prefecture_names: list of prefecture names (generated with weighted adjacency matrix)
    Returns:
        X: normalized input of shape (num_nodes, num_features, num_timesteps)
        means: means of each feature
        stds: std of each feature
    """
    # load the pollution data
    with open('output/cleaned_data.pkl', 'rb') as f:
        df = pickle.load(f)

    # Set the model input X dimension
    num_timesteps = min([len(df[prefecture]) for prefecture in df.keys()]) 
    X = np.zeros((len(prefecture_names), len(features), num_timesteps)) # (num_nodes, num_features, num_timesteps)

    # Preserve the time index for the later plotting
    timestamps = df[prefecture_names[0]].index

    # Fill X
    for i in range(len(prefecture_names)):
        for j in range(len(features)):
            for k in range(num_timesteps):
                X[i, j, k] = df[prefecture_names[i]].iloc[k][features[j]]
    
    X = X.astype(np.float32)

    # Normalization using Z-score method
    means = np.mean(X, axis=(0, 2)) # mean of each feature
    X = X - means.reshape(1, -1, 1) # subtract mean for each feature
    stds = np.std(X, axis=(0, 2)) # std of each feature
    X = X / stds.reshape(1, -1, 1) # devided by std for each feature

    # Replace the values that are too small with 10e-5
    X[X <= 10e-5] = 10e-5

    return X, means, stds, timestamps

def segment_timesteps(X, num_timesteps_input, num_timesteps_output, output_feature_index):
    """
    Args:
        X: Node features of shape (num_nodes, num_features, num_timesteps)
        num_timesteps_input: number of timesteps for input
        num_timesteps_output: number of timesteps for output
    Returns:
        features_tensor: features of shape (num_samples, num_nodes, num_timesteps_input, num_features)
        target_tensor: target of shape (num_samples, num_nodes, num_features, num_timesteps_output)
    """
    # Generate the beginning index and the ending index of a sample, which
    # contains (num_points_for_training + num_points_for_predicting) points
    indices = [(i, i + (num_timesteps_input + num_timesteps_output)) for i
               in range(X.shape[2] - (
                num_timesteps_input + num_timesteps_output) + 1)]

    features, target = [], []
    for i, j in indices:
        features.append(
            X[:, :, i: i + num_timesteps_input].transpose(
                (0, 2, 1))) # (num_nodes, num_timesteps_input, num_features)
        target.append(X[:, output_feature_index, i + num_timesteps_input: j]) # (num_nodes, num_features, num_timesteps_output)

    features_tensor = torch.from_numpy(np.array(features))
    target_tensor = torch.from_numpy(np.array(target))

    return features_tensor, target_tensor


class ModelDataset(Dataset):
    '''
    To use batch_size in DataLoader for validation
    '''
    def __init__(self, features_tensor, target_tensor):
        self.features = features_tensor
        self.targets = target_tensor

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]
    

def split_dataset(X, timestamps, num_timesteps_input, num_timesteps_output, output_feature_index=0):
    """
    Args:
        X: Node features of shape (num_nodes, num_features, num_timesteps)
        timestamps: timestamps of X 
        num_timesteps_input: number of timesteps for input
        num_timesteps_output: number of timesteps for output
    Returns:
        split_data: dictionary containing train, val, test data
            *_input: features of shape (num_samples, num_nodes, num_timesteps_input, num_features)
            *_target: target of shape (num_samples, num_nodes, num_timesteps_output)
    """
    # Split dataset into train, val, test
    split_line1 = int(X.shape[2] * 0.6)
    split_line2 = int(X.shape[2] * 0.8)

    train_original_data = X[:, :, :split_line1]
    val_original_data = X[:, :, split_line1:split_line2]
    test_original_data = X[:, :, split_line2:]

    test_timestamps = timestamps[split_line2:]


    train_input, train_target = segment_timesteps(train_original_data,
                                                       num_timesteps_input=num_timesteps_input,
                                                       num_timesteps_output=num_timesteps_output,
                                                       output_feature_index=output_feature_index)
    val_input, val_target = segment_timesteps(val_original_data,
                                             num_timesteps_input=num_timesteps_input,
                                             num_timesteps_output=num_timesteps_output,
                                             output_feature_index=output_feature_index)
    test_input, test_target = segment_timesteps(test_original_data,
                                               num_timesteps_input=num_timesteps_input,
                                               num_timesteps_output=num_timesteps_output,
                                               output_feature_index=output_feature_index)
        
    split_data = {}
    split_data['train_input'] = train_input
    split_data['train_target'] = train_target
    split_data['val_input'] = val_input
    split_data['val_target'] = val_target
    split_data['test_input'] = test_input
    split_data['test_target'] = test_target
    split_data['test_timestamps'] = test_timestamps

    return split_data

