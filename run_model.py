import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from weighted_adj_mat import get_weighted_adjacency_matrix
from preprocessing import get_spatial_conv_multiplier, get_normalized_node_features, split_dataset
from model import STGCN


def prep_data(input_features: list, output_feature_index: int, num_timesteps_input: int, num_timesteps_output: int, split_ratio: tuple = (0.6, 0.8)):
#def prep_data(input_features: list, num_timesteps_input: int, num_timesteps_output: int, split_ratio: tuple = (0.6, 0.8)):
    '''
    Wrapper function to load, clean, and preprocess the data for the model input.    
    Args:
        input_features: list of input features
        output_feature_index: output feature as the index of the input features list
        num_timesteps_input: input window size as the number of timesteps
        num_timesteps_output: output window size as the number of timesteps
    Returns:
        data: dictionary containing train, val, test data and timestamps of the test data
        W: weighted adjacency matrix of shape (num_nodes, num_nodes)
        means: means of each feature (num_features,)
        stds: std of each feature (num_features,)
        prefecture_names: ordered list of prefecture names (num_nodes,). The order is consistent throughout everywhere in the analysis.
    '''
    # generate the adjacency matrix and its prefecture names following the matrix row/column order
    W, prefecture_names = get_weighted_adjacency_matrix(sigma=100, epsilon=10e-4)

    # generate the multiplier in the spatial convoltional block
    #spatial_conv_multiplier = get_spatial_conv_multiplier(W)

    # get the model input
    # timestamps: time index of the input (num_timesteps,), same through all the nodes and features
    X, means, stds, timestamps, X_raw = get_normalized_node_features(input_features, prefecture_names)

    # split the data into train, val, test
    data = split_dataset(X, X_raw, timestamps, num_timesteps_input, num_timesteps_output, output_feature_index, split_ratio)
    #data = split_dataset(X, X_raw, timestamps, num_timesteps_input, num_timesteps_output, split_ratio)

    return data, W, means, stds, prefecture_names

def train_model(data, W, num_timesteps_input, num_timesteps_output, dropout_rate, learning_rate, epochs, batch_size, device, verbose=True):

    train_input = data['train_input'] 
    train_target = data['train_target'] 
    val_input = data['val_input'] 
    val_target = data['val_target']

    W = torch.from_numpy(W)
    W = W.to(device=device)

    # Input shape (num_nodes, num_features, num_timestamp_input, num_timestamp_output)
    # Output shape (num_nodes, num_timestamp_output)

    model = STGCN(W.shape[0], # nodes,
                train_input.shape[3], # features
                num_timesteps_input, 
                num_timesteps_output,
                dropout_rate=dropout_rate).to(device=device) 

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_criterion = nn.MSELoss(reduction='mean')

    training_losses = []
    validation_losses = []

    for epoch in range(epochs):

        permutation = torch.randperm(train_input.shape[0])

        epoch_training_losses = []
        for i in range(0, train_input.shape[0], batch_size):
            model.train() # Set model to training mode
            optimizer.zero_grad() # Reset gradients

            indices = permutation[i:i + batch_size]
            X_batch, y_batch = train_input[indices], train_target[indices]
            X_batch = X_batch.to(device=device)
            y_batch = y_batch.to(device=device)


            prediction = model(W, X_batch) # Forward propagation
            loss = loss_criterion(prediction, y_batch)
            loss.backward() # Backpropagation, Compute gradients
            optimizer.step() # Update weights and biases
            epoch_training_losses.append(loss.detach().cpu().numpy())

        loss = sum(epoch_training_losses)/len(epoch_training_losses)

        training_losses.append(loss)

        # Run validation
        with torch.no_grad(): # Disable gradient calculation for evaluation
            model.eval() # Set model to evaluation mode
            val_input = val_input.to(device=device)
            val_target = val_target.to(device=device)

            prediction = model(W, val_input)
            val_loss = loss_criterion(prediction, val_target).to(device="cpu")
            validation_losses.append(val_loss.detach().numpy().item())

        if epoch % 10 == 0 and verbose:
            print("epochs: ", epoch, 
                "Training loss: {}".format(training_losses[-1]),
                "Validation loss: {}".format(validation_losses[-1]))
            
    return model, training_losses, validation_losses


def unnormalize(output, feature_means: float, feature_stds: float) -> torch.Tensor:
    '''
    Args:
        output: target or prediction array of shape (num_timesteps, num_nodes, num_features)
        feature_means: mean of the output feature
        feature_stds: std of the output feature
    Returns:
        numpy array of shape (num_timesteps, num_nodes, num_features)
    '''
    return output*feature_stds+feature_means


def evaluate_model(model, data, W, means, stds, output_feature_index, device) -> dict:
#def evaluate_model(model, data, W, means, stds, num_timesteps_input, device, prefecture_names) -> dict:
    '''
    Args:
        model: trained model to evaluate
        data: dictionary containing train, val, test data
            test_input: features of shape (num_samples, num_nodes, num_timesteps_input, num_features) - first num_timesteps_input days x num_samples
            test_target: target of shape (num_samples, num_nodes, 1, num_timesteps_output) - followed num_timesteps_output days x num_samples
            test_timestamps: timestamps of the test data
        W: weighted adjacency matrix of shape (num_nodes, num_nodes)
        means: means of each feature (num_features,)
        stds: std of each feature (num_features,)
        output_feature_index: output feature as the index of the input features list 
        num_timesteps_input: input time window size as the number of timesteps
        device: device to run the model, default is cpu
    Return:
        input_unnormalized: unnormalized input of shape (num_samples, num_nodes, num_timesteps_input, num_features)
        prediction_unnormalized: unnormalized prediction of shape (num_samples, num_nodes, 1, num_timesteps_output)
        target_unnormalized: unnormalized target of shape (num_samples, num_nodes, 1, num_timesteps_output)
        results: dictionary containing prefecture_index, prefecture_name, predictions, targets, timestamps
    '''        
    test_input = data['test_input'] 
    test_target = data['test_target'] 
    test_timestamps = data['test_timestamps']

    W = torch.from_numpy(W)
    W = W.to(device=device)

    # Evaluate the model on the test set
    with torch.no_grad(): # Disable gradient calculation for evaluation
        model.eval() # Set model to evaluation mode
        test_input = test_input.to(device=device)
        test_target = test_target.to(device=device)

        prediction = model(W, test_input)

        prediction_unnormalized = unnormalize(prediction.detach().cpu().numpy(), means[output_feature_index], stds[output_feature_index])
        target_unnormalized = unnormalize(test_target.detach().cpu().numpy(), means[output_feature_index], stds[output_feature_index])
        input_unnormalized = unnormalize(test_input.detach().cpu().numpy(), 
                                                means[output_feature_index], stds[output_feature_index])
        
        #print('test_prediction_unnormalized.shape', prediction_unnormalized.shape)
        #print('test_target_unnormalized.shape', target_unnormalized.shape)
        #print('test_input_unnormalized.shape', input_unnormalized.shape) 


    return test_input, test_target, test_timestamps, input_unnormalized, prediction_unnormalized, target_unnormalized

def run_model(input_features, output_feature_index, num_timesteps_input, num_timesteps_output, dropout_rate, learning_rate, epochs, batch_size, device, 
              verbose=True, split_ratio: tuple = (0.6, 0.8)):
#def run_model(input_features, num_timesteps_input, num_timesteps_output, dropout_rate, learning_rate, epochs, batch_size, device, 
#              verbose=True, split_ratio: tuple = (0.6, 0.8)):

    data, W, means, stds, prefecture_names = \
        prep_data(input_features, output_feature_index, num_timesteps_input, num_timesteps_output, split_ratio)
        #prep_data(input_features, num_timesteps_input, num_timesteps_output, split_ratio)
    model, training_losses, validation_losses = \
        train_model(data, W, num_timesteps_input, num_timesteps_output, dropout_rate, learning_rate, epochs, batch_size, device, verbose)
    test_input, test_target, test_timestamps, input_unnormalized, prediction_unnormalized, target_unnormalized = \
        evaluate_model(model, data, W, means, stds, output_feature_index, device)
    
    W = torch.from_numpy(W)

    test_data = {
        'input': test_input,
        'target': test_target,
        'input_unnormalized': input_unnormalized,
        'prediction_unnormalized': prediction_unnormalized,
        'target_unnormalized': target_unnormalized,
        'timestamps': test_timestamps,
    }

    return model, W, prefecture_names, means, stds, training_losses, validation_losses, test_data, data
    #return training_losses, validation_losses, means, stds, model, W

